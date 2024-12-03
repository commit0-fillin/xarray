from __future__ import annotations
import functools
import itertools
import math
import warnings
from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
import numpy as np
from packaging.version import Version
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.arithmetic import CoarsenArithmetic
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import CoarsenBoundaryOptions, SideOptions, T_Xarray
from xarray.core.utils import either_dict_or_kwargs, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
try:
    import bottleneck
except ImportError:
    bottleneck = None
if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    RollingKey = Any
    _T = TypeVar('_T')
_ROLLING_REDUCE_DOCSTRING_TEMPLATE = "Reduce this object's data windows by applying `{name}` along its dimension.\n\nParameters\n----------\nkeep_attrs : bool, default: None\n    If True, the attributes (``attrs``) will be copied from the original\n    object to the new one. If False, the new object will be returned\n    without attributes. If None uses the global default.\n**kwargs : dict\n    Additional keyword arguments passed on to `{name}`.\n\nReturns\n-------\nreduced : same type as caller\n    New object with `{name}` applied along its rolling dimension.\n"

class Rolling(Generic[T_Xarray]):
    """A object that implements the moving window pattern.

    See Also
    --------
    xarray.Dataset.groupby
    xarray.DataArray.groupby
    xarray.Dataset.rolling
    xarray.DataArray.rolling
    """
    __slots__ = ('obj', 'window', 'min_periods', 'center', 'dim')
    _attributes = ('window', 'min_periods', 'center', 'dim')
    dim: list[Hashable]
    window: list[int]
    center: list[bool]
    obj: T_Xarray
    min_periods: int

    def __init__(self, obj: T_Xarray, windows: Mapping[Any, int], min_periods: int | None=None, center: bool | Mapping[Any, bool]=False) -> None:
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            window along (e.g. `time`) to the size of the moving window.
        min_periods : int or None, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or dict-like Hashable to bool, default: False
            Set the labels at the center of the window. If dict-like, set this
            property per rolling dimension.

        Returns
        -------
        rolling : type of input argument
        """
        self.dim = []
        self.window = []
        for d, w in windows.items():
            self.dim.append(d)
            if w <= 0:
                raise ValueError('window must be > 0')
            self.window.append(w)
        self.center = self._mapping_to_list(center, default=False)
        self.obj = obj
        missing_dims = tuple((dim for dim in self.dim if dim not in self.obj.dims))
        if missing_dims:
            raise KeyError(f'Window dimensions {missing_dims} not found in {self.obj.__class__.__name__} dimensions {tuple(self.obj.dims)}')
        if min_periods is not None and min_periods <= 0:
            raise ValueError('min_periods must be greater than zero or None')
        self.min_periods = math.prod(self.window) if min_periods is None else min_periods

    def __repr__(self) -> str:
        """provide a nice str repr of our rolling object"""
        attrs = ['{k}->{v}{c}'.format(k=k, v=w, c='(center)' if c else '') for k, w, c in zip(self.dim, self.window, self.center)]
        return '{klass} [{attrs}]'.format(klass=self.__class__.__name__, attrs=','.join(attrs))

    def __len__(self) -> int:
        return math.prod((self.obj.sizes[d] for d in self.dim))

    def _reduce_method(name: str, fillna: Any, rolling_agg_func: Callable | None=None) -> Callable[..., T_Xarray]:
        """Constructs reduction methods built on a numpy reduction function (e.g. sum),
        a numbagg reduction function (e.g. move_sum), a bottleneck reduction function
        (e.g. move_sum), or a Rolling reduction (_mean).

        The logic here for which function to run is quite diffuse, across this method &
        _array_reduce. Arguably we could refactor this. But one constraint is that we
        need context of xarray options, of the functions each library offers, of
        the array (e.g. dtype).
        """
        def func(self, keep_attrs: bool | None = None, **kwargs):
            from xarray.core.computation import apply_ufunc
            rolling_agg = rolling_agg_func or getattr(self, f'_{name}')
            
            keep_attrs = _get_keep_attrs(keep_attrs)
            return apply_ufunc(
                rolling_agg,
                self.obj,
                input_core_dims=[self.dim],
                kwargs=dict(window=self.window, 
                            center=self.center, 
                            min_periods=self.min_periods,
                            **kwargs),
                keep_attrs=keep_attrs,
                dask="allowed",
            ).fillna(fillna)
        
        func.__name__ = name
        func.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=name)
        return func
    _mean.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name='mean')
    argmax = _reduce_method('argmax', dtypes.NINF)
    argmin = _reduce_method('argmin', dtypes.INF)
    max = _reduce_method('max', dtypes.NINF)
    min = _reduce_method('min', dtypes.INF)
    prod = _reduce_method('prod', 1)
    sum = _reduce_method('sum', 0)
    mean = _reduce_method('mean', None, _mean)
    std = _reduce_method('std', None)
    var = _reduce_method('var', None)
    median = _reduce_method('median', None)
    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name='count')

class DataArrayRolling(Rolling['DataArray']):
    __slots__ = ('window_labels',)

    def __init__(self, obj: DataArray, windows: Mapping[Any, int], min_periods: int | None=None, center: bool | Mapping[Any, bool]=False) -> None:
        """
        Moving window object for DataArray.
        You should use DataArray.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool, default: False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        xarray.DataArray.rolling
        xarray.DataArray.groupby
        xarray.Dataset.rolling
        xarray.Dataset.groupby
        """
        super().__init__(obj, windows, min_periods=min_periods, center=center)
        self.window_labels = self.obj[self.dim[0]]

    def __iter__(self) -> Iterator[tuple[DataArray, DataArray]]:
        if self.ndim > 1:
            raise ValueError('__iter__ is only supported for 1d-rolling')
        dim0 = self.dim[0]
        window0 = int(self.window[0])
        offset = (window0 + 1) // 2 if self.center[0] else 1
        stops = np.arange(offset, self.obj.sizes[dim0] + offset)
        starts = stops - window0
        starts[:window0 - offset] = 0
        for label, start, stop in zip(self.window_labels, starts, stops):
            window = self.obj.isel({dim0: slice(start, stop)})
            counts = window.count(dim=[dim0])
            window = window.where(counts >= self.min_periods)
            yield (label, window)

    def construct(self, window_dim: Hashable | Mapping[Any, Hashable] | None=None, stride: int | Mapping[Any, int]=1, fill_value: Any=dtypes.NA, keep_attrs: bool | None=None, **window_dim_kwargs: Hashable) -> DataArray:
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : Hashable or dict-like to Hashable, optional
            A mapping from dimension name to the new window dimension names.
        stride : int or mapping of int, default: 1
            Size of stride for the rolling window.
        fill_value : default: dtypes.NA
            Filling value to match the dimension size.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **window_dim_kwargs : Hashable, optional
            The keyword arguments form of ``window_dim`` {dim: new_name, ...}.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))

        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling = da.rolling(b=3, center=True)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        Dimensions without coordinates: a, b, window_dim

        """
        from xarray.core.dataarray import DataArray
        
        window_dim = either_dict_or_kwargs(window_dim, window_dim_kwargs, "construct")
        
        if len(window_dim) != len(self.dim):
            raise ValueError(
                f"window_dim has length {len(window_dim)}, "
                f"but rolling window has dimensions {self.dim}"
            )

        window = {d: self.window[i] for i, d in enumerate(self.dim)}
        center = {d: self.center[i] for i, d in enumerate(self.dim)}

        data = self.obj.data
        dims = self.obj.dims
        coords = self.obj.coords.copy()

        for dim, name in window_dim.items():
            if dim not in dims:
                raise ValueError(f"Dimension {dim} not found in DataArray dimensions")

            coord = coords[dim]
            axis = dims.index(dim)

            if isinstance(stride, Mapping):
                dim_stride = stride.get(dim, 1)
            else:
                dim_stride = stride

            shape = list(data.shape)
            shape[axis] = shape[axis] - window[dim] + 1
            shape.insert(axis + 1, window[dim])

            strides = list(data.strides)
            strides.insert(axis + 1, data.strides[axis])

            data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
            data = data[tuple(slice(None, None, dim_stride) if i != axis + 1 else slice(None) for i in range(data.ndim))]

            if center.get(dim, False):
                pad = (window[dim] - 1) // 2
                newcoord = coord[pad:-pad:dim_stride] if pad > 0 else coord[::dim_stride]
            else:
                newcoord = coord[:-window[dim]+1:dim_stride]

            coords[dim] = newcoord
            coords[name] = DataArray(np.arange(window[dim]), dims=name, name=name)

            dims = dims[:axis+1] + (name,) + dims[axis+1:]

        result = DataArray(data, coords=coords, dims=dims, name=self.obj.name)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        if keep_attrs:
            result.attrs = self.obj.attrs
        return result.fillna(fill_value)

    def reduce(self, func: Callable, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 4)> Size: 64B
        array([[nan, nan,  3.,  6.],
               [nan, nan, 15., 18.]])
        Dimensions without coordinates: a, b

        >>> rolling = da.rolling(b=3, min_periods=1)
        >>> rolling.reduce(np.nansum)
        <xarray.DataArray (a: 2, b: 4)> Size: 64B
        array([[ 0.,  1.,  3.,  6.],
               [ 4.,  9., 15., 18.]])
        Dimensions without coordinates: a, b
        """
        from xarray.core.computation import apply_ufunc
        
        keep_attrs = _get_keep_attrs(keep_attrs)
        
        def rolling_reduce(x):
            return self._array_reduce(x, func, **kwargs)
        
        return apply_ufunc(
            rolling_reduce,
            self.obj,
            input_core_dims=[self.dim],
            keep_attrs=keep_attrs,
            dask="allowed",
        )

    def _counts(self, keep_attrs: bool | None) -> DataArray:
        """Number of non-nan entries in each rolling window."""
        from xarray.core.computation import apply_ufunc
        
        keep_attrs = _get_keep_attrs(keep_attrs)
        
        def count_func(x):
            return np.sum(~np.isnan(x), axis=-1)
        
        return apply_ufunc(
            count_func,
            self.obj,
            input_core_dims=[self.dim],
            kwargs=dict(
                window=self.window,
                center=self.center,
                min_periods=self.min_periods,
            ),
            keep_attrs=keep_attrs,
            dask="allowed",
        )

class DatasetRolling(Rolling['Dataset']):
    __slots__ = ('rollings',)

    def __init__(self, obj: Dataset, windows: Mapping[Any, int], min_periods: int | None=None, center: bool | Mapping[Any, bool]=False) -> None:
        """
        Moving window object for Dataset.
        You should use Dataset.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : Dataset
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping of hashable to bool, default: False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        xarray.Dataset.rolling
        xarray.DataArray.rolling
        xarray.Dataset.groupby
        xarray.DataArray.groupby
        """
        super().__init__(obj, windows, min_periods, center)
        self.rollings = {}
        for key, da in self.obj.data_vars.items():
            dims, center = ([], {})
            for i, d in enumerate(self.dim):
                if d in da.dims:
                    dims.append(d)
                    center[d] = self.center[i]
            if dims:
                w = {d: windows[d] for d in dims}
                self.rollings[key] = DataArrayRolling(da, w, min_periods, center)

    def reduce(self, func: Callable, keep_attrs: bool | None=None, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Dataset with summarized data variables.
        """
        keep_attrs = _get_keep_attrs(keep_attrs)
        reduced = {}
        
        for key, da in self.obj.data_vars.items():
            if key in self.rollings:
                reduced[key] = self.rollings[key].reduce(func, keep_attrs=keep_attrs, **kwargs)
            else:
                reduced[key] = da
        
        return self.obj._replace(reduced, attrs=(self.obj.attrs if keep_attrs else None))

    def construct(self, window_dim: Hashable | Mapping[Any, Hashable] | None=None, stride: int | Mapping[Any, int]=1, fill_value: Any=dtypes.NA, keep_attrs: bool | None=None, **window_dim_kwargs: Hashable) -> Dataset:
        """
        Convert this rolling object to xr.Dataset,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
            Just a string can be used for 1d-rolling.
        stride : int, optional
            size of stride for the rolling window.
        fill_value : Any, default: dtypes.NA
            Filling value to match the dimension size.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        Dataset with variables converted from rolling object.
        """
        keep_attrs = _get_keep_attrs(keep_attrs)
        window_dim = either_dict_or_kwargs(window_dim, window_dim_kwargs, "construct")
        
        if len(self.dim) == 1 and len(window_dim) == 0:
            [d] = self.dim
            window_dim = {d: f"{d}_window"}
        
        dataset = {}
        for key, da in self.obj.data_vars.items():
            if key in self.rollings:
                dataset[key] = self.rollings[key].construct(
                    window_dim=window_dim,
                    stride=stride,
                    fill_value=fill_value,
                    keep_attrs=keep_attrs
                )
            else:
                dataset[key] = da
        
        return self.obj._replace(dataset, attrs=(self.obj.attrs if keep_attrs else None))

class Coarsen(CoarsenArithmetic, Generic[T_Xarray]):
    """A object that implements the coarsen.

    See Also
    --------
    Dataset.coarsen
    DataArray.coarsen
    """
    __slots__ = ('obj', 'boundary', 'coord_func', 'windows', 'side', 'trim_excess')
    _attributes = ('windows', 'side', 'trim_excess')
    obj: T_Xarray
    windows: Mapping[Hashable, int]
    side: SideOptions | Mapping[Hashable, SideOptions]
    boundary: CoarsenBoundaryOptions
    coord_func: Mapping[Hashable, str | Callable]

    def __init__(self, obj: T_Xarray, windows: Mapping[Any, int], boundary: CoarsenBoundaryOptions, side: SideOptions | Mapping[Any, SideOptions], coord_func: str | Callable | Mapping[Any, str | Callable]) -> None:
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        boundary : {"exact", "trim", "pad"}
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of window size. If 'trim', the excess indexes are trimmed.
            If 'pad', NA will be padded.
        side : 'left' or 'right' or mapping from dimension to 'left' or 'right'
        coord_func : function (name) or mapping from coordinate name to function (name).

        Returns
        -------
        coarsen

        """
        self.obj = obj
        self.windows = windows
        self.side = side
        self.boundary = boundary
        missing_dims = tuple((dim for dim in windows.keys() if dim not in self.obj.dims))
        if missing_dims:
            raise ValueError(f'Window dimensions {missing_dims} not found in {self.obj.__class__.__name__} dimensions {tuple(self.obj.dims)}')
        if utils.is_dict_like(coord_func):
            coord_func_map = coord_func
        else:
            coord_func_map = {d: coord_func for d in self.obj.dims}
        for c in self.obj.coords:
            if c not in coord_func_map:
                coord_func_map[c] = duck_array_ops.mean
        self.coord_func = coord_func_map

    def __repr__(self) -> str:
        """provide a nice str repr of our coarsen object"""
        attrs = [f'{k}->{getattr(self, k)}' for k in self._attributes if getattr(self, k, None) is not None]
        return '{klass} [{attrs}]'.format(klass=self.__class__.__name__, attrs=','.join(attrs))

    def construct(self, window_dim=None, keep_attrs=None, **window_dim_kwargs) -> T_Xarray:
        """
        Convert this Coarsen object to a DataArray or Dataset,
        where the coarsening dimension is split or reshaped to two
        new dimensions.

        Parameters
        ----------
        window_dim: mapping
            A mapping from existing dimension name to new dimension names.
            The size of the second dimension will be the length of the
            coarsening window.
        keep_attrs: bool, optional
            Preserve attributes if True
        **window_dim_kwargs : {dim: new_name, ...}
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        Dataset or DataArray with reshaped dimensions

        Examples
        --------
        >>> da = xr.DataArray(np.arange(24), dims="time")
        >>> da.coarsen(time=12).construct(time=("year", "month"))
        <xarray.DataArray (year: 2, month: 12)> Size: 192B
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        Dimensions without coordinates: year, month

        See Also
        --------
        DataArrayRolling.construct
        DatasetRolling.construct
        """
        pass

class DataArrayCoarsen(Coarsen['DataArray']):
    __slots__ = ()
    _reduce_extra_args_docstring = ''

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool=False, numeric_only: bool=False) -> Callable[..., DataArray]:
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        def wrapped_func(self: DataArrayCoarsen, keep_attrs: bool | None = None, **kwargs):
            from xarray.core.computation import apply_ufunc
            
            keep_attrs = _get_keep_attrs(keep_attrs)
            
            if include_skipna:
                skipna = kwargs.pop("skipna", None)
                if skipna is not None:
                    func_kwargs = {"skipna": skipna}
                else:
                    func_kwargs = {}
            else:
                func_kwargs = {}
            
            func_kwargs.update(kwargs)
            
            return apply_ufunc(
                func,
                self.obj,
                input_core_dims=[self.dim],
                kwargs=dict(
                    dim=self.dim,
                    windows=self.windows,
                    func_kwargs=func_kwargs,
                    keep_attrs=keep_attrs,
                ),
                dask="allowed",
            )
        
        return wrapped_func

    def reduce(self, func: Callable, keep_attrs: bool | None=None, **kwargs) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, axis, **kwargs)`
            to return the result of collapsing an np.ndarray over the coarsening
            dimensions.  It must be possible to provide the `axis` argument
            with a tuple of integers.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> coarsen = da.coarsen(b=2)
        >>> coarsen.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 2)> Size: 32B
        array([[ 1,  5],
               [ 9, 13]])
        Dimensions without coordinates: a, b
        """
        from xarray.core.computation import apply_ufunc
        
        keep_attrs = _get_keep_attrs(keep_attrs)
        
        def wrapped_func(data):
            axes = [self.obj.get_axis_num(dim) for dim in self.dim]
            return func(data, axis=tuple(axes), **kwargs)
        
        result = apply_ufunc(
            wrapped_func,
            self.obj,
            input_core_dims=[self.dim],
            output_core_dims=[[]],
            kwargs={"windows": self.windows},
            keep_attrs=keep_attrs,
            dask="allowed",
        )
        
        new_coords = {}
        for coord_name, coord in self.obj.coords.items():
            if coord_name in self.dim:
                new_coords[coord_name] = coord[::self.windows[coord_name]]
            else:
                new_coords[coord_name] = coord
        
        return result.assign_coords(new_coords)

class DatasetCoarsen(Coarsen['Dataset']):
    __slots__ = ()
    _reduce_extra_args_docstring = ''

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool=False, numeric_only: bool=False) -> Callable[..., Dataset]:
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        def wrapped_func(self: DatasetCoarsen, keep_attrs: bool | None = None, **kwargs):
            from xarray.core.computation import apply_ufunc
            
            keep_attrs = _get_keep_attrs(keep_attrs)
            
            if include_skipna:
                skipna = kwargs.pop("skipna", None)
                if skipna is not None:
                    func_kwargs = {"skipna": skipna}
                else:
                    func_kwargs = {}
            else:
                func_kwargs = {}
            
            func_kwargs.update(kwargs)
            
            reduced = {}
            for name, var in self.obj.data_vars.items():
                if numeric_only and not np.issubdtype(var.dtype, np.number):
                    continue
                
                reduced[name] = apply_ufunc(
                    func,
                    var,
                    input_core_dims=[self.dim],
                    kwargs=dict(
                        dim=self.dim,
                        windows=self.windows,
                        func_kwargs=func_kwargs,
                        keep_attrs=keep_attrs,
                    ),
                    dask="allowed",
                )
            
            return self.obj._replace(reduced, attrs=(self.obj.attrs if keep_attrs else None))
        
        return wrapped_func

    def reduce(self, func: Callable, keep_attrs=None, **kwargs) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, axis, **kwargs)`
            to return the result of collapsing an np.ndarray over the coarsening
            dimensions.  It must be possible to provide the `axis` argument with
            a tuple of integers.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Arrays with summarized data.
        """
        from xarray.core.computation import apply_ufunc
        
        keep_attrs = _get_keep_attrs(keep_attrs)
        
        def wrapped_func(data):
            axes = [self.obj.get_axis_num(dim) for dim in self.dim]
            return func(data, axis=tuple(axes), **kwargs)
        
        reduced = {}
        for name, var in self.obj.data_vars.items():
            reduced[name] = apply_ufunc(
                wrapped_func,
                var,
                input_core_dims=[self.dim],
                output_core_dims=[[]],
                kwargs={"windows": self.windows},
                keep_attrs=keep_attrs,
                dask="allowed",
            )
        
        new_coords = {}
        for coord_name, coord in self.obj.coords.items():
            if coord_name in self.dim:
                new_coords[coord_name] = coord[::self.windows[coord_name]]
            else:
                new_coords[coord_name] = coord
        
        result = self.obj._replace(reduced, coords=new_coords)
        if keep_attrs:
            result.attrs = self.obj.attrs
        
        return result
