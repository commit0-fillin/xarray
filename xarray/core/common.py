from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import Frozen, either_dict_or_kwargs, is_scalar
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
try:
    import cftime
except ImportError:
    cftime = None
ALL_DIMS = ...
if TYPE_CHECKING:
    import datetime
    from numpy.typing import DTypeLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.indexes import Index
    from xarray.core.resample import Resample
    from xarray.core.rolling_exp import RollingExp
    from xarray.core.types import DatetimeLike, DTypeLikeSave, ScalarOrArray, Self, SideOptions, T_Chunks, T_DataWithCoords, T_Variable
    from xarray.core.variable import Variable
    from xarray.groupers import Resampler
    DTypeMaybeMapping = Union[DTypeLikeSave, Mapping[Any, DTypeLikeSave]]
T_Resample = TypeVar('T_Resample', bound='Resample')
C = TypeVar('C')
T = TypeVar('T')

class ImplementsArrayReduce:
    __slots__ = ()
    _reduce_extra_args_docstring = dedent("        dim : str or sequence of str, optional\n            Dimension(s) over which to apply `{name}`.\n        axis : int or sequence of int, optional\n            Axis(es) over which to apply `{name}`. Only one of the 'dim'\n            and 'axis' arguments can be supplied. If neither are supplied, then\n            `{name}` is calculated over axes.")
    _cum_extra_args_docstring = dedent("        dim : str or sequence of str, optional\n            Dimension over which to apply `{name}`.\n        axis : int or sequence of int, optional\n            Axis over which to apply `{name}`. Only one of the 'dim'\n            and 'axis' arguments can be supplied.")

class ImplementsDatasetReduce:
    __slots__ = ()
    _reduce_extra_args_docstring = dedent('\n        dim : str or sequence of str, optional\n            Dimension(s) over which to apply `{name}`.  By default `{name}` is\n            applied over all dimensions.\n        ').strip()
    _cum_extra_args_docstring = dedent("\n        dim : str or sequence of str, optional\n            Dimension over which to apply `{name}`.\n        axis : int or sequence of int, optional\n            Axis over which to apply `{name}`. Only one of the 'dim'\n            and 'axis' arguments can be supplied.\n        ").strip()

class AbstractArray:
    """Shared base class for DataArray and Variable."""
    __slots__ = ()

    def __bool__(self: Any) -> bool:
        return bool(self.values)

    def __float__(self: Any) -> float:
        return float(self.values)

    def __int__(self: Any) -> int:
        return int(self.values)

    def __complex__(self: Any) -> complex:
        return complex(self.values)

    def __array__(self: Any, dtype: DTypeLike | None=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def __format__(self: Any, format_spec: str='') -> str:
        if format_spec != '':
            if self.shape == ():
                return self.data.__format__(format_spec)
            else:
                raise NotImplementedError(f'Using format_spec is only supported when shape is (). Got shape = {self.shape}.')
        else:
            return self.__repr__()

    def __iter__(self: Any) -> Iterator[Any]:
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        return self._iter()

    def get_axis_num(self, dim: Hashable | Iterable[Hashable]) -> int | tuple[int, ...]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if isinstance(dim, Iterable) and not isinstance(dim, str):
            return tuple(self.get_axis_num(d) for d in dim)
        else:
            return self.dims.index(dim)

    @property
    def sizes(self: Any) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        Dataset.sizes
        """
        return Frozen(dict(zip(self.dims, self.shape)))

class AttrAccessMixin:
    """Mixin class that allows getting keys with attribute access"""
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        """Verify that all subclasses explicitly define ``__slots__``. If they don't,
        raise error in the core xarray module and a FutureWarning in third-party
        extensions.
        """
        if not hasattr(object.__new__(cls), '__dict__'):
            return
        elif cls.__module__.startswith('xarray.'):
            raise AttributeError(f'{cls.__name__} must explicitly define __slots__')
        else:
            cls.__setattr__ = cls._setattr_dict
            warnings.warn(f'xarray subclass {cls.__name__} should explicitly define __slots__', FutureWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        return (self.attrs,)

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-autocompletion"""
        return (self.coords, self.data_vars)

    def __getattr__(self, name: str) -> Any:
        if name not in {'__dict__', '__setstate__'}:
            for source in self._attr_sources:
                with suppress(KeyError):
                    return source[name]
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def _setattr_dict(self, name: str, value: Any) -> None:
        """Deprecated third party subclass (see ``__init_subclass__`` above)"""
        warnings.warn(
            f"Setting attribute {name!r} on a {type(self).__name__!r} object is deprecated. "
            "Use __setitem__ style assignment instead.",
            FutureWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Objects with ``__slots__`` raise AttributeError if you try setting an
        undeclared attribute. This is desirable, but the error message could use some
        improvement.
        """
        try:
            object.__setattr__(self, name, value)
        except AttributeError as e:
            if str(e) != f'{type(self).__name__!r} object has no attribute {name!r}':
                raise
            raise AttributeError(f"cannot set attribute {name!r} on a {type(self).__name__!r} object. Use __setitem__ styleassignment (e.g., `ds['name'] = ...`) instead of assigning variables.") from e

    def __dir__(self) -> list[str]:
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = {item for source in self._attr_sources for item in source if isinstance(item, str)}
        return sorted(set(dir(type(self))) | extra_attrs)

    def _ipython_key_completions_(self) -> list[str]:
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self.keys())

class TreeAttrAccessMixin(AttrAccessMixin):
    """Mixin class that allows getting keys with attribute access"""
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        """This method overrides the check from ``AttrAccessMixin`` that ensures
        ``__dict__`` is absent in a class, with ``__slots__`` used instead.
        ``DataTree`` has some dynamically defined attributes in addition to those
        defined in ``__slots__``. (GH9068)
        """
        if not hasattr(object.__new__(cls), '__dict__'):
            return
        super().__init_subclass__(**kwargs)

def get_squeeze_dims(xarray_obj, dim: Hashable | Iterable[Hashable] | None=None, axis: int | Iterable[int] | None=None) -> list[Hashable]:
    """Get a list of dimensions to squeeze out."""
    if dim is not None and axis is not None:
        raise ValueError("cannot specify both 'dim' and 'axis'")
    if dim is None and axis is None:
        return [d for d, s in xarray_obj.sizes.items() if s == 1]
    if isinstance(dim, Hashable) and not isinstance(dim, Iterable):
        dim = [dim]
    if isinstance(axis, int):
        axis = [axis]
    if dim is not None:
        return [d for d in dim if xarray_obj.sizes[d] == 1]
    if axis is not None:
        return [xarray_obj.dims[a] for a in axis if xarray_obj.shape[a] == 1]
    return []

class DataWithCoords(AttrAccessMixin):
    """Shared base class for Dataset and DataArray."""
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index]
    __slots__ = ('_close',)

    def squeeze(self, dim: Hashable | Iterable[Hashable] | None=None, drop: bool=False, axis: int | Iterable[int] | None=None) -> Self:
        """Return a new object with squeezed data.

        Parameters
        ----------
        dim : None or Hashable or iterable of Hashable, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.
        drop : bool, default: False
            If ``drop=True``, drop squeezed coordinates instead of making them
            scalar.
        axis : None or int or iterable of int, optional
            Like dim, but positional.

        Returns
        -------
        squeezed : same type as caller
            This object, but with with all or a subset of the dimensions of
            length 1 removed.

        See Also
        --------
        numpy.squeeze
        """
        pass

    def clip(self, min: ScalarOrArray | None=None, max: ScalarOrArray | None=None, *, keep_attrs: bool | None=None) -> Self:
        """
        Return an array whose values are limited to ``[min, max]``.
        At least one of max or min must be given.

        Parameters
        ----------
        min : None or Hashable, optional
            Minimum value. If None, no lower clipping is performed.
        max : None or Hashable, optional
            Maximum value. If None, no upper clipping is performed.
        keep_attrs : bool or None, optional
            If True, the attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new
            object will be returned without attributes.

        Returns
        -------
        clipped : same type as caller
            This object, but with with values < min are replaced with min,
            and those > max with max.

        See Also
        --------
        numpy.clip : equivalent function
        """
        pass

    def get_index(self, key: Hashable) -> pd.Index:
        """Get an index for a dimension, with fall-back to a default RangeIndex"""
        pass

    def assign_coords(self, coords: Mapping | None=None, **coords_kwargs: Any) -> Self:
        """Assign new coordinates to this object.

        Returns a new object with all the original data in addition to the new
        coordinates.

        Parameters
        ----------
        coords : mapping of dim to coord, optional
            A mapping whose keys are the names of the coordinates and values are the
            coordinates to assign. The mapping will generally be a dict or
            :class:`Coordinates`.

            * If a value is a standard data value — for example, a ``DataArray``,
              scalar, or array — the data is simply assigned as a coordinate.

            * If a value is callable, it is called with this object as the only
              parameter, and the return value is used as new coordinate variables.

            * A coordinate can also be defined and attached to an existing dimension
              using a tuple with the first element the dimension name and the second
              element the values for this new coordinate.

        **coords_kwargs : optional
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the new coordinates in addition to the existing
            data.

        Examples
        --------
        Convert `DataArray` longitude coordinates from 0-359 to -180-179:

        >>> da = xr.DataArray(
        ...     np.random.rand(4),
        ...     coords=[np.array([358, 359, 0, 1])],
        ...     dims="lon",
        ... )
        >>> da
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B 358 359 0 1
        >>> da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B -2 -1 0 1

        The function also accepts dictionary arguments:

        >>> da.assign_coords({"lon": (((da.lon + 180) % 360) - 180)})
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B -2 -1 0 1

        New coordinate can also be attached to an existing dimension:

        >>> lon_2 = np.array([300, 289, 0, 1])
        >>> da.assign_coords(lon_2=("lon", lon_2))
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B 358 359 0 1
            lon_2    (lon) int64 32B 300 289 0 1

        Note that the same result can also be obtained with a dict e.g.

        >>> _ = da.assign_coords({"lon_2": ("lon", lon_2)})

        Note the same method applies to `Dataset` objects.

        Convert `Dataset` longitude coordinates from 0-359 to -180-179:

        >>> temperature = np.linspace(20, 32, num=16).reshape(2, 2, 4)
        >>> precipitation = 2 * np.identity(4).reshape(2, 2, 4)
        >>> ds = xr.Dataset(
        ...     data_vars=dict(
        ...         temperature=(["x", "y", "time"], temperature),
        ...         precipitation=(["x", "y", "time"], precipitation),
        ...     ),
        ...     coords=dict(
        ...         lon=(["x", "y"], [[260.17, 260.68], [260.21, 260.77]]),
        ...         lat=(["x", "y"], [[42.25, 42.21], [42.63, 42.59]]),
        ...         time=pd.date_range("2014-09-06", periods=4),
        ...         reference_time=pd.Timestamp("2014-09-05"),
        ...     ),
        ...     attrs=dict(description="Weather-related data"),
        ... )
        >>> ds
        <xarray.Dataset> Size: 360B
        Dimensions:         (x: 2, y: 2, time: 4)
        Coordinates:
            lon             (x, y) float64 32B 260.2 260.7 260.2 260.8
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 32B 2014-09-06 ... 2014-09-09
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 128B 20.0 20.8 21.6 ... 30.4 31.2 32.0
            precipitation   (x, y, time) float64 128B 2.0 0.0 0.0 0.0 ... 0.0 0.0 2.0
        Attributes:
            description:  Weather-related data
        >>> ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        <xarray.Dataset> Size: 360B
        Dimensions:         (x: 2, y: 2, time: 4)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 32B 2014-09-06 ... 2014-09-09
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 128B 20.0 20.8 21.6 ... 30.4 31.2 32.0
            precipitation   (x, y, time) float64 128B 2.0 0.0 0.0 0.0 ... 0.0 0.0 2.0
        Attributes:
            description:  Weather-related data

        See Also
        --------
        Dataset.assign
        Dataset.swap_dims
        Dataset.set_coords
        """
        pass

    def assign_attrs(self, *args: Any, **kwargs: Any) -> Self:
        """Assign new attrs to this object.

        Returns a new object equivalent to ``self.attrs.update(*args, **kwargs)``.

        Parameters
        ----------
        *args
            positional arguments passed into ``attrs.update``.
        **kwargs
            keyword arguments passed into ``attrs.update``.

        Examples
        --------
        >>> dataset = xr.Dataset({"temperature": [25, 30, 27]})
        >>> dataset
        <xarray.Dataset> Size: 24B
        Dimensions:      (temperature: 3)
        Coordinates:
          * temperature  (temperature) int64 24B 25 30 27
        Data variables:
            *empty*

        >>> new_dataset = dataset.assign_attrs(
        ...     units="Celsius", description="Temperature data"
        ... )
        >>> new_dataset
        <xarray.Dataset> Size: 24B
        Dimensions:      (temperature: 3)
        Coordinates:
          * temperature  (temperature) int64 24B 25 30 27
        Data variables:
            *empty*
        Attributes:
            units:        Celsius
            description:  Temperature data

        # Attributes of the new dataset

        >>> new_dataset.attrs
        {'units': 'Celsius', 'description': 'Temperature data'}

        Returns
        -------
        assigned : same type as caller
            A new object with the new attrs in addition to the existing data.

        See Also
        --------
        Dataset.assign
        """
        pass

    def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args: Any, **kwargs: Any) -> T:
        """
        Apply ``func(self, *args, **kwargs)``

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : callable
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        *args
            positional arguments passed into ``func``.
        **kwargs
            a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : Any
            the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        .. code:: python

            f(g(h(ds), arg1=a), arg2=b, arg3=c)

        You can write

        .. code:: python

            (ds.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        .. code:: python

            (ds.pipe(h).pipe(g, arg1=a).pipe((f, "arg2"), arg1=a, arg3=c))

        Examples
        --------
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset> Size: 96B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 32B 0.4237 0.6459 0.4376 0.8918

        >>> def adder(data, arg):
        ...     return data + arg
        ...
        >>> def div(data, arg):
        ...     return data / arg
        ...
        >>> def sub_mult(data, sub_arg, mult_arg):
        ...     return (data * mult_arg) - sub_arg
        ...
        >>> x.pipe(adder, 2)
        <xarray.Dataset> Size: 96B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 12.98 16.3 14.06 12.9
            precipitation  (lat, lon) float64 32B 2.424 2.646 2.438 2.892

        >>> x.pipe(adder, arg=2)
        <xarray.Dataset> Size: 96B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 12.98 16.3 14.06 12.9
            precipitation  (lat, lon) float64 32B 2.424 2.646 2.438 2.892

        >>> (
        ...     x.pipe(adder, arg=2)
        ...     .pipe(div, arg=2)
        ...     .pipe(sub_mult, sub_arg=2, mult_arg=2)
        ... )
        <xarray.Dataset> Size: 96B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 32B 0.4237 0.6459 0.4376 0.8918

        See Also
        --------
        pandas.DataFrame.pipe
        """
        pass

    def rolling_exp(self: T_DataWithCoords, window: Mapping[Any, int] | None=None, window_type: str='span', **window_kwargs) -> RollingExp[T_DataWithCoords]:
        """
        Exponentially-weighted moving window.
        Similar to EWM in pandas

        Requires the optional Numbagg dependency.

        Parameters
        ----------
        window : mapping of hashable to int, optional
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        window_type : {"span", "com", "halflife", "alpha"}, default: "span"
            The format of the previously supplied window. Each is a simple
            numerical transformation of the others. Described in detail:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        **window_kwargs : optional
            The keyword arguments form of ``window``.
            One of window or window_kwargs must be provided.

        See Also
        --------
        core.rolling_exp.RollingExp
        """
        pass

    def _resample(self, resample_cls: type[T_Resample], indexer: Mapping[Hashable, str | Resampler] | None, skipna: bool | None, closed: SideOptions | None, label: SideOptions | None, offset: pd.Timedelta | datetime.timedelta | str | None, origin: str | DatetimeLike, restore_coord_dims: bool | None, **indexer_kwargs: str | Resampler) -> T_Resample:
        """Returns a Resample object for performing resampling operations.

        Handles both downsampling and upsampling. The resampled
        dimension must be a datetime-like coordinate. If any intervals
        contain no values from the original object, they will be given
        the value ``NaN``.

        Parameters
        ----------
        indexer : {dim: freq}, optional
            Mapping from the dimension name to resample frequency [1]_. The
            dimension must be datetime-like.
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : {"left", "right"}, optional
            Side of each interval to treat as closed.
        label : {"left", "right"}, optional
            Side of each interval to use for labeling.
        origin : {'epoch', 'start', 'start_day', 'end', 'end_day'}, pd.Timestamp, datetime.datetime, np.datetime64, or cftime.datetime, default 'start_day'
            The datetime on which to adjust the grouping. The timezone of origin
            must match the timezone of the index.

            If a datetime is not used, these values are also supported:
            - 'epoch': `origin` is 1970-01-01
            - 'start': `origin` is the first value of the timeseries
            - 'start_day': `origin` is the first day at midnight of the timeseries
            - 'end': `origin` is the last value of the timeseries
            - 'end_day': `origin` is the ceiling midnight of the last day
        offset : pd.Timedelta, datetime.timedelta, or str, default is None
            An offset timedelta added to the origin.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        **indexer_kwargs : {dim: freq}
            The keyword arguments form of ``indexer``.
            One of indexer or indexer_kwargs must be provided.

        Returns
        -------
        resampled : same type as caller
            This object resampled.

        Examples
        --------
        Downsample monthly time-series data to seasonal data:

        >>> da = xr.DataArray(
        ...     np.linspace(0, 11, num=12),
        ...     coords=[
        ...         pd.date_range(
        ...             "1999-12-15",
        ...             periods=12,
        ...             freq=pd.DateOffset(months=1),
        ...         )
        ...     ],
        ...     dims="time",
        ... )
        >>> da
        <xarray.DataArray (time: 12)> Size: 96B
        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
        Coordinates:
          * time     (time) datetime64[ns] 96B 1999-12-15 2000-01-15 ... 2000-11-15
        >>> da.resample(time="QS-DEC").mean()
        <xarray.DataArray (time: 4)> Size: 32B
        array([ 1.,  4.,  7., 10.])
        Coordinates:
          * time     (time) datetime64[ns] 32B 1999-12-01 2000-03-01 ... 2000-09-01

        Upsample monthly time-series data to daily data:

        >>> da.resample(time="1D").interpolate("linear")  # +doctest: ELLIPSIS
        <xarray.DataArray (time: 337)> Size: 3kB
        array([ 0.        ,  0.03225806,  0.06451613,  0.09677419,  0.12903226,
                0.16129032,  0.19354839,  0.22580645,  0.25806452,  0.29032258,
                0.32258065,  0.35483871,  0.38709677,  0.41935484,  0.4516129 ,
                0.48387097,  0.51612903,  0.5483871 ,  0.58064516,  0.61290323,
                0.64516129,  0.67741935,  0.70967742,  0.74193548,  0.77419355,
                0.80645161,  0.83870968,  0.87096774,  0.90322581,  0.93548387,
                0.96774194,  1.        ,  1.03225806,  1.06451613,  1.09677419,
                1.12903226,  1.16129032,  1.19354839,  1.22580645,  1.25806452,
                1.29032258,  1.32258065,  1.35483871,  1.38709677,  1.41935484,
                1.4516129 ,  1.48387097,  1.51612903,  1.5483871 ,  1.58064516,
                1.61290323,  1.64516129,  1.67741935,  1.70967742,  1.74193548,
                1.77419355,  1.80645161,  1.83870968,  1.87096774,  1.90322581,
                1.93548387,  1.96774194,  2.        ,  2.03448276,  2.06896552,
                2.10344828,  2.13793103,  2.17241379,  2.20689655,  2.24137931,
                2.27586207,  2.31034483,  2.34482759,  2.37931034,  2.4137931 ,
                2.44827586,  2.48275862,  2.51724138,  2.55172414,  2.5862069 ,
                2.62068966,  2.65517241,  2.68965517,  2.72413793,  2.75862069,
                2.79310345,  2.82758621,  2.86206897,  2.89655172,  2.93103448,
                2.96551724,  3.        ,  3.03225806,  3.06451613,  3.09677419,
                3.12903226,  3.16129032,  3.19354839,  3.22580645,  3.25806452,
        ...
                7.87096774,  7.90322581,  7.93548387,  7.96774194,  8.        ,
                8.03225806,  8.06451613,  8.09677419,  8.12903226,  8.16129032,
                8.19354839,  8.22580645,  8.25806452,  8.29032258,  8.32258065,
                8.35483871,  8.38709677,  8.41935484,  8.4516129 ,  8.48387097,
                8.51612903,  8.5483871 ,  8.58064516,  8.61290323,  8.64516129,
                8.67741935,  8.70967742,  8.74193548,  8.77419355,  8.80645161,
                8.83870968,  8.87096774,  8.90322581,  8.93548387,  8.96774194,
                9.        ,  9.03333333,  9.06666667,  9.1       ,  9.13333333,
                9.16666667,  9.2       ,  9.23333333,  9.26666667,  9.3       ,
                9.33333333,  9.36666667,  9.4       ,  9.43333333,  9.46666667,
                9.5       ,  9.53333333,  9.56666667,  9.6       ,  9.63333333,
                9.66666667,  9.7       ,  9.73333333,  9.76666667,  9.8       ,
                9.83333333,  9.86666667,  9.9       ,  9.93333333,  9.96666667,
               10.        , 10.03225806, 10.06451613, 10.09677419, 10.12903226,
               10.16129032, 10.19354839, 10.22580645, 10.25806452, 10.29032258,
               10.32258065, 10.35483871, 10.38709677, 10.41935484, 10.4516129 ,
               10.48387097, 10.51612903, 10.5483871 , 10.58064516, 10.61290323,
               10.64516129, 10.67741935, 10.70967742, 10.74193548, 10.77419355,
               10.80645161, 10.83870968, 10.87096774, 10.90322581, 10.93548387,
               10.96774194, 11.        ])
        Coordinates:
          * time     (time) datetime64[ns] 3kB 1999-12-15 1999-12-16 ... 2000-11-15

        Limit scope of upsampling method

        >>> da.resample(time="1D").nearest(tolerance="1D")
        <xarray.DataArray (time: 337)> Size: 3kB
        array([ 0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan,  2.,  2.,  2., nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,
                3.,  3., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  4.,  4.,  4., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan,  5.,  5.,  5., nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                6.,  6.,  6., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  7.,  7.,  7., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan,  8.,  8.,  8., nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan,  9.,  9.,  9., nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, 10., 10., 10., nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 11., 11.])
        Coordinates:
          * time     (time) datetime64[ns] 3kB 1999-12-15 1999-12-16 ... 2000-11-15

        See Also
        --------
        pandas.Series.resample
        pandas.DataFrame.resample

        References
        ----------
        .. [1] https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects
        """
        pass

    def where(self, cond: Any, other: Any=dtypes.NA, drop: bool=False) -> Self:
        """Filter elements from this object according to a condition.

        Returns elements from 'DataArray', where 'cond' is True,
        otherwise fill in 'other'.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic.

        Parameters
        ----------
        cond : DataArray, Dataset, or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, the callable is passed this object, and the result is used as
            the value for cond.
        other : scalar, DataArray, Dataset, or callable, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, these locations are filled with NA. If a callable, it must
            expect this object as its only parameter.
        drop : bool, default: False
            If True, coordinate labels that only correspond to False values of
            the condition are dropped from the result.

        Returns
        -------
        DataArray or Dataset
            Same xarray type as caller, with dtype float64.

        Examples
        --------
        >>> a = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
        >>> a
        <xarray.DataArray (x: 5, y: 5)> Size: 200B
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 4)
        <xarray.DataArray (x: 5, y: 5)> Size: 200B
        array([[ 0.,  1.,  2.,  3., nan],
               [ 5.,  6.,  7., nan, nan],
               [10., 11., nan, nan, nan],
               [15., nan, nan, nan, nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 5, -1)
        <xarray.DataArray (x: 5, y: 5)> Size: 200B
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8, -1],
               [10, 11, 12, -1, -1],
               [15, 16, -1, -1, -1],
               [20, -1, -1, -1, -1]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 4, drop=True)
        <xarray.DataArray (x: 4, y: 4)> Size: 128B
        array([[ 0.,  1.,  2.,  3.],
               [ 5.,  6.,  7., nan],
               [10., 11., nan, nan],
               [15., nan, nan, nan]])
        Dimensions without coordinates: x, y

        >>> a.where(lambda x: x.x + x.y < 4, lambda x: -x)
        <xarray.DataArray (x: 5, y: 5)> Size: 200B
        array([[  0,   1,   2,   3,  -4],
               [  5,   6,   7,  -8,  -9],
               [ 10,  11, -12, -13, -14],
               [ 15, -16, -17, -18, -19],
               [-20, -21, -22, -23, -24]])
        Dimensions without coordinates: x, y

        >>> a.where(a.x + a.y < 4, drop=True)
        <xarray.DataArray (x: 4, y: 4)> Size: 128B
        array([[ 0.,  1.,  2.,  3.],
               [ 5.,  6.,  7., nan],
               [10., 11., nan, nan],
               [15., nan, nan, nan]])
        Dimensions without coordinates: x, y

        See Also
        --------
        numpy.where : corresponding numpy function
        where : equivalent function
        """
        pass

    def set_close(self, close: Callable[[], None] | None) -> None:
        """Register the function that releases any resources linked to this object.

        This method controls how xarray cleans up resources associated
        with this object when the ``.close()`` method is called. It is mostly
        intended for backend developers and it is rarely needed by regular
        end-users.

        Parameters
        ----------
        close : callable
            The function that when called like ``close()`` releases
            any resources linked to this object.
        """
        pass

    def close(self) -> None:
        """Release any resources linked to this object."""
        pass

    def isnull(self, keep_attrs: bool | None=None) -> Self:
        """Test each value in the array for whether it is a missing value.

        Parameters
        ----------
        keep_attrs : bool or None, optional
            If True, the attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new
            object will be returned without attributes.

        Returns
        -------
        isnull : DataArray or Dataset
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.isnull

        Examples
        --------
        >>> array = xr.DataArray([1, np.nan, 3], dims="x")
        >>> array
        <xarray.DataArray (x: 3)> Size: 24B
        array([ 1., nan,  3.])
        Dimensions without coordinates: x
        >>> array.isnull()
        <xarray.DataArray (x: 3)> Size: 3B
        array([False,  True, False])
        Dimensions without coordinates: x
        """
        pass

    def notnull(self, keep_attrs: bool | None=None) -> Self:
        """Test each value in the array for whether it is not a missing value.

        Parameters
        ----------
        keep_attrs : bool or None, optional
            If True, the attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new
            object will be returned without attributes.

        Returns
        -------
        notnull : DataArray or Dataset
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.notnull

        Examples
        --------
        >>> array = xr.DataArray([1, np.nan, 3], dims="x")
        >>> array
        <xarray.DataArray (x: 3)> Size: 24B
        array([ 1., nan,  3.])
        Dimensions without coordinates: x
        >>> array.notnull()
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False,  True])
        Dimensions without coordinates: x
        """
        pass

    def isin(self, test_elements: Any) -> Self:
        """Tests each value in the array for whether it is in test elements.

        Parameters
        ----------
        test_elements : array_like
            The values against which to test each value of `element`.
            This argument is flattened if an array or array_like.
            See numpy notes for behavior with non-array-like parameters.

        Returns
        -------
        isin : DataArray or Dataset
            Has the same type and shape as this object, but with a bool dtype.

        Examples
        --------
        >>> array = xr.DataArray([1, 2, 3], dims="x")
        >>> array.isin([1, 3])
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False,  True])
        Dimensions without coordinates: x

        See Also
        --------
        numpy.isin
        """
        pass

    def astype(self, dtype, *, order=None, casting=None, subok=None, copy=None, keep_attrs=True) -> Self:
        """
        Copy of the xarray object, with data cast to a specified type.
        Leaves coordinate dtype unchanged.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout order of the result. ‘C’ means C order,
            ‘F’ means Fortran order, ‘A’ means ‘F’ order if all the arrays are
            Fortran contiguous, ‘C’ order otherwise, and ‘K’ means as close to
            the order the array elements appear in memory as possible.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.
        keep_attrs : bool, optional
            By default, astype keeps attributes. Set to False to remove
            attributes in the returned object.

        Returns
        -------
        out : same as object
            New object with data cast to the specified type.

        Notes
        -----
        The ``order``, ``casting``, ``subok`` and ``copy`` arguments are only passed
        through to the ``astype`` method of the underlying array when a value
        different than ``None`` is supplied.
        Make sure to only supply these arguments if the underlying array class
        supports them.

        See Also
        --------
        numpy.ndarray.astype
        dask.array.Array.astype
        sparse.COO.astype
        """
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getitem__(self, value):
        raise NotImplementedError()

def full_like(other: Dataset | DataArray | Variable, fill_value: Any, dtype: DTypeMaybeMapping | None=None, *, chunks: T_Chunks=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None) -> Dataset | DataArray | Variable:
    """
    Return a new object with the same shape and type as a given object.

    Returned object will be chunked if if the given object is chunked, or if chunks or chunked_array_type are specified.

    Parameters
    ----------
    other : DataArray, Dataset or Variable
        The reference object in input
    fill_value : scalar or dict-like
        Value to fill the new object with before returning it. If
        other is a Dataset, may also be a dict-like mapping data
        variables to fill values.
    dtype : dtype or dict-like of dtype, optional
        dtype of the new array. If a dict-like, maps dtypes to
        variables. If omitted, it defaults to other.dtype.
    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
        ``{"x": 5, "y": 5}``.
    chunked_array_type: str, optional
        Which chunked array type to coerce the underlying data array to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        For example, with dask as the default chunked array type, this method would pass additional kwargs
        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.

    Returns
    -------
    out : same as object
        New object with the same shape and type as other, with the data
        filled with fill_value. Coords will be copied from other.
        If other is based on dask, the new one will be as well, and will be
        split in the same chunks.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     np.arange(6).reshape(2, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": [1, 2], "lon": [0, 1, 2]},
    ... )
    >>> x
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.full_like(x, 1)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[1, 1, 1],
           [1, 1, 1]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.full_like(x, 0.5)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0, 0, 0],
           [0, 0, 0]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.full_like(x, 0.5, dtype=np.double)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.full_like(x, np.nan, dtype=np.double)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[nan, nan, nan],
           [nan, nan, nan]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> ds = xr.Dataset(
    ...     {"a": ("x", [3, 5, 2]), "b": ("x", [9, 1, 0])}, coords={"x": [2, 4, 6]}
    ... )
    >>> ds
    <xarray.Dataset> Size: 72B
    Dimensions:  (x: 3)
    Coordinates:
      * x        (x) int64 24B 2 4 6
    Data variables:
        a        (x) int64 24B 3 5 2
        b        (x) int64 24B 9 1 0
    >>> xr.full_like(ds, fill_value={"a": 1, "b": 2})
    <xarray.Dataset> Size: 72B
    Dimensions:  (x: 3)
    Coordinates:
      * x        (x) int64 24B 2 4 6
    Data variables:
        a        (x) int64 24B 1 1 1
        b        (x) int64 24B 2 2 2
    >>> xr.full_like(ds, fill_value={"a": 1, "b": 2}, dtype={"a": bool, "b": float})
    <xarray.Dataset> Size: 51B
    Dimensions:  (x: 3)
    Coordinates:
      * x        (x) int64 24B 2 4 6
    Data variables:
        a        (x) bool 3B True True True
        b        (x) float64 24B 2.0 2.0 2.0

    See Also
    --------
    zeros_like
    ones_like

    """
    pass

def _full_like_variable(other: Variable, fill_value: Any, dtype: DTypeLike | None=None, chunks: T_Chunks=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None) -> Variable:
    """Inner function of full_like, where other must be a variable"""
    pass

def zeros_like(other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping | None=None, *, chunks: T_Chunks=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None) -> Dataset | DataArray | Variable:
    """Return a new object of zeros with the same shape and
    type as a given dataarray or dataset.

    Parameters
    ----------
    other : DataArray, Dataset or Variable
        The reference object. The output will have the same dimensions and coordinates as this object.
    dtype : dtype, optional
        dtype of the new array. If omitted, it defaults to other.dtype.
    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
        ``{"x": 5, "y": 5}``.
    chunked_array_type: str, optional
        Which chunked array type to coerce the underlying data array to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        For example, with dask as the default chunked array type, this method would pass additional kwargs
        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.

    Returns
    -------
    out : DataArray, Dataset or Variable
        New object of zeros with the same shape and type as other.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     np.arange(6).reshape(2, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": [1, 2], "lon": [0, 1, 2]},
    ... )
    >>> x
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.zeros_like(x)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0, 0, 0],
           [0, 0, 0]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.zeros_like(x, dtype=float)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0., 0., 0.],
           [0., 0., 0.]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    See Also
    --------
    ones_like
    full_like

    """
    pass

def ones_like(other: Dataset | DataArray | Variable, dtype: DTypeMaybeMapping | None=None, *, chunks: T_Chunks=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None) -> Dataset | DataArray | Variable:
    """Return a new object of ones with the same shape and
    type as a given dataarray or dataset.

    Parameters
    ----------
    other : DataArray, Dataset, or Variable
        The reference object. The output will have the same dimensions and coordinates as this object.
    dtype : dtype, optional
        dtype of the new array. If omitted, it defaults to other.dtype.
    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)`` or
        ``{"x": 5, "y": 5}``.
    chunked_array_type: str, optional
        Which chunked array type to coerce the underlying data array to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        For example, with dask as the default chunked array type, this method would pass additional kwargs
        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.

    Returns
    -------
    out : same as object
        New object of ones with the same shape and type as other.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     np.arange(6).reshape(2, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": [1, 2], "lon": [0, 1, 2]},
    ... )
    >>> x
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    >>> xr.ones_like(x)
    <xarray.DataArray (lat: 2, lon: 3)> Size: 48B
    array([[1, 1, 1],
           [1, 1, 1]])
    Coordinates:
      * lat      (lat) int64 16B 1 2
      * lon      (lon) int64 24B 0 1 2

    See Also
    --------
    zeros_like
    full_like

    """
    pass

def is_np_datetime_like(dtype: DTypeLike) -> bool:
    """Check if a dtype is a subclass of the numpy datetime types"""
    pass

def is_np_timedelta_like(dtype: DTypeLike) -> bool:
    """Check whether dtype is of the timedelta64 dtype."""
    pass

def _contains_cftime_datetimes(array: Any) -> bool:
    """Check if a array inside a Variable contains cftime.datetime objects"""
    pass

def contains_cftime_datetimes(var: T_Variable) -> bool:
    """Check if an xarray.Variable contains cftime.datetime objects"""
    pass

def _contains_datetime_like_objects(var: T_Variable) -> bool:
    """Check if a variable contains datetime like objects (either
    np.datetime64, np.timedelta64, or cftime.datetime)
    """
    pass
