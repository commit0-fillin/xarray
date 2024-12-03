from __future__ import annotations
import copy
import warnings
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import DataArrayGroupByAggregations, DatasetGroupByAggregations
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.coordinates import Coordinates
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import PandasIndex, create_default_index_implicit, filter_indexes_from_coords
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, QuantileMethods, T_DataArray, T_DataWithCoords, T_Xarray
from xarray.core.utils import FrozenMappingWarningOnValuesAccess, contains_only_chunked_or_numpy, either_dict_or_kwargs, hashable, is_scalar, maybe_wrap_array, module_available, peek_at
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import GroupIndex, GroupIndices, GroupKey
    from xarray.core.utils import Frozen
    from xarray.groupers import Grouper

def _consolidate_slices(slices: list[slice]) -> list[slice]:
    """Consolidate adjacent slices in a list of slices."""
    consolidated = []
    for s in slices:
        if consolidated and s.start == consolidated[-1].stop:
            consolidated[-1] = slice(consolidated[-1].start, s.stop)
        else:
            consolidated.append(s)
    return consolidated

def _inverse_permutation_indices(positions, N: int | None=None) -> np.ndarray | None:
    """Like inverse_permutation, but also handles slices.

    Parameters
    ----------
    positions : list of ndarray or slice
        If slice objects, all are assumed to be slices.

    Returns
    -------
    np.ndarray of indices or None, if no permutation is necessary.
    """
    if isinstance(positions, slice):
        return None
    if isinstance(positions[0], slice):
        positions = _consolidate_slices(positions)
        if len(positions) == 1 and positions[0] == slice(None):
            return None
        indices = np.concatenate([np.arange(s.start, s.stop) for s in positions])
    else:
        indices = np.concatenate(positions)
    
    if N is None:
        N = indices.max() + 1
    else:
        assert N >= len(indices)
    
    inverse = np.full(N, -1, dtype=np.intp)
    inverse[indices] = np.arange(len(indices))
    return inverse

class _DummyGroup(Generic[T_Xarray]):
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """
    __slots__ = ('name', 'coords', 'size', 'dataarray')

    def __init__(self, obj: T_Xarray, name: Hashable, coords) -> None:
        self.name = name
        self.coords = coords
        self.size = obj.sizes[name]

    def __array__(self) -> np.ndarray:
        return np.arange(self.size)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, = key
        return self.values[key]

    def to_array(self) -> DataArray:
        """Deprecated version of to_dataarray."""
        warnings.warn(
            "The `to_array` method is deprecated and will be removed in a future version. "
            "Please use `to_dataarray` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_dataarray()
T_Group = Union['T_DataArray', _DummyGroup]

@dataclass
class ResolvedGrouper(Generic[T_DataWithCoords]):
    """
    Wrapper around a Grouper object.

    The Grouper object represents an abstract instruction to group an object.
    The ResolvedGrouper object is a concrete version that contains all the common
    logic necessary for a GroupBy problem including the intermediates necessary for
    executing a GroupBy calculation. Specialization to the grouping problem at hand,
    is accomplished by calling the `factorize` method on the encapsulated Grouper
    object.

    This class is private API, while Groupers are public.
    """
    grouper: Grouper
    group: T_Group
    obj: T_DataWithCoords
    codes: DataArray = field(init=False, repr=False)
    full_index: pd.Index = field(init=False, repr=False)
    group_indices: GroupIndices = field(init=False, repr=False)
    unique_coord: Variable | _DummyGroup = field(init=False, repr=False)
    group1d: T_Group = field(init=False, repr=False)
    stacked_obj: T_DataWithCoords = field(init=False, repr=False)
    stacked_dim: Hashable | None = field(init=False, repr=False)
    inserted_dims: list[Hashable] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grouper = copy.deepcopy(self.grouper)
        self.group = _resolve_group(self.obj, self.group)
        self.group1d, self.stacked_obj, self.stacked_dim, self.inserted_dims = _ensure_1d(group=self.group, obj=self.obj)
        self.factorize()

    @property
    def name(self) -> Hashable:
        """Name for the grouped coordinate after reduction."""
        return self.grouper.name

    @property
    def size(self) -> int:
        """Number of groups."""
        return len(self.full_index)

    def __len__(self) -> int:
        """Number of groups."""
        return len(self.full_index)

class GroupBy(Generic[T_Xarray]):
    """A object that implements the split-apply-combine pattern.

    Modeled after `pandas.GroupBy`. The `GroupBy` object can be iterated over
    (unique_value, grouped_array) pairs, but the main way to interact with a
    groupby object are with the `apply` or `reduce` methods. You can also
    directly call numpy methods like `mean` or `std`.

    You should create a GroupBy object by using the `DataArray.groupby` or
    `Dataset.groupby` methods.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    """
    __slots__ = ('_full_index', '_inserted_dims', '_group', '_group_dim', '_group_indices', '_groups', 'groupers', '_obj', '_restore_coord_dims', '_stacked_dim', '_unique_coord', '_dims', '_sizes', '_original_obj', '_original_group', '_bins', '_codes')
    _obj: T_Xarray
    groupers: tuple[ResolvedGrouper]
    _restore_coord_dims: bool
    _original_obj: T_Xarray
    _original_group: T_Group
    _group_indices: GroupIndices
    _codes: DataArray
    _group_dim: Hashable
    _groups: dict[GroupKey, GroupIndex] | None
    _dims: tuple[Hashable, ...] | Frozen[Hashable, int] | None
    _sizes: Mapping[Hashable, int] | None

    def __init__(self, obj: T_Xarray, groupers: tuple[ResolvedGrouper], restore_coord_dims: bool=True) -> None:
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        grouper : Grouper
            Grouper object
        restore_coord_dims : bool, default: True
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        """
        self.groupers = groupers
        self._original_obj = obj
        grouper, = self.groupers
        self._original_group = grouper.group
        self._obj = grouper.stacked_obj
        self._restore_coord_dims = restore_coord_dims
        self._group_indices = grouper.group_indices
        self._codes = self._maybe_unstack(grouper.codes)
        self._group_dim, = grouper.group1d.dims
        self._groups = None
        self._dims = None
        self._sizes = None

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        DataArray.sizes
        Dataset.sizes
        """
        if self._sizes is None:
            self._sizes = self._obj.sizes
        return self._sizes

    @property
    def groups(self) -> dict[GroupKey, GroupIndex]:
        """
        Mapping from group labels to indices. The indices can be used to index the underlying object.
        """
        if self._groups is None:
            self._groups = dict(zip(self.groupers[0].unique_coord.data, self._group_indices))
        return self._groups

    def __getitem__(self, key: GroupKey) -> T_Xarray:
        """
        Get DataArray or Dataset corresponding to a particular group label.
        """
        grouper, = self.groupers
        return self._obj.isel({self._group_dim: self.groups[key]})

    def __len__(self) -> int:
        grouper, = self.groupers
        return grouper.size

    def __iter__(self) -> Iterator[tuple[GroupKey, T_Xarray]]:
        grouper, = self.groupers
        return zip(grouper.unique_coord.data, self._iter_grouped())

    def __repr__(self) -> str:
        grouper, = self.groupers
        return '{}, grouped over {!r}\n{!r} groups with labels {}.'.format(self.__class__.__name__, grouper.name, grouper.full_index.size, ', '.join(format_array_flat(grouper.full_index, 30).split()))

    def _iter_grouped(self) -> Iterator[T_Xarray]:
        """Iterate over each element in this group"""
        for indices in self._group_indices:
            yield self._obj.isel({self._group_dim: indices})

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling or binning). If we
        reduced on that dimension, we want to restore the full index.
        """
        grouper, = self.groupers
        if grouper.name in combined.dims:
            indexers = {grouper.name: grouper.full_index}
            combined = combined.reindex(**indexers)
        return combined

    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        grouper, = self.groupers
        if grouper.stacked_dim is not None:
            obj = obj.unstack(grouper.stacked_dim)
            for dim in grouper.inserted_dims:
                if dim in obj.coords:
                    obj.coords[dim] = grouper.obj[dim]
        return obj

    def _flox_reduce(self, dim: Dims, keep_attrs: bool | None=None, **kwargs: Any):
        """Adaptor function that translates our groupby API to that of flox."""
        import flox

        grouper, = self.groupers
        if isinstance(dim, str):
            dim = [dim]
        elif dim is ...:
            dim = list(self._obj.dims)
        elif dim is None:
            dim = [self._group_dim]

        keep_attrs = _get_keep_attrs(keep_attrs, self._obj)
        result = flox.xarray.reduce(
            self._obj,
            dim=dim,
            group=grouper.group,
            keep_attrs=keep_attrs,
            **kwargs
        )
        return self._maybe_restore_empty_groups(result)

    def fillna(self, value: Any) -> T_Xarray:
        """Fill missing values in this object by group.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value
            Used to fill all matching missing values by group. Needs
            to be of a valid type for the wrapped object's fillna
            method.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.fillna
        DataArray.fillna
        """
        return self.map(lambda x: x.fillna(value))

    @_deprecate_positional_args('v2023.10.0')
    def quantile(self, q: ArrayLike, dim: Dims=None, *, method: QuantileMethods='linear', keep_attrs: bool | None=None, skipna: bool | None=None, interpolation: QuantileMethods | None=None) -> T_Xarray:
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float or sequence of float
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or Iterable of Hashable, optional
            Dimension(s) over which to apply quantile.
            Defaults to the grouped dimension.
        method : str, default: "linear"
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points. The options sorted by their R
            type as summarized in the H&F paper [1]_ are:

                1. "inverted_cdf"
                2. "averaged_inverted_cdf"
                3. "closest_observation"
                4. "interpolated_inverted_cdf"
                5. "hazen"
                6. "weibull"
                7. "linear"  (default)
                8. "median_unbiased"
                9. "normal_unbiased"

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument
            was previously called "interpolation", renamed in accordance with numpy
            version 1.22.0.
        keep_attrs : bool or None, default: None
            If True, the dataarray's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result is a
            scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile. In either case a
            quantile dimension is added to the return array. The other
            dimensions are the dimensions that remain after the
            reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile
        DataArray.quantile

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [[1.3, 8.4, 0.7, 6.9], [0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],
        ...     coords={"x": [0, 0, 1], "y": [1, 1, 2, 2]},
        ...     dims=("x", "y"),
        ... )
        >>> ds = xr.Dataset({"a": da})
        >>> da.groupby("x").quantile(0)
        <xarray.DataArray (x: 2, y: 4)> Size: 64B
        array([[0.7, 4.2, 0.7, 1.5],
               [6.5, 7.3, 2.6, 1.9]])
        Coordinates:
          * y         (y) int64 32B 1 1 2 2
            quantile  float64 8B 0.0
          * x         (x) int64 16B 0 1
        >>> ds.groupby("y").quantile(0, dim=...)
        <xarray.Dataset> Size: 40B
        Dimensions:   (y: 2)
        Coordinates:
            quantile  float64 8B 0.0
          * y         (y) int64 16B 1 2
        Data variables:
            a         (y) float64 16B 0.7 0.7
        >>> da.groupby("x").quantile([0, 0.5, 1])
        <xarray.DataArray (x: 2, y: 4, quantile: 3)> Size: 192B
        array([[[0.7 , 1.  , 1.3 ],
                [4.2 , 6.3 , 8.4 ],
                [0.7 , 5.05, 9.4 ],
                [1.5 , 4.2 , 6.9 ]],
        <BLANKLINE>
               [[6.5 , 6.5 , 6.5 ],
                [7.3 , 7.3 , 7.3 ],
                [2.6 , 2.6 , 2.6 ],
                [1.9 , 1.9 , 1.9 ]]])
        Coordinates:
          * y         (y) int64 32B 1 1 2 2
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
          * x         (x) int64 16B 0 1
        >>> ds.groupby("y").quantile([0, 0.5, 1], dim=...)
        <xarray.Dataset> Size: 88B
        Dimensions:   (y: 2, quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
          * y         (y) int64 16B 1 2
        Data variables:
            a         (y, quantile) float64 48B 0.7 5.35 8.4 0.7 2.25 9.4

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if interpolation is not None:
            warnings.warn(
                "The `interpolation` argument has been deprecated and will be removed "
                "in a future version. Use `method` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = interpolation

        return self._flox_reduce(
            dim=dim,
            func="quantile",
            q=q,
            method=method,
            keep_attrs=keep_attrs,
            skipna=skipna,
        )

    def where(self, cond, other=dtypes.NA) -> T_Xarray:
        """Return elements from `self` or `other` depending on `cond`.

        Parameters
        ----------
        cond : DataArray or Dataset
            Locations at which to preserve this objects values. dtypes have to be `bool`
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, inserts missing values.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.where
        """
        return self.map(lambda x: x.where(cond, other))

    def first(self, skipna: bool | None=None, keep_attrs: bool | None=None):
        """Return the first element of each group along the group dimension"""
        return self._flox_reduce(
            dim=self._group_dim,
            func="first",
            skipna=skipna,
            keep_attrs=keep_attrs,
        )

    def last(self, skipna: bool | None=None, keep_attrs: bool | None=None):
        """Return the last element of each group along the group dimension"""
        return self._flox_reduce(
            dim=self._group_dim,
            func="last",
            skipna=skipna,
            keep_attrs=keep_attrs,
        )

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates by group.

        See Also
        --------
        Dataset.assign_coords
        Dataset.swap_dims
        """
        coords_kwargs = either_dict_or_kwargs(coords, coords_kwargs, "assign_coords")
        return self.map(lambda x: x.assign_coords(**coords_kwargs))

class DataArrayGroupByBase(GroupBy['DataArray'], DataArrayGroupbyArithmetic):
    """GroupBy object specialized to grouping DataArray objects"""
    __slots__ = ()
    _dims: tuple[Hashable, ...] | None

    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        for indices in self._group_indices:
            yield self._obj.isel({self._group_dim: indices}, drop=True).variable

    def map(self, func: Callable[..., DataArray], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> DataArray:
        """Apply a function to each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:

            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.

            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        *args : tuple, optional
            Positional arguments passed to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped_shortcut() if shortcut else self._iter_grouped()
        applied = (func(arr, *args, **kwargs) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)

    def apply(self, func, shortcut=False, args=(), **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayGroupBy.map
        """
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied, shortcut=False):
        """Recombine the applied objects like the original."""
        grouper, = self.groupers
        if shortcut:
            combined = duck_array_ops.concat(applied, axis=self._group_axis)
            return self._obj._replace_maybe_drop_dims(combined)
        else:
            applied_example, applied = peek_at(applied)
            coord = grouper.group
            if coord.name in applied_example.dims:
                stacked = concat(applied, dim=coord)
                return stacked
            else:
                stacked = concat(applied, dim=grouper.name)
                return stacked

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. If None, apply over the
            groupby dimension, if "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = self._group_dim
        
        if isinstance(dim, str) and dim not in self._obj.dims:
            raise ValueError(f"Dimension '{dim}' not found in object")
        
        if axis is not None:
            raise ValueError("axis argument not supported for xarray objects")
        
        keep_attrs = _get_keep_attrs(keep_attrs, self._obj)
        
        def wrapped_func(x):
            return getattr(x, func.__name__)(**kwargs)
        
        result = self._flox_reduce(
            dim=dim,
            func=wrapped_func,
            keep_attrs=keep_attrs,
            keepdims=keepdims,
        )
        
        return result

class DataArrayGroupBy(DataArrayGroupByBase, DataArrayGroupByAggregations, ImplementsArrayReduce):
    __slots__ = ()

class DatasetGroupByBase(GroupBy['Dataset'], DatasetGroupbyArithmetic):
    __slots__ = ()
    _dims: Frozen[Hashable, int] | None

    def map(self, func: Callable[..., Dataset], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> Dataset:
        """Apply a function to each Dataset in the group and concatenate them
        together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments to pass to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        grouped = self._iter_grouped()
        applied = (func(ds, *args, **kwargs) for ds in grouped)
        return self._combine(applied)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DatasetGroupBy.map
        """
        return self.map(func, args=args, shortcut=shortcut, **kwargs)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        grouper, = self.groupers
        applied_example, applied = peek_at(applied)
        coord = grouper.group
        if coord.name in applied_example.dims:
            combined = concat(applied, dim=coord)
        else:
            combined = concat(applied, dim=grouper.name)
        return combined

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : ..., str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default apply over the
            groupby dimension, with "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = self._group_dim
        elif dim is ...:
            dim = list(self._obj.dims)
        
        if axis is not None:
            raise ValueError("axis argument not supported for xarray objects")
        
        keep_attrs = _get_keep_attrs(keep_attrs, self._obj)
        
        def wrapped_func(x):
            return getattr(x, func.__name__)(**kwargs)
        
        result = self._flox_reduce(
            dim=dim,
            func=wrapped_func,
            keep_attrs=keep_attrs,
            keepdims=keepdims,
        )
        
        return result

    def assign(self, **kwargs: Any) -> Dataset:
        """Assign data variables by group.

        See Also
        --------
        Dataset.assign
        """
        return self.map(lambda ds: ds.assign(**kwargs))

class DatasetGroupBy(DatasetGroupByBase, DatasetGroupByAggregations, ImplementsDatasetReduce):
    __slots__ = ()
