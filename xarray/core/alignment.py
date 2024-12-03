from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import Index, Indexes, PandasIndex, PandasMultiIndex, indexes_all_equal, safe_cast_to_index
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import Alignable, JoinOptions, T_DataArray, T_Dataset, T_DuckArray

def reindex_variables(variables: Mapping[Any, Variable], dim_pos_indexers: Mapping[Any, Any], copy: bool=True, fill_value: Any=dtypes.NA, sparse: bool=False) -> dict[Hashable, Variable]:
    """Conform a dictionary of variables onto a new set of variables reindexed
    with dimension positional indexers and possibly filled with missing values.

    Not public API.

    """
    new_variables = {}
    for name, var in variables.items():
        indexers = {k: v for k, v in dim_pos_indexers.items() if k in var.dims}
        if indexers:
            new_variables[name] = var.reindex(
                indexers, copy=copy, fill_value=fill_value, sparse=sparse
            )
        elif copy:
            new_variables[name] = var.copy(deep=False)
        else:
            new_variables[name] = var
    return new_variables
CoordNamesAndDims = tuple[tuple[Hashable, tuple[Hashable, ...]], ...]
MatchingIndexKey = tuple[CoordNamesAndDims, type[Index]]
NormalizedIndexes = dict[MatchingIndexKey, Index]
NormalizedIndexVars = dict[MatchingIndexKey, dict[Hashable, Variable]]

class Aligner(Generic[T_Alignable]):
    """Implements all the complex logic for the re-indexing and alignment of Xarray
    objects.

    For internal use only, not public API.
    Usage:

    aligner = Aligner(*objects, **kwargs)
    aligner.align()
    aligned_objects = aligner.results

    """
    objects: tuple[T_Alignable, ...]
    results: tuple[T_Alignable, ...]
    objects_matching_indexes: tuple[dict[MatchingIndexKey, Index], ...]
    join: str
    exclude_dims: frozenset[Hashable]
    exclude_vars: frozenset[Hashable]
    copy: bool
    fill_value: Any
    sparse: bool
    indexes: dict[MatchingIndexKey, Index]
    index_vars: dict[MatchingIndexKey, dict[Hashable, Variable]]
    all_indexes: dict[MatchingIndexKey, list[Index]]
    all_index_vars: dict[MatchingIndexKey, list[dict[Hashable, Variable]]]
    aligned_indexes: dict[MatchingIndexKey, Index]
    aligned_index_vars: dict[MatchingIndexKey, dict[Hashable, Variable]]
    reindex: dict[MatchingIndexKey, bool]
    reindex_kwargs: dict[str, Any]
    unindexed_dim_sizes: dict[Hashable, set]
    new_indexes: Indexes[Index]

    def __init__(self, objects: Iterable[T_Alignable], join: str='inner', indexes: Mapping[Any, Any] | None=None, exclude_dims: str | Iterable[Hashable]=frozenset(), exclude_vars: Iterable[Hashable]=frozenset(), method: str | None=None, tolerance: float | Iterable[float] | str | None=None, copy: bool=True, fill_value: Any=dtypes.NA, sparse: bool=False):
        self.objects = tuple(objects)
        self.objects_matching_indexes = ()
        if join not in ['inner', 'outer', 'override', 'exact', 'left', 'right']:
            raise ValueError(f'invalid value for join: {join}')
        self.join = join
        self.copy = copy
        self.fill_value = fill_value
        self.sparse = sparse
        if method is None and tolerance is None:
            self.reindex_kwargs = {}
        else:
            self.reindex_kwargs = {'method': method, 'tolerance': tolerance}
        if isinstance(exclude_dims, str):
            exclude_dims = [exclude_dims]
        self.exclude_dims = frozenset(exclude_dims)
        self.exclude_vars = frozenset(exclude_vars)
        if indexes is None:
            indexes = {}
        self.indexes, self.index_vars = self._normalize_indexes(indexes)
        self.all_indexes = {}
        self.all_index_vars = {}
        self.unindexed_dim_sizes = {}
        self.aligned_indexes = {}
        self.aligned_index_vars = {}
        self.reindex = {}
        self.results = tuple()

    def _normalize_indexes(self, indexes: Mapping[Any, Any | T_DuckArray]) -> tuple[NormalizedIndexes, NormalizedIndexVars]:
        """Normalize the indexes/indexers used for re-indexing or alignment.

        Return dictionaries of xarray Index objects and coordinate variables
        such that we can group matching indexes based on the dictionary keys.

        """
        normalized_indexes = {}
        normalized_index_vars = {}

        for key, idx in indexes.items():
            if isinstance(idx, Index):
                index = idx
                index_vars = idx.to_variables()
            else:
                index = PandasIndex(idx, name=key)
                index_vars = {key: Variable((key,), idx)}

            coord_names_and_dims = tuple((name, var.dims) for name, var in index_vars.items())
            matching_key = (coord_names_and_dims, type(index))

            normalized_indexes[matching_key] = index
            normalized_index_vars[matching_key] = index_vars

        return normalized_indexes, normalized_index_vars

    def assert_no_index_conflict(self) -> None:
        """Check for uniqueness of both coordinate and dimension names across all sets
        of matching indexes.

        We need to make sure that all indexes used for re-indexing or alignment
        are fully compatible and do not conflict each other.

        Note: perhaps we could choose less restrictive constraints and instead
        check for conflicts among the dimension (position) indexers returned by
        `Index.reindex_like()` for each matching pair of object index / aligned
        index?
        (ref: https://github.com/pydata/xarray/issues/1603#issuecomment-442965602)

        """
        all_names = set()
        all_dims = set()

        for matching_key in self.indexes:
            coord_names_and_dims, _ = matching_key
            for name, dims in coord_names_and_dims:
                if name in all_names:
                    raise ValueError(f"Conflicting coordinate name: {name}")
                all_names.add(name)

                for dim in dims:
                    if dim in all_dims:
                        raise ValueError(f"Conflicting dimension name: {dim}")
                    all_dims.add(dim)

    def _need_reindex(self, dim, cmp_indexes) -> bool:
        """Whether or not we need to reindex variables for a set of
        matching indexes.

        We don't reindex when all matching indexes are equal for two reasons:
        - It's faster for the usual case (already aligned objects).
        - It ensures it's possible to do operations that don't require alignment
          on indexes with duplicate values (which cannot be reindexed with
          pandas). This is useful, e.g., for overwriting such duplicate indexes.

        """
        if dim in self.exclude_dims:
            return False

        if self.join == 'override':
            return False

        if len(cmp_indexes) == 1:
            return False

        first_index = cmp_indexes[0]
        return not all(indexes_all_equal(first_index, other) for other in cmp_indexes[1:])

    def align_indexes(self) -> None:
        """Compute all aligned indexes and their corresponding coordinate variables."""
        for matching_key, cmp_indexes in self.all_indexes.items():
            dim = matching_key[0][0][1][0]
            if self._need_reindex(dim, cmp_indexes):
                aligned_index = self._align_index(matching_key, cmp_indexes)
                self.aligned_indexes[matching_key] = aligned_index
                self.aligned_index_vars[matching_key] = aligned_index.to_variables()
                self.reindex[matching_key] = True
            else:
                self.aligned_indexes[matching_key] = cmp_indexes[0]
                self.aligned_index_vars[matching_key] = self.all_index_vars[matching_key][0]
                self.reindex[matching_key] = False

    def _align_index(self, matching_key, cmp_indexes):
        join = self.join if self.join != 'override' else 'outer'
        if join == 'inner':
            index = cmp_indexes[0].intersection(cmp_indexes[1:])
        elif join == 'outer':
            index = cmp_indexes[0].union(cmp_indexes[1:])
        elif join in ['left', 'right']:
            index = cmp_indexes[0 if join == 'left' else -1]
        else:
            raise ValueError(f"Invalid join option: {join}")
        return index
T_Obj1 = TypeVar('T_Obj1', bound='Alignable')
T_Obj2 = TypeVar('T_Obj2', bound='Alignable')
T_Obj3 = TypeVar('T_Obj3', bound='Alignable')
T_Obj4 = TypeVar('T_Obj4', bound='Alignable')
T_Obj5 = TypeVar('T_Obj5', bound='Alignable')

def align(*objects: T_Alignable, join: JoinOptions='inner', copy: bool=True, indexes=None, exclude: str | Iterable[Hashable]=frozenset(), fill_value=dtypes.NA) -> tuple[T_Alignable, ...]:
    """
    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with ``fill_value``.
    The default fill value is NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {"outer", "inner", "left", "right", "exact", "override"}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    copy : bool, default: True
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : str, iterable of hashable or None, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    Returns
    -------
    aligned : tuple of DataArray or Dataset
        Tuple of objects with the same type as `*objects` with aligned
        coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.
    """
    if join == 'exact':
        return _exact_align(*objects)

    aligner = Aligner(objects, join=join, copy=copy, indexes=indexes,
                      exclude_dims=exclude, fill_value=fill_value)
    aligner.align_indexes()
    aligner.assert_no_index_conflict()

    aligned = []
    for obj in aligner.objects:
        aligned_obj = obj.reindex_like(aligner.aligned_indexes)
        aligned.append(aligned_obj)

    return tuple(aligned)

def _exact_align(*objects):
    first = objects[0]
    for other in objects[1:]:
        if not first.indexes.equals(other.indexes):
            raise ValueError("Cannot align objects with join='exact' because "
                             "indexes are not equal")
    return objects

def deep_align(objects: Iterable[Any], join: JoinOptions='inner', copy: bool=True, indexes=None, exclude: str | Iterable[Hashable]=frozenset(), raise_on_invalid: bool=True, fill_value=dtypes.NA) -> list[Any]:
    """Align objects for merging, recursing into dictionary values.

    This function is not public API.
    """
    from xarray.core.dataset import Dataset
    from xarray.core.dataarray import DataArray

    def is_alignable(obj):
        return isinstance(obj, (Dataset, DataArray))

    def align_obj(obj):
        if is_alignable(obj):
            return align(obj, join=join, copy=copy, indexes=indexes,
                         exclude=exclude, fill_value=fill_value)[0]
        elif isinstance(obj, dict):
            return {k: align_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [align_obj(item) for item in obj]
        else:
            return obj

    aligned = []
    for obj in objects:
        if is_alignable(obj) or isinstance(obj, (dict, list)):
            aligned.append(align_obj(obj))
        elif raise_on_invalid:
            raise ValueError(f"Object {obj} is not alignable")
        else:
            aligned.append(obj)

    return aligned

def reindex(obj: T_Alignable, indexers: Mapping[Any, Any], method: str | None=None, tolerance: float | Iterable[float] | str | None=None, copy: bool=True, fill_value: Any=dtypes.NA, sparse: bool=False, exclude_vars: Iterable[Hashable]=frozenset()) -> T_Alignable:
    """Re-index either a Dataset or a DataArray.

    Not public API.

    """
    from xarray.core.dataset import Dataset
    from xarray.core.dataarray import DataArray

    if isinstance(obj, Dataset):
        return obj.reindex(indexers, method=method, tolerance=tolerance,
                           copy=copy, fill_value=fill_value, sparse=sparse,
                           exclude_vars=exclude_vars)
    elif isinstance(obj, DataArray):
        return obj.reindex(indexers, method=method, tolerance=tolerance,
                           copy=copy, fill_value=fill_value, sparse=sparse)
    else:
        raise TypeError(f"Expected Dataset or DataArray, got {type(obj)}")

def reindex_like(obj: T_Alignable, other: Dataset | DataArray, method: str | None=None, tolerance: float | Iterable[float] | str | None=None, copy: bool=True, fill_value: Any=dtypes.NA) -> T_Alignable:
    """Re-index either a Dataset or a DataArray like another Dataset/DataArray.

    Not public API.

    """
    indexers = {}
    for dim in other.dims:
        if dim in obj.dims:
            indexers[dim] = other.indexes[dim]
    
    return reindex(obj, indexers, method=method, tolerance=tolerance,
                   copy=copy, fill_value=fill_value)

def broadcast(*args: T_Alignable, exclude: str | Iterable[Hashable] | None=None) -> tuple[T_Alignable, ...]:
    """Explicitly broadcast any number of DataArray or Dataset objects against
    one another.

    xarray objects automatically broadcast against each other in arithmetic
    operations, so this function should not be necessary for normal use.

    If no change is needed, the input data is returned to the output without
    being copied.

    Parameters
    ----------
    *args : DataArray or Dataset
        Arrays to broadcast against each other.
    exclude : str, iterable of hashable or None, optional
        Dimensions that must not be broadcasted

    Returns
    -------
    broadcast : tuple of DataArray or tuple of Dataset
        The same data as the input arrays, but with additional dimensions
        inserted so that all data arrays have the same dimensions and shape.

    Examples
    --------
    Broadcast two data arrays against one another to fill out their dimensions:

    >>> a = xr.DataArray([1, 2, 3], dims="x")
    >>> b = xr.DataArray([5, 6], dims="y")
    >>> a
    <xarray.DataArray (x: 3)> Size: 24B
    array([1, 2, 3])
    Dimensions without coordinates: x
    >>> b
    <xarray.DataArray (y: 2)> Size: 16B
    array([5, 6])
    Dimensions without coordinates: y
    >>> a2, b2 = xr.broadcast(a, b)
    >>> a2
    <xarray.DataArray (x: 3, y: 2)> Size: 48B
    array([[1, 1],
           [2, 2],
           [3, 3]])
    Dimensions without coordinates: x, y
    >>> b2
    <xarray.DataArray (x: 3, y: 2)> Size: 48B
    array([[5, 6],
           [5, 6],
           [5, 6]])
    Dimensions without coordinates: x, y

    Fill out the dimensions of all data variables in a dataset:

    >>> ds = xr.Dataset({"a": a, "b": b})
    >>> (ds2,) = xr.broadcast(ds)  # use tuple unpacking to extract one dataset
    >>> ds2
    <xarray.Dataset> Size: 96B
    Dimensions:  (x: 3, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        a        (x, y) int64 48B 1 1 2 2 3 3
        b        (x, y) int64 48B 5 6 5 6 5 6
    """
    from xarray.core.dataset import Dataset
    from xarray.core.dataarray import DataArray

    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)

    args = align(*args, join='outer', copy=False, exclude=exclude)

    common_dims = set()
    for arg in args:
        common_dims.update(arg.dims)
    common_dims -= exclude

    result = []
    for arg in args:
        missing_dims = common_dims - set(arg.dims)
        if missing_dims:
            expanded = arg.expand_dims({dim: 1 for dim in missing_dims})
            result.append(expanded)
        else:
            result.append(arg)

    return tuple(result)
