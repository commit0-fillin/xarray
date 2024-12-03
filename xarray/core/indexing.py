from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable, overload
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import NDArrayMixin, either_dict_or_kwargs, get_valid_numpy_dtype, is_duck_array, is_duck_dask_array, is_scalar, to_0d_array
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from xarray.core.indexes import Index
    from xarray.core.types import Self
    from xarray.core.variable import Variable
    from xarray.namedarray._typing import _Shape, duckarray
    from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

@dataclass
class IndexSelResult:
    """Index query results.

    Attributes
    ----------
    dim_indexers: dict
        A dictionary where keys are array dimensions and values are
        location-based indexers.
    indexes: dict, optional
        New indexes to replace in the resulting DataArray or Dataset.
    variables : dict, optional
        New variables to replace in the resulting DataArray or Dataset.
    drop_coords : list, optional
        Coordinate(s) to drop in the resulting DataArray or Dataset.
    drop_indexes : list, optional
        Index(es) to drop in the resulting DataArray or Dataset.
    rename_dims : dict, optional
        A dictionary in the form ``{old_dim: new_dim}`` for dimension(s) to
        rename in the resulting DataArray or Dataset.

    """
    dim_indexers: dict[Any, Any]
    indexes: dict[Any, Index] = field(default_factory=dict)
    variables: dict[Any, Variable] = field(default_factory=dict)
    drop_coords: list[Hashable] = field(default_factory=list)
    drop_indexes: list[Hashable] = field(default_factory=list)
    rename_dims: dict[Any, Hashable] = field(default_factory=dict)

    def as_tuple(self):
        """Unlike ``dataclasses.astuple``, return a shallow copy.

        See https://stackoverflow.com/a/51802661

        """
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))

def group_indexers_by_index(obj: T_Xarray, indexers: Mapping[Any, Any], options: Mapping[str, Any]) -> list[tuple[Index, dict[Any, Any]]]:
    """Returns a list of unique indexes and their corresponding indexers."""
    grouped_indexers = defaultdict(dict)
    for k, v in indexers.items():
        index = obj.xindexes.get(k)
        if index is None:
            raise KeyError(f"Dimension {k!r} does not have an index")
        grouped_indexers[index][k] = v
    return list(grouped_indexers.items())

def map_index_queries(obj: T_Xarray, indexers: Mapping[Any, Any], method=None, tolerance: int | float | Iterable[int | float] | None=None, **indexers_kwargs: Any) -> IndexSelResult:
    """Execute index queries from a DataArray / Dataset and label-based indexers
    and return the (merged) query results.

    """
    options = {"method": method, "tolerance": tolerance, **indexers_kwargs}
    grouped = group_indexers_by_index(obj, indexers, options)
    
    dim_indexers = {}
    new_indexes = {}
    variables = {}
    
    for index, sub_indexers in grouped:
        result = index.sel(sub_indexers, **options)
        dim_indexers.update(result.dim_indexers)
        new_indexes.update(result.indexes)
        variables.update(result.variables)
    
    return IndexSelResult(dim_indexers, new_indexes, variables)

def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        key = (key,)
    
    new_key = []
    ellipsis_count = sum(1 for k in key if k is Ellipsis)
    
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif ellipsis_count == 1:
        ellipsis_index = key.index(Ellipsis)
        num_slices = ndim - (len(key) - 1)
        new_key = list(key[:ellipsis_index]) + [slice(None)] * num_slices + list(key[ellipsis_index + 1:])
    else:
        new_key = list(key)
    
    if len(new_key) > ndim:
        raise IndexError("too many indices for array")
    
    new_key.extend([slice(None)] * (ndim - len(new_key)))
    return tuple(new_key)

def _normalize_slice(sl: slice, size: int) -> slice:
    """
    Ensure that given slice only contains positive start and stop values
    (stop can be -1 for full-size slices with negative steps, e.g. [-10::-1])

    Examples
    --------
    >>> _normalize_slice(slice(0, 9), 10)
    slice(0, 9, 1)
    >>> _normalize_slice(slice(0, -1), 10)
    slice(0, 9, 1)
    """
    start, stop, step = sl.start, sl.stop, sl.step

    if step is None:
        step = 1

    if start is None:
        start = 0 if step > 0 else size - 1
    elif start < 0:
        start += size

    if stop is None:
        stop = size if step > 0 else -1
    elif stop < 0:
        stop += size

    if step > 0:
        stop = max(start, min(stop, size))
    else:
        stop = min(start, max(stop, -1))

    return slice(start, stop, step)

def _expand_slice(slice_: slice, size: int) -> np.ndarray[Any, np.dtype[np.integer]]:
    """
    Expand slice to an array containing only positive integers.

    Examples
    --------
    >>> _expand_slice(slice(0, 9), 10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> _expand_slice(slice(0, -1), 10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    normalized_slice = _normalize_slice(slice_, size)
    return np.arange(normalized_slice.start, normalized_slice.stop, normalized_slice.step)

def slice_slice(old_slice: slice, applied_slice: slice, size: int) -> slice:
    """Given a slice and the size of the dimension to which it will be applied,
    index it with another slice to return a new slice equivalent to applying
    the slices sequentially
    """
    old_array = _expand_slice(old_slice, size)
    applied_array = _expand_slice(applied_slice, len(old_array))
    new_array = old_array[applied_array]
    
    if len(new_array) == 0:
        return slice(0, 0, 1)
    
    start = new_array[0]
    stop = new_array[-1] + 1 if len(new_array) > 1 else start + 1
    step = new_array[1] - new_array[0] if len(new_array) > 1 else 1
    
    return slice(start, stop, step)

class ExplicitIndexer:
    """Base class for explicit indexer objects.

    ExplicitIndexer objects wrap a tuple of values given by their ``tuple``
    property. These tuples should always have length equal to the number of
    dimensions on the indexed array.

    Do not instantiate BaseIndexer objects directly: instead, use one of the
    sub-classes BasicIndexer, OuterIndexer or VectorizedIndexer.
    """
    __slots__ = ('_key',)

    def __init__(self, key: tuple[Any, ...]):
        if type(self) is ExplicitIndexer:
            raise TypeError('cannot instantiate base ExplicitIndexer objects')
        self._key = tuple(key)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.tuple})'

class IndexCallable:
    """Provide getitem and setitem syntax for callable objects."""
    __slots__ = ('getter', 'setter')

    def __init__(self, getter: Callable[..., Any], setter: Callable[..., Any] | None=None):
        self.getter = getter
        self.setter = setter

    def __getitem__(self, key: Any) -> Any:
        return self.getter(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.setter is None:
            raise NotImplementedError('Setting values is not supported for this indexer.')
        self.setter(key, value)

class BasicIndexer(ExplicitIndexer):
    """Tuple for basic indexing.

    All elements should be int or slice objects. Indexing follows NumPy's
    rules for basic indexing: each axis is independently sliced and axes
    indexed with an integer are dropped from the result.
    """
    __slots__ = ()

    def __init__(self, key: tuple[int | np.integer | slice, ...]):
        if not isinstance(key, tuple):
            raise TypeError(f'key must be a tuple: {key!r}')
        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            else:
                raise TypeError(f'unexpected indexer type for {type(self).__name__}: {k!r}')
            new_key.append(k)
        super().__init__(tuple(new_key))

class OuterIndexer(ExplicitIndexer):
    """Tuple for outer/orthogonal indexing.

    All elements should be int, slice or 1-dimensional np.ndarray objects with
    an integer dtype. Indexing is applied independently along each axis, and
    axes indexed with an integer are dropped from the result. This type of
    indexing works like MATLAB/Fortran.
    """
    __slots__ = ()

    def __init__(self, key: tuple[int | np.integer | slice | np.ndarray[Any, np.dtype[np.generic]], ...]):
        if not isinstance(key, tuple):
            raise TypeError(f'key must be a tuple: {key!r}')
        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            elif is_duck_array(k):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(f'invalid indexer array, does not have integer dtype: {k!r}')
                if k.ndim > 1:
                    raise TypeError(f'invalid indexer array for {type(self).__name__}; must be scalar or have 1 dimension: {k!r}')
                k = k.astype(np.int64)
            else:
                raise TypeError(f'unexpected indexer type for {type(self).__name__}: {k!r}')
            new_key.append(k)
        super().__init__(tuple(new_key))

class VectorizedIndexer(ExplicitIndexer):
    """Tuple for vectorized indexing.

    All elements should be slice or N-dimensional np.ndarray objects with an
    integer dtype and the same number of dimensions. Indexing follows proposed
    rules for np.ndarray.vindex, which matches NumPy's advanced indexing rules
    (including broadcasting) except sliced axes are always moved to the end:
    https://github.com/numpy/numpy/pull/6256
    """
    __slots__ = ()

    def __init__(self, key: tuple[slice | np.ndarray[Any, np.dtype[np.generic]], ...]):
        if not isinstance(key, tuple):
            raise TypeError(f'key must be a tuple: {key!r}')
        new_key = []
        ndim = None
        for k in key:
            if isinstance(k, slice):
                k = as_integer_slice(k)
            elif is_duck_dask_array(k):
                raise ValueError('Vectorized indexing with Dask arrays is not supported. Please pass a numpy array by calling ``.compute``. See https://github.com/dask/dask/issues/8958.')
            elif is_duck_array(k):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(f'invalid indexer array, does not have integer dtype: {k!r}')
                if ndim is None:
                    ndim = k.ndim
                elif ndim != k.ndim:
                    ndims = [k.ndim for k in key if isinstance(k, np.ndarray)]
                    raise ValueError(f'invalid indexer key: ndarray arguments have different numbers of dimensions: {ndims}')
                k = k.astype(np.int64)
            else:
                raise TypeError(f'unexpected indexer type for {type(self).__name__}: {k!r}')
            new_key.append(k)
        super().__init__(tuple(new_key))

class ExplicitlyIndexed:
    """Mixin to mark support for Indexer subclasses in indexing."""
    __slots__ = ()

    def __array__(self, dtype: np.typing.DTypeLike=None) -> np.ndarray:
        return np.asarray(self.get_duck_array(), dtype=dtype)

class ExplicitlyIndexedNDArrayMixin(NDArrayMixin, ExplicitlyIndexed):
    __slots__ = ()

    def __array__(self, dtype: np.typing.DTypeLike=None) -> np.ndarray:
        return np.asarray(self.get_duck_array(), dtype=dtype)

class ImplicitToExplicitIndexingAdapter(NDArrayMixin):
    """Wrap an array, converting tuples into the indicated explicit indexer."""
    __slots__ = ('array', 'indexer_cls')

    def __init__(self, array, indexer_cls: type[ExplicitIndexer]=BasicIndexer):
        self.array = as_indexable(array)
        self.indexer_cls = indexer_cls
        self.__array_interface__ = self.array.__array_interface__

    def __array__(self, dtype: np.typing.DTypeLike=None) -> np.ndarray:
        return np.asarray(self.get_duck_array(), dtype=dtype)

    def __getitem__(self, key: Any):
        key = expanded_indexer(key, self.ndim)
        indexer = self.indexer_cls(key)
        result = apply_indexer(self.array, indexer)
        if isinstance(result, ExplicitlyIndexed):
            return type(self)(result, self.indexer_cls)
        else:
            return result

class LazilyIndexedArray(ExplicitlyIndexedNDArrayMixin):
    """Wrap an array to make basic and outer indexing lazy."""
    __slots__ = ('array', 'key', '_shape')

    def __init__(self, array: Any, key: ExplicitIndexer | None=None):
        """
        Parameters
        ----------
        array : array_like
            Array like object to index.
        key : ExplicitIndexer, optional
            Array indexer. If provided, it is assumed to already be in
            canonical expanded form.
        """
        if isinstance(array, type(self)) and key is None:
            key = array.key
            array = array.array
        if key is None:
            key = BasicIndexer((slice(None),) * array.ndim)
        self.array = as_indexable(array)
        self.key = key
        shape: _Shape = ()
        for size, k in zip(self.array.shape, self.key.tuple):
            if isinstance(k, slice):
                shape += (len(range(*k.indices(size))),)
            elif isinstance(k, np.ndarray):
                shape += (k.size,)
        self._shape = shape

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return type(self)(self.array, self._updated_key(indexer))

    def __setitem__(self, key: BasicIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(key)
        full_key = self._updated_key(key)
        self.array[full_key] = value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(array={self.array!r}, key={self.key!r})'
LazilyOuterIndexedArray = LazilyIndexedArray

class LazilyVectorizedIndexedArray(ExplicitlyIndexedNDArrayMixin):
    """Wrap an array to make vectorized indexing lazy."""
    __slots__ = ('array', 'key')

    def __init__(self, array: duckarray[Any, Any], key: ExplicitIndexer):
        """
        Parameters
        ----------
        array : array_like
            Array like object to index.
        key : VectorizedIndexer
        """
        if isinstance(key, (BasicIndexer, OuterIndexer)):
            self.key = _outer_to_vectorized_indexer(key, array.shape)
        elif isinstance(key, VectorizedIndexer):
            self.key = _arrayize_vectorized_indexer(key, array.shape)
        self.array = as_indexable(array)

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        if all((isinstance(ind, integer_types) for ind in indexer.tuple)):
            key = BasicIndexer(tuple((k[indexer.tuple] for k in self.key.tuple)))
            return LazilyIndexedArray(self.array, key)
        return type(self)(self.array, self._updated_key(indexer))

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        raise NotImplementedError('Lazy item assignment with the vectorized indexer is not yet implemented. Load your data first by .load() or compute().')

    def __repr__(self) -> str:
        return f'{type(self).__name__}(array={self.array!r}, key={self.key!r})'

def _wrap_numpy_scalars(array):
    """Wrap NumPy scalars in 0d arrays."""
    if isinstance(array, np.generic):
        return np.array(array)
    return array

class CopyOnWriteArray(ExplicitlyIndexedNDArrayMixin):
    __slots__ = ('array', '_copied')

    def __init__(self, array: duckarray[Any, Any]):
        self.array = as_indexable(array)
        self._copied = False

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return type(self)(_wrap_numpy_scalars(self.array[indexer]))

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        self._ensure_copied()
        self.array[indexer] = value

    def __deepcopy__(self, memo):
        return type(self)(self.array)

class MemoryCachedArray(ExplicitlyIndexedNDArrayMixin):
    __slots__ = ('array',)

    def __init__(self, array):
        self.array = _wrap_numpy_scalars(as_indexable(array))

    def __array__(self, dtype: np.typing.DTypeLike=None) -> np.ndarray:
        return np.asarray(self.get_duck_array(), dtype=dtype)

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return type(self)(_wrap_numpy_scalars(self.array[indexer]))

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        self.array[indexer] = value

def as_indexable(array):
    """
    This function always returns a ExplicitlyIndexed subclass,
    so that the vectorized indexing is always possible with the returned
    object.
    """
    if isinstance(array, ExplicitlyIndexed):
        return array
    elif isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    elif hasattr(array, '__array_function__'):
        return NdArrayLikeIndexingAdapter(array)
    elif hasattr(array, '__array_namespace__'):
        return ArrayApiIndexingAdapter(array)
    elif is_duck_dask_array(array):
        return DaskIndexingAdapter(array)
    elif isinstance(array, pd.Index):
        return PandasIndexingAdapter(array)
    elif isinstance(array, pd.MultiIndex):
        return PandasMultiIndexingAdapter(array)
    else:
        raise TypeError(f"Cannot create indexer for object of type: {type(array)}")

def _outer_to_vectorized_indexer(indexer: BasicIndexer | OuterIndexer, shape: _Shape) -> VectorizedIndexer:
    """Convert an OuterIndexer into an vectorized indexer.

    Parameters
    ----------
    indexer : Outer/Basic Indexer
        An indexer to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    VectorizedIndexer
        Tuple suitable for use to index a NumPy array with vectorized indexing.
        Each element is an array: broadcasting them together gives the shape
        of the result.
    """
    vectorized = []
    for dim, key in zip(shape, indexer.tuple):
        if isinstance(key, slice):
            vectorized.append(np.arange(dim)[key])
        elif isinstance(key, np.ndarray):
            vectorized.append(key)
        else:
            vectorized.append(np.array([key]))
    return VectorizedIndexer(tuple(vectorized))

def _outer_to_numpy_indexer(indexer: BasicIndexer | OuterIndexer, shape: _Shape):
    """Convert an OuterIndexer into an indexer for NumPy.

    Parameters
    ----------
    indexer : Basic/OuterIndexer
        An indexer to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    tuple
        Tuple suitable for use to index a NumPy array.
    """
    numpy_indexer = []
    for dim, key in zip(shape, indexer.tuple):
        if isinstance(key, slice):
            numpy_indexer.append(key)
        elif isinstance(key, np.ndarray):
            if key.ndim == 0:
                numpy_indexer.append(key.item())
            elif key.ndim == 1:
                numpy_indexer.append(key)
            else:
                raise ValueError(f"Invalid indexer array with {key.ndim} dimensions")
        else:
            numpy_indexer.append(key)
    return tuple(numpy_indexer)

def _combine_indexers(old_key, shape: _Shape, new_key) -> VectorizedIndexer:
    """Combine two indexers.

    Parameters
    ----------
    old_key : ExplicitIndexer
        The first indexer for the original array
    shape : tuple of ints
        Shape of the original array to be indexed by old_key
    new_key
        The second indexer for indexing original[old_key]
    """
    if isinstance(old_key, VectorizedIndexer):
        old_key = _arrayize_vectorized_indexer(old_key, shape)
    else:
        old_key = _outer_to_vectorized_indexer(old_key, shape)

    if not isinstance(new_key, ExplicitIndexer):
        new_key = BasicIndexer(expanded_indexer(new_key, len(old_key.tuple)))

    if isinstance(new_key, VectorizedIndexer):
        return VectorizedIndexer(tuple(o[n] for o, n in zip(old_key.tuple, new_key.tuple)))
    else:
        return VectorizedIndexer(tuple(o[n] for o, n in zip(old_key.tuple, _outer_to_vectorized_indexer(new_key, tuple(len(o) for o in old_key.tuple)).tuple)))

@enum.unique
class IndexingSupport(enum.Enum):
    BASIC = 0
    OUTER = 1
    OUTER_1VECTOR = 2
    VECTORIZED = 3

def explicit_indexing_adapter(key: ExplicitIndexer, shape: _Shape, indexing_support: IndexingSupport, raw_indexing_method: Callable[..., Any]) -> Any:
    """Support explicit indexing by delegating to a raw indexing method.

    Outer and/or vectorized indexers are supported by indexing a second time
    with a NumPy array.

    Parameters
    ----------
    key : ExplicitIndexer
        Explicit indexing object.
    shape : Tuple[int, ...]
        Shape of the indexed array.
    indexing_support : IndexingSupport enum
        Form of indexing supported by raw_indexing_method.
    raw_indexing_method : callable
        Function (like ndarray.__getitem__) that when called with indexing key
        in the form of a tuple returns an indexed array.

    Returns
    -------
    Indexing result, in the form of a duck numpy-array.
    """
    if isinstance(key, BasicIndexer):
        return raw_indexing_method(key.tuple)
    elif indexing_support == IndexingSupport.BASIC:
        array = raw_indexing_method(())
    elif isinstance(key, OuterIndexer):
        if indexing_support == IndexingSupport.OUTER:
            return raw_indexing_method(key.tuple)
        array = raw_indexing_method(())
    elif isinstance(key, VectorizedIndexer):
        if indexing_support == IndexingSupport.VECTORIZED:
            return raw_indexing_method(key.tuple)
        array = raw_indexing_method(())
    else:
        raise TypeError(f"Unexpected key type: {type(key)}")

    return array[_outer_to_numpy_indexer(key, shape)]

def apply_indexer(indexable, indexer: ExplicitIndexer):
    """Apply an indexer to an indexable object."""
    if isinstance(indexable, ExplicitlyIndexed):
        return indexable[indexer]
    elif isinstance(indexable, np.ndarray):
        return NumpyIndexingAdapter(indexable)[indexer]
    elif hasattr(indexable, '__array_function__'):
        return NdArrayLikeIndexingAdapter(indexable)[indexer]
    elif hasattr(indexable, '__array_namespace__'):
        return ArrayApiIndexingAdapter(indexable)[indexer]
    elif is_duck_dask_array(indexable):
        return DaskIndexingAdapter(indexable)[indexer]
    elif isinstance(indexable, pd.Index):
        return PandasIndexingAdapter(indexable)[indexer]
    elif isinstance(indexable, pd.MultiIndex):
        return PandasMultiIndexingAdapter(indexable)[indexer]
    else:
        raise TypeError(f"Cannot apply indexer to object of type: {type(indexable)}")

def set_with_indexer(indexable, indexer: ExplicitIndexer, value: Any) -> None:
    """Set values in an indexable object using an indexer."""
    if isinstance(indexable, ExplicitlyIndexed):
        indexable[indexer] = value
    elif isinstance(indexable, np.ndarray):
        NumpyIndexingAdapter(indexable)[indexer] = value
    elif hasattr(indexable, '__array_function__'):
        NdArrayLikeIndexingAdapter(indexable)[indexer] = value
    elif hasattr(indexable, '__array_namespace__'):
        ArrayApiIndexingAdapter(indexable)[indexer] = value
    elif is_duck_dask_array(indexable):
        DaskIndexingAdapter(indexable)[indexer] = value
    elif isinstance(indexable, pd.Index):
        PandasIndexingAdapter(indexable)[indexer] = value
    elif isinstance(indexable, pd.MultiIndex):
        PandasMultiIndexingAdapter(indexable)[indexer] = value
    else:
        raise TypeError(f"Cannot set values with indexer for object of type: {type(indexable)}")

def _decompose_slice(key: slice, size: int) -> tuple[slice, slice]:
    """convert a slice to successive two slices. The first slice always has
    a positive step.

    >>> _decompose_slice(slice(2, 98, 2), 99)
    (slice(2, 98, 2), slice(None, None, None))

    >>> _decompose_slice(slice(98, 2, -2), 99)
    (slice(4, 99, 2), slice(None, None, -1))

    >>> _decompose_slice(slice(98, 2, -2), 98)
    (slice(3, 98, 2), slice(None, None, -1))

    >>> _decompose_slice(slice(360, None, -10), 361)
    (slice(0, 361, 10), slice(None, None, -1))
    """
    start, stop, step = key.start, key.stop, key.step
    if step is None or step > 0:
        return key, slice(None)
    
    if start is None:
        start = size - 1
    elif start < 0:
        start += size
    
    if stop is None:
        stop = -1
    elif stop < 0:
        stop += size
    
    if start < stop:
        return slice(0, 0, 1), slice(None)
    
    newstart = max(0, stop + 1)
    newstop = min(start + 1, size)
    newstep = -step
    
    return slice(newstart, newstop, newstep), slice(None, None, -1)

def _decompose_vectorized_indexer(indexer: VectorizedIndexer, shape: _Shape, indexing_support: IndexingSupport) -> tuple[ExplicitIndexer, ExplicitIndexer]:
    """
    Decompose vectorized indexer to the successive two indexers, where the
    first indexer will be used to index backend arrays, while the second one
    is used to index loaded on-memory np.ndarray.

    Parameters
    ----------
    indexer : VectorizedIndexer
    indexing_support : one of IndexerSupport entries

    Returns
    -------
    backend_indexer: OuterIndexer or BasicIndexer
    np_indexers: an ExplicitIndexer (VectorizedIndexer / BasicIndexer)

    Notes
    -----
    This function is used to realize the vectorized indexing for the backend
    arrays that only support basic or outer indexing.

    As an example, let us consider to index a few elements from a backend array
    with a vectorized indexer ([0, 3, 1], [2, 3, 2]).
    Even if the backend array only supports outer indexing, it is more
    efficient to load a subslice of the array than loading the entire array,

    >>> array = np.arange(36).reshape(6, 6)
    >>> backend_indexer = OuterIndexer((np.array([0, 1, 3]), np.array([2, 3])))
    >>> # load subslice of the array
    ... array = NumpyIndexingAdapter(array).oindex[backend_indexer]
    >>> np_indexer = VectorizedIndexer((np.array([0, 2, 1]), np.array([0, 1, 0])))
    >>> # vectorized indexing for on-memory np.ndarray.
    ... NumpyIndexingAdapter(array).vindex[np_indexer]
    array([ 2, 21,  8])
    """
    if indexing_support == IndexingSupport.VECTORIZED:
        return indexer, BasicIndexer(tuple(slice(None) for _ in shape))

    backend_indexer_list = []
    np_indexer_list = []
    
    for dim_indexer in indexer.tuple:
        if isinstance(dim_indexer, slice):
            backend_indexer_list.append(dim_indexer)
            np_indexer_list.append(slice(None))
        else:
            unique_values = np.unique(dim_indexer)
            backend_indexer_list.append(unique_values)
            np_indexer_list.append(np.searchsorted(unique_values, dim_indexer))
    
    if indexing_support == IndexingSupport.OUTER:
        backend_indexer = OuterIndexer(tuple(backend_indexer_list))
    else:
        backend_indexer = BasicIndexer(tuple(slice(min(idx), max(idx) + 1) if isinstance(idx, np.ndarray) else idx for idx in backend_indexer_list))
    
    np_indexer = VectorizedIndexer(tuple(np_indexer_list))
    
    return backend_indexer, np_indexer

def _decompose_outer_indexer(indexer: BasicIndexer | OuterIndexer, shape: _Shape, indexing_support: IndexingSupport) -> tuple[ExplicitIndexer, ExplicitIndexer]:
    """
    Decompose outer indexer to the successive two indexers, where the
    first indexer will be used to index backend arrays, while the second one
    is used to index the loaded on-memory np.ndarray.

    Parameters
    ----------
    indexer : OuterIndexer or BasicIndexer
    indexing_support : One of the entries of IndexingSupport

    Returns
    -------
    backend_indexer: OuterIndexer or BasicIndexer
    np_indexers: an ExplicitIndexer (OuterIndexer / BasicIndexer)

    Notes
    -----
    This function is used to realize the vectorized indexing for the backend
    arrays that only support basic or outer indexing.

    As an example, let us consider to index a few elements from a backend array
    with a orthogonal indexer ([0, 3, 1], [2, 3, 2]).
    Even if the backend array only supports basic indexing, it is more
    efficient to load a subslice of the array than loading the entire array,

    >>> array = np.arange(36).reshape(6, 6)
    >>> backend_indexer = BasicIndexer((slice(0, 3), slice(2, 4)))
    >>> # load subslice of the array
    ... array = NumpyIndexingAdapter(array)[backend_indexer]
    >>> np_indexer = OuterIndexer((np.array([0, 2, 1]), np.array([0, 1, 0])))
    >>> # outer indexing for on-memory np.ndarray.
    ... NumpyIndexingAdapter(array).oindex[np_indexer]
    array([[ 2,  3,  2],
           [14, 15, 14],
           [ 8,  9,  8]])
    """
    if indexing_support == IndexingSupport.OUTER:
        return indexer, BasicIndexer(tuple(slice(None) for _ in shape))

    backend_indexer_list = []
    np_indexer_list = []
    
    for dim_indexer, dim_size in zip(indexer.tuple, shape):
        if isinstance(dim_indexer, slice):
            backend_indexer_list.append(dim_indexer)
            np_indexer_list.append(slice(None))
        elif isinstance(dim_indexer, np.ndarray):
            if dim_indexer.dtype == bool:
                dim_indexer = np.nonzero(dim_indexer)[0]
            if len(dim_indexer) > 0:
                start, stop = dim_indexer.min(), dim_indexer.max() + 1
                backend_indexer_list.append(slice(start, stop))
                np_indexer_list.append(dim_indexer - start)
            else:
                backend_indexer_list.append(slice(0, 0))
                np_indexer_list.append(dim_indexer)
        else:
            backend_indexer_list.append(slice(dim_indexer, dim_indexer + 1))
            np_indexer_list.append(0)
    
    if indexing_support == IndexingSupport.BASIC:
        backend_indexer = BasicIndexer(tuple(backend_indexer_list))
    else:
        backend_indexer = OuterIndexer(tuple(backend_indexer_list))
    
    np_indexer = OuterIndexer(tuple(np_indexer_list))
    
    return backend_indexer, np_indexer

def _arrayize_vectorized_indexer(indexer: VectorizedIndexer, shape: _Shape) -> VectorizedIndexer:
    """Return an identical vindex but slices are replaced by arrays"""
    def maybe_expand(k, length):
        if isinstance(k, slice):
            return np.arange(*k.indices(length))
        return k

    return VectorizedIndexer(tuple(maybe_expand(k, length) for k, length in zip(indexer.tuple, shape)))

def _chunked_array_with_chunks_hint(array, chunks, chunkmanager: ChunkManagerEntrypoint[Any]):
    """Create a chunked array using the chunks hint for dimensions of size > 1."""
    if chunks is None:
        return array

    shape = array.shape
    chunks = tuple(
        chunk if len(np.atleast_1d(chunk)) > 1 else -1
        for chunk, size in zip(chunks, shape)
    )
    return chunkmanager.chunked_array(array, chunks=chunks)

def create_mask(indexer: ExplicitIndexer, shape: _Shape, data: duckarray[Any, Any] | None=None):
    """Create a mask for indexing with a fill-value.

    Parameters
    ----------
    indexer : ExplicitIndexer
        Indexer with -1 in integer or ndarray value to indicate locations in
        the result that should be masked.
    shape : tuple
        Shape of the array being indexed.
    data : optional
        Data for which mask is being created. If data is a dask arrays, its chunks
        are used as a hint for chunks on the resulting mask. If data is a sparse
        array, the returned mask is also a sparse array.

    Returns
    -------
    mask : bool, np.ndarray, SparseArray or dask.array.Array with dtype=bool
        Same type as data. Has the same shape as the indexing result.
    """
    if isinstance(indexer, BasicIndexer):
        mask = np.zeros(shape, dtype=bool)
        mask_indexer = tuple(
            slice(None) if not isinstance(ind, integer_types) else ind
            for ind in indexer.tuple
        )
        mask[mask_indexer] = True
    elif isinstance(indexer, OuterIndexer):
        mask = np.ones(shape, dtype=bool)
        for dim, ind in enumerate(indexer.tuple):
            if isinstance(ind, np.ndarray):
                mask &= ind != -1
    else:  # VectorizedIndexer
        mask = indexer.tuple[0] != -1
        for ind in indexer.tuple[1:]:
            mask &= ind != -1

    if data is not None:
        if is_duck_dask_array(data):
            import dask.array as da
            chunks = data.chunks
            mask = _chunked_array_with_chunks_hint(mask, chunks, da)
        elif is_sparse_array(data):
            import sparse
            mask = sparse.COO.from_numpy(mask)

    return mask

def _posify_mask_subindexer(index: np.ndarray[Any, np.dtype[np.generic]]) -> np.ndarray[Any, np.dtype[np.generic]]:
    """Convert masked indices in a flat array to the nearest unmasked index.

    Parameters
    ----------
    index : np.ndarray
        One dimensional ndarray with dtype=int.

    Returns
    -------
    np.ndarray
        One dimensional ndarray with all values equal to -1 replaced by an
        adjacent non-masked element.
    """
    masked = index == -1
    if not np.any(masked):
        return index

    unmasked_locs = np.nonzero(~masked)[0]
    if len(unmasked_locs) == 0:
        return np.zeros_like(index)

    first_unmasked = unmasked_locs[0]
    last_unmasked = unmasked_locs[-1]

    index = index.copy()
    if first_unmasked > 0:
        index[:first_unmasked] = index[first_unmasked]
    if last_unmasked < len(index) - 1:
        index[last_unmasked + 1:] = index[last_unmasked]

    remaining_masked = index == -1
    if np.any(remaining_masked):
        unmasked_locs = np.nonzero(~remaining_masked)[0]
        masked_locs = np.nonzero(remaining_masked)[0]
        index[masked_locs] = index[unmasked_locs[np.searchsorted(unmasked_locs, masked_locs) - 1]]

    return index

def posify_mask_indexer(indexer: ExplicitIndexer) -> ExplicitIndexer:
    """Convert masked values (-1) in an indexer to nearest unmasked values.

    This routine is useful for dask, where it can be much faster to index
    adjacent points than arbitrary points from the end of an array.

    Parameters
    ----------
    indexer : ExplicitIndexer
        Input indexer.

    Returns
    -------
    ExplicitIndexer
        Same type of input, with all values in ndarray keys equal to -1
        replaced by an adjacent non-masked element.
    """
    if isinstance(indexer, VectorizedIndexer):
        return VectorizedIndexer(tuple(_posify_mask_subindexer(ind) for ind in indexer.tuple))
    elif isinstance(indexer, OuterIndexer):
        return OuterIndexer(tuple(
            _posify_mask_subindexer(ind) if isinstance(ind, np.ndarray) else ind
            for ind in indexer.tuple
        ))
    else:
        return indexer

def is_fancy_indexer(indexer: Any) -> bool:
    """Return False if indexer is a int, slice, a 1-dimensional list, or a 0 or
    1-dimensional ndarray; in all other cases return True
    """
    if isinstance(indexer, (int, slice)):
        return False
    elif isinstance(indexer, list):
        return any(isinstance(item, list) for item in indexer)
    elif isinstance(indexer, np.ndarray):
        return indexer.ndim > 1
    else:
        return True

class NumpyIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a NumPy array to use explicit indexing."""
    __slots__ = ('array',)

    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError(f'NumpyIndexingAdapter only wraps np.ndarray. Trying to wrap {type(array)}')
        self.array = array

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        array = self.array
        key = indexer.tuple + (Ellipsis,)
        return array[key]

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        array = self.array
        key = indexer.tuple + (Ellipsis,)
        self._safe_setitem(array, key, value)

class NdArrayLikeIndexingAdapter(NumpyIndexingAdapter):
    __slots__ = ('array',)

    def __init__(self, array):
        if not hasattr(array, '__array_function__'):
            raise TypeError('NdArrayLikeIndexingAdapter must wrap an object that implements the __array_function__ protocol')
        self.array = array

class ArrayApiIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap an array API array to use explicit indexing."""
    __slots__ = ('array',)

    def __init__(self, array):
        if not hasattr(array, '__array_namespace__'):
            raise TypeError('ArrayApiIndexingAdapter must wrap an object that implements the __array_namespace__ protocol')
        self.array = array

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return self.array[indexer.tuple]

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        self.array[indexer.tuple] = value

class DaskIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a dask array to support explicit indexing."""
    __slots__ = ('array',)

    def __init__(self, array):
        """This adapter is created in Variable.__getitem__ in
        Variable._broadcast_indexes.
        """
        self.array = array

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        return self.array[indexer.tuple]

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        self._check_and_raise_if_non_basic_indexer(indexer)
        self.array[indexer.tuple] = value

class PandasIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a pandas.Index to preserve dtypes and handle explicit indexing."""
    __slots__ = ('array', '_dtype')
    array: pd.Index
    _dtype: np.dtype

    def __init__(self, array: pd.Index, dtype: DTypeLike=None):
        from xarray.core.indexes import safe_cast_to_index
        self.array = safe_cast_to_index(array)
        if dtype is None:
            self._dtype = get_valid_numpy_dtype(array)
        else:
            self._dtype = np.dtype(dtype)

    def __array__(self, dtype: DTypeLike=None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        array = self.array
        if isinstance(array, pd.PeriodIndex):
            with suppress(AttributeError):
                array = array.astype('object')
        return np.asarray(array.values, dtype=dtype)

    def __getitem__(self, indexer: ExplicitIndexer) -> PandasIndexingAdapter | NumpyIndexingAdapter | np.ndarray | np.datetime64 | np.timedelta64:
        key = self._prepare_key(indexer.tuple)
        if getattr(key, 'ndim', 0) > 1:
            indexable = NumpyIndexingAdapter(np.asarray(self))
            return indexable[indexer]
        result = self.array[key]
        return self._handle_result(result)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(array={self.array!r}, dtype={self.dtype!r})'

class PandasMultiIndexingAdapter(PandasIndexingAdapter):
    """Handles explicit indexing for a pandas.MultiIndex.

    This allows creating one instance for each multi-index level while
    preserving indexing efficiency (memoized + might reuse another instance with
    the same multi-index).
    """
    __slots__ = ('array', '_dtype', 'level', 'adapter')
    array: pd.MultiIndex
    _dtype: np.dtype
    level: str | None

    def __init__(self, array: pd.MultiIndex, dtype: DTypeLike=None, level: str | None=None):
        super().__init__(array, dtype)
        self.level = level

    def __array__(self, dtype: DTypeLike=None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        if self.level is not None:
            return np.asarray(self.array.get_level_values(self.level).values, dtype=dtype)
        else:
            return super().__array__(dtype)

    def __getitem__(self, indexer: ExplicitIndexer):
        result = super().__getitem__(indexer)
        if isinstance(result, type(self)):
            result.level = self.level
        return result

    def __repr__(self) -> str:
        if self.level is None:
            return super().__repr__()
        else:
            props = f'(array={self.array!r}, level={self.level!r}, dtype={self.dtype!r})'
            return f'{type(self).__name__}{props}'
