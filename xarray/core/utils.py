"""Internal utilities; not for external use"""
from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import Collection, Container, Hashable, ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping, MutableSet, Sequence, ValuesView
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, overload
import numpy as np
import pandas as pd
from xarray.namedarray.utils import ReprObject, drop_missing_dims, either_dict_or_kwargs, infix_dims, is_dask_collection, is_dict_like, is_duck_array, is_duck_dask_array, module_available, to_0d_object_array
if TYPE_CHECKING:
    from xarray.core.types import Dims, ErrorOptionsWithWarn
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

def get_valid_numpy_dtype(array: np.ndarray | pd.Index) -> np.dtype:
    """Return a numpy compatible dtype from either
    a numpy array or a pandas.Index.

    Used for wrapping a pandas.Index as an xarray.Variable.

    """
    if isinstance(array, pd.Index):
        dtype = array.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return np.dtype('O')
        return dtype
    return array.dtype

def maybe_coerce_to_str(index, original_coords):
    """maybe coerce a pandas Index back to a nunpy array of type str

    pd.Index uses object-dtype to store str - try to avoid this for coords
    """
    if isinstance(index, pd.Index) and index.dtype == object:
        if original_coords.dtype.kind in ['U', 'S']:
            return index.values.astype(original_coords.dtype)
    return index

def maybe_wrap_array(original, new_array):
    """Wrap a transformed array with __array_wrap__ if it can be done safely.

    This lets us treat arbitrary functions that take and return ndarray objects
    like ufuncs, as long as they return an array with the same shape.
    """
    if hasattr(original, '__array_wrap__'):
        if new_array.shape == original.shape:
            return original.__array_wrap__(new_array)
    return new_array

def equivalent(first: T, second: T) -> bool:
    """Compare two objects for equivalence (identity or equality), using
    array_equiv if either object is an ndarray. If both objects are lists,
    equivalent is sequentially called on all the elements.
    """
    if first is second:
        return True
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equiv(first, second)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(equivalent(i, j) for i, j in zip(first, second))
    return first == second

def peek_at(iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    iterator = iter(iterable)
    try:
        first_value = next(iterator)
    except StopIteration:
        return None, iter([])
    return first_value, itertools.chain([first_value], iterator)

def update_safety_check(first_dict: Mapping[K, V], second_dict: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> None:
    """Check the safety of updating one dictionary with another.

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by identity (they are the same item) or
    the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        All items in the second dictionary are checked against for conflicts
        against items in the first dictionary.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.
    """
    for k, v in second_dict.items():
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError(f"unsafe to merge dictionaries: conflicting values for key {k!r}")

def remove_incompatible_items(first_dict: MutableMapping[K, V], second_dict: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> None:
    """Remove incompatible items from the first dictionary in-place.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.
    """
    pass
try:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
except ImportError:
    if TYPE_CHECKING:
        raise
    else:

        def is_scalar(value: Any, include_0d: bool=True) -> bool:
            """Whether to treat a value as a scalar.

            Any non-iterable, string, or 0-D array
            """
            pass
else:

    def is_scalar(value: Any, include_0d: bool=True) -> TypeGuard[Hashable]:
        """Whether to treat a value as a scalar.

        Any non-iterable, string, or 0-D array
        """
        pass

def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray."""
    pass

def dict_equiv(first: Mapping[K, V], second: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> bool:
    """Test equivalence of two dict-like objects. If any of the values are
    numpy arrays, compare them correctly.

    Parameters
    ----------
    first, second : dict-like
        Dictionaries to compare for equality
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    equals : bool
        True if the dictionaries are equal
    """
    pass

def compat_dict_intersection(first_dict: Mapping[K, V], second_dict: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> MutableMapping[K, V]:
    """Return the intersection of two dictionaries as a new dictionary.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    intersection : dict
        Intersection of the contents.
    """
    pass

def compat_dict_union(first_dict: Mapping[K, V], second_dict: Mapping[K, V], compat: Callable[[V, V], bool]=equivalent) -> MutableMapping[K, V]:
    """Return the union of two dictionaries as a new dictionary.

    An exception is raised if any keys are found in both dictionaries and the
    values are not compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    union : dict
        union of the contents.
    """
    pass

class Frozen(Mapping[K, V]):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """
    __slots__ = ('mapping',)

    def __init__(self, mapping: Mapping[K, V]):
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.mapping!r})'

class FrozenMappingWarningOnValuesAccess(Frozen[K, V]):
    """
    Class which behaves like a Mapping but warns if the values are accessed.

    Temporary object to aid in deprecation cycle of `Dataset.dims` (see GH issue #8496).
    `Dataset.dims` is being changed from returning a mapping of dimension names to lengths to just
    returning a frozen set of dimension names (to increase consistency with `DataArray.dims`).
    This class retains backwards compatibility but raises a warning only if the return value
    of ds.dims is used like a dictionary (i.e. it doesn't raise a warning if used in a way that
    would also be valid for a FrozenSet, e.g. iteration).
    """
    __slots__ = ('mapping',)

    def __getitem__(self, key: K) -> V:
        self._warn()
        return super().__getitem__(key)

class HybridMappingProxy(Mapping[K, V]):
    """Implements the Mapping interface. Uses the wrapped mapping for item lookup
    and a separate wrapped keys collection for iteration.

    Can be used to construct a mapping object from another dict-like object without
    eagerly accessing its items or when a mapping object is expected but only
    iteration over keys is actually used.

    Note: HybridMappingProxy does not validate consistency of the provided `keys`
    and `mapping`. It is the caller's responsibility to ensure that they are
    suitable for the task at hand.
    """
    __slots__ = ('_keys', 'mapping')

    def __init__(self, keys: Collection[K], mapping: Mapping[K, V]):
        self._keys = keys
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

class OrderedSet(MutableSet[T]):
    """A simple ordered set.

    The API matches the builtin set, but it preserves insertion order of elements, like
    a dict. Note that, unlike in an OrderedDict, equality tests are not order-sensitive.
    """
    _d: dict[T, None]
    __slots__ = ('_d',)

    def __init__(self, values: Iterable[T] | None=None):
        self._d = {}
        if values is not None:
            self.update(values)

    def __contains__(self, value: Hashable) -> bool:
        return value in self._d

    def __iter__(self) -> Iterator[T]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({list(self)!r})'

class NdimSizeLenMixin:
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """
    __slots__ = ()

    @property
    def ndim(self: Any) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        pass

    @property
    def size(self: Any) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        pass

    def __len__(self: Any) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError('len() of unsized object')

class NDArrayMixin(NdimSizeLenMixin):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """
    __slots__ = ()

    def __getitem__(self: Any, key):
        return self.array[key]

    def __repr__(self: Any) -> str:
        return f'{type(self).__name__}(array={self.array!r})'

@contextlib.contextmanager
def close_on_error(f):
    """Context manager to ensure that a file opened by xarray is closed if an
    exception is raised before the user sees the file object.
    """
    pass

def is_remote_uri(path: str) -> bool:
    """Finds URLs of the form protocol:// or protocol::

    This also matches for http[s]://, which were the only remote URLs
    supported in <=v0.16.2.
    """
    pass

def is_uniform_spaced(arr, **kwargs) -> bool:
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    pass

def hashable(v: Any) -> TypeGuard[Hashable]:
    """Determine whether `v` can be hashed."""
    pass

def iterable(v: Any) -> TypeGuard[Iterable[Any]]:
    """Determine whether `v` is iterable."""
    pass

def iterable_of_hashable(v: Any) -> TypeGuard[Iterable[Hashable]]:
    """Determine whether `v` is an Iterable of Hashables."""
    pass

def decode_numpy_dict_values(attrs: Mapping[K, V]) -> dict[K, V]:
    """Convert attribute values from numpy objects to native Python objects,
    for use in to_dict
    """
    pass

def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    pass

class HiddenKeyDict(MutableMapping[K, V]):
    """Acts like a normal dictionary, but hides certain keys."""
    __slots__ = ('_data', '_hidden_keys')

    def __init__(self, data: MutableMapping[K, V], hidden_keys: Iterable[K]):
        self._data = data
        self._hidden_keys = frozenset(hidden_keys)

    def __setitem__(self, key: K, value: V) -> None:
        self._raise_if_hidden(key)
        self._data[key] = value

    def __getitem__(self, key: K) -> V:
        self._raise_if_hidden(key)
        return self._data[key]

    def __delitem__(self, key: K) -> None:
        self._raise_if_hidden(key)
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        for k in self._data:
            if k not in self._hidden_keys:
                yield k

    def __len__(self) -> int:
        num_hidden = len(self._hidden_keys & self._data.keys())
        return len(self._data) - num_hidden

def get_temp_dimname(dims: Container[Hashable], new_dim: Hashable) -> Hashable:
    """Get an new dimension name based on new_dim, that is not used in dims.
    If the same name exists, we add an underscore(s) in the head.

    Example1:
        dims: ['a', 'b', 'c']
        new_dim: ['_rolling']
        -> ['_rolling']
    Example2:
        dims: ['a', 'b', 'c', '_rolling']
        new_dim: ['_rolling']
        -> ['__rolling']
    """
    pass

def drop_dims_from_indexers(indexers: Mapping[Any, Any], dims: Iterable[Hashable] | Mapping[Any, int], missing_dims: ErrorOptionsWithWarn) -> Mapping[Hashable, Any]:
    """Depending on the setting of missing_dims, drop any dimensions from indexers that
    are not present in dims.

    Parameters
    ----------
    indexers : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """
    pass

def parse_dims(dim: Dims, all_dims: tuple[Hashable, ...], *, check_exists: bool=True, replace_none: bool=True) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    If you supply e.g. a set of dimensions the order cannot be
    conserved, but for sequences it will be.

    Parameters
    ----------
    dim : str, Iterable of Hashable, "..." or None
        Dimension(s) to parse.
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None or "...".

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    pass

def parse_ordered_dims(dim: Dims, all_dims: tuple[Hashable, ...], *, check_exists: bool=True, replace_none: bool=True) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    An ellipsis ("...") in a sequence of dimensions will be
    replaced with all remaining dimensions. This only makes sense when
    the input is a sequence and not e.g. a set.

    Parameters
    ----------
    dim : str, Sequence of Hashable or "...", "..." or None
        Dimension(s) to parse. If "..." appears in a Sequence
        it always gets replaced with all remaining dims
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None.

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    pass
_Accessor = TypeVar('_Accessor')

class UncachedAccessor(Generic[_Accessor]):
    """Acts like a property, but on both classes and class instances

    This class is necessary because some tools (e.g. pydoc and sphinx)
    inspect classes for which property returns itself and not the
    accessor.
    """

    def __init__(self, accessor: type[_Accessor]) -> None:
        self._accessor = accessor

    @overload
    def __get__(self, obj: None, cls) -> type[_Accessor]:
        ...

    @overload
    def __get__(self, obj: object, cls) -> _Accessor:
        ...

    def __get__(self, obj: None | object, cls) -> type[_Accessor] | _Accessor:
        if obj is None:
            return self._accessor
        return self._accessor(obj)

class Default(Enum):
    token = 0
_default = Default.token

def contains_only_chunked_or_numpy(obj) -> bool:
    """Returns True if xarray object contains only numpy arrays or chunked arrays (i.e. pure dask or cubed).

    Expects obj to be Dataset or DataArray"""
    pass

def find_stack_level(test_mode=False) -> int:
    """Find the first place in the stack that is not inside xarray or the Python standard library.

    This is unless the code emanates from a test, in which case we would prefer
    to see the xarray source.

    This function is taken from pandas and modified to exclude standard library paths.

    Parameters
    ----------
    test_mode : bool
        Flag used for testing purposes to switch off the detection of test
        directories in the stack trace.

    Returns
    -------
    stacklevel : int
        First level in the stack that is not part of xarray or the Python standard library.
    """
    pass

def emit_user_level_warning(message, category=None) -> None:
    """Emit a warning at the user level by inspecting the stack trace."""
    pass

def consolidate_dask_from_array_kwargs(from_array_kwargs: dict[Any, Any], name: str | None=None, lock: bool | None=None, inline_array: bool | None=None) -> dict[Any, Any]:
    """
    Merge dask-specific kwargs with arbitrary from_array_kwargs dict.

    Temporary function, to be deleted once explicitly passing dask-specific kwargs to .chunk() is deprecated.
    """
    pass
