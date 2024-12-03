from __future__ import annotations
import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast
import numpy as np
from packaging.version import Version
from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike
if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    from numpy.typing import NDArray
    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray
        DaskCollection: Any = NDArray
    from xarray.namedarray._typing import _Dim, duckarray
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

@lru_cache
def module_available(module: str, minversion: str | None=None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    try:
        mod = importlib.util.find_spec(module)
        if mod is None:
            return False
        if minversion is not None:
            import pkg_resources
            try:
                return pkg_resources.parse_version(pkg_resources.get_distribution(module).version) >= pkg_resources.parse_version(minversion)
            except pkg_resources.DistributionNotFound:
                return False
        return True
    except ImportError:
        return False

def to_0d_object_array(value: object) -> NDArray[np.object_]:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    return np.array(value, dtype=object)

def drop_missing_dims(supplied_dims: Iterable[_Dim], dims: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn) -> _DimsLike:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : Iterable of Hashable
    dims : Iterable of Hashable
    missing_dims : {"raise", "warn", "ignore"}
    """
    dims_set = set(dims)
    result = []
    missing = []

    for dim in supplied_dims:
        if dim in dims_set:
            result.append(dim)
        else:
            missing.append(dim)

    if missing:
        if missing_dims == "raise":
            raise ValueError(f"Dimensions {missing} not found in array dimensions {dims_set}")
        elif missing_dims == "warn":
            warnings.warn(f"Dimensions {missing} not found in array dimensions {dims_set}")

    return tuple(result)

def infix_dims(dims_supplied: Iterable[_Dim], dims_all: Iterable[_Dim], missing_dims: ErrorOptionsWithWarn='raise') -> Iterator[_Dim]:
    """
    Resolves a supplied list containing an ellipsis representing other items, to
    a generator with the 'realized' list of all items
    """
    dims_supplied = list(dims_supplied)
    dims_all = list(dims_all)

    if Ellipsis not in dims_supplied:
        yield from drop_missing_dims(dims_supplied, dims_all, missing_dims)
        return

    ellipsis_index = dims_supplied.index(Ellipsis)
    before_ellipsis = dims_supplied[:ellipsis_index]
    after_ellipsis = dims_supplied[ellipsis_index + 1:]

    for dim in drop_missing_dims(before_ellipsis, dims_all, missing_dims):
        yield dim

    dims_set = set(dims_all)
    for dim in dims_all:
        if dim not in before_ellipsis and dim not in after_ellipsis:
            yield dim

    for dim in drop_missing_dims(after_ellipsis, dims_all, missing_dims):
        yield dim

class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""
    __slots__ = ('_value',)
    _value: str

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other: ReprObject | Any) -> bool:
        return self._value == other._value if isinstance(other, ReprObject) else False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._value))
