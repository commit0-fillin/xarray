from __future__ import annotations
import functools
import sys
from typing import Any, Literal
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard
import numpy as np
from xarray.namedarray import utils
NA = utils.ReprObject('<NA>')

@functools.total_ordering
class AlwaysGreaterThan:

    def __gt__(self, other: Any) -> Literal[True]:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

@functools.total_ordering
class AlwaysLessThan:

    def __lt__(self, other: Any) -> Literal[True]:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()
PROMOTE_TO_OBJECT: tuple[tuple[type[np.generic], type[np.generic]], ...] = ((np.number, np.character), (np.bool_, np.character), (np.bytes_, np.str_))

def maybe_promote(dtype: np.dtype[np.generic]) -> tuple[np.dtype[np.generic], Any]:
    """Simpler equivalent of pandas.core.common._maybe_promote

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    dtype : Promoted dtype that can hold missing values.
    fill_value : Valid missing value for the promoted dtype.
    """
    if dtype.kind in "mM":
        return dtype, np.datetime64("NaT")
    elif dtype.kind == "f":
        return dtype, np.nan
    elif dtype.kind in "iu":
        return np.dtype("float64"), np.nan
    elif dtype.kind == "b":
        return np.dtype("object"), NA
    else:
        return np.dtype("object"), NA
NAT_TYPES = {np.datetime64('NaT').dtype, np.timedelta64('NaT').dtype}

def get_fill_value(dtype: np.dtype[np.generic]) -> Any:
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    if dtype.kind in "mM":
        return np.datetime64("NaT")
    elif dtype.kind == "f":
        return np.nan
    elif dtype.kind in "iu":
        return np.iinfo(dtype).min
    elif dtype.kind == "b":
        return None
    else:
        return NA

def get_pos_infinity(dtype: np.dtype[np.generic], max_for_int: bool=False) -> float | complex | AlwaysGreaterThan:
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    max_for_int : bool
        Return np.iinfo(dtype).max instead of np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if dtype.kind == "f":
        return np.inf
    elif dtype.kind in "iu":
        return np.iinfo(dtype).max if max_for_int else np.inf
    elif dtype.kind == "c":
        return complex(np.inf, np.inf)
    else:
        return INF

def get_neg_infinity(dtype: np.dtype[np.generic], min_for_int: bool=False) -> float | complex | AlwaysLessThan:
    """Return an appropriate negative infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    min_for_int : bool
        Return np.iinfo(dtype).min instead of -np.inf

    Returns
    -------
    fill_value : negative infinity value corresponding to this dtype.
    """
    if dtype.kind == "f":
        return -np.inf
    elif dtype.kind in "iu":
        return np.iinfo(dtype).min if min_for_int else -np.inf
    elif dtype.kind == "c":
        return complex(-np.inf, -np.inf)
    else:
        return NINF

def is_datetime_like(dtype: np.dtype[np.generic]) -> TypeGuard[np.datetime64 | np.timedelta64]:
    """Check if a dtype is a subclass of the numpy datetime types"""
    return dtype.kind in "mM"

def result_type(*arrays_and_dtypes: np.typing.ArrayLike | np.typing.DTypeLike) -> np.dtype[np.generic]:
    """Like np.result_type, but with type promotion rules matching pandas.

    Examples of changed behavior:
    number + string -> object (not string)
    bytes + unicode -> object (not unicode)

    Parameters
    ----------
    *arrays_and_dtypes : list of arrays and dtypes
        The dtype is extracted from both numpy and dask arrays.

    Returns
    -------
    numpy.dtype for the result.
    """
    dtypes = []
    for array_or_dtype in arrays_and_dtypes:
        if hasattr(array_or_dtype, "dtype"):
            dtypes.append(array_or_dtype.dtype)
        else:
            dtypes.append(np.dtype(array_or_dtype))

    result = np.result_type(*dtypes)

    for t1, t2 in PROMOTE_TO_OBJECT:
        if any(issubclass(dt.type, t1) for dt in dtypes) and any(issubclass(dt.type, t2) for dt in dtypes):
            return np.dtype("object")

    return result
