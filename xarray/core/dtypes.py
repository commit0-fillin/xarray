from __future__ import annotations
import functools
from typing import Any
import numpy as np
from pandas.api.types import is_extension_array_dtype
from xarray.core import array_api_compat, npcompat, utils
NA = utils.ReprObject('<NA>')

@functools.total_ordering
class AlwaysGreaterThan:

    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))

@functools.total_ordering
class AlwaysLessThan:

    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()
PROMOTE_TO_OBJECT: tuple[tuple[type[np.generic], type[np.generic]], ...] = ((np.number, np.character), (np.bool_, np.character), (np.bytes_, np.str_))

def maybe_promote(dtype: np.dtype) -> tuple[np.dtype, Any]:
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
        return np.dtype(float), np.nan
    elif dtype.kind == "b":
        return np.dtype(object), NA
    else:
        return np.dtype(object), NA
NAT_TYPES = {np.datetime64('NaT').dtype, np.timedelta64('NaT').dtype}

def get_fill_value(dtype):
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in "mM":
        return np.datetime64("NaT")
    elif dtype.kind == "f":
        return np.nan
    elif dtype.kind in "iu":
        return np.iinfo(dtype).min
    elif dtype.kind == "b":
        return None
    elif dtype.kind in "SU":
        return ""
    else:
        return NA

def get_pos_infinity(dtype, max_for_int=False):
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
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.inf
    elif dtype.kind in "iu":
        return np.iinfo(dtype).max if max_for_int else np.inf
    elif dtype.kind in "mM":
        return np.datetime64("NaT")
    else:
        return INF

def get_neg_infinity(dtype, min_for_int=False):
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
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return -np.inf
    elif dtype.kind in "iu":
        return np.iinfo(dtype).min if min_for_int else -np.inf
    elif dtype.kind in "mM":
        return np.datetime64("NaT")
    else:
        return NINF

def is_datetime_like(dtype) -> bool:
    """Check if a dtype is a subclass of the numpy datetime types"""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)

def is_object(dtype) -> bool:
    """Check if a dtype is object"""
    return np.issubdtype(dtype, np.object_)

def is_string(dtype) -> bool:
    """Check if a dtype is a string dtype"""
    return np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_)

def isdtype(dtype, kind: str | tuple[str, ...], xp=None) -> bool:
    """Compatibility wrapper for isdtype() from the array API standard.

    Unlike xp.isdtype(), kind must be a string.
    """
    if xp is None:
        xp = array_api_compat.get_namespace(dtype)
    
    if isinstance(kind, str):
        kind = (kind,)
    
    return any(xp.isdtype(dtype, k) for k in kind)

def result_type(*arrays_and_dtypes: np.typing.ArrayLike | np.typing.DTypeLike, xp=None) -> np.dtype:
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
    if xp is None:
        xp = array_api_compat.get_namespace(*arrays_and_dtypes)

    dtypes = []
    for array_or_dtype in arrays_and_dtypes:
        if hasattr(array_or_dtype, "dtype"):
            dtypes.append(array_or_dtype.dtype)
        else:
            dtypes.append(np.dtype(array_or_dtype))

    result = xp.result_type(*dtypes)

    for pair in PROMOTE_TO_OBJECT:
        if any(np.issubdtype(dt, pair[0]) for dt in dtypes) and any(np.issubdtype(dt, pair[1]) for dt in dtypes):
            return np.dtype(object)

    return result
