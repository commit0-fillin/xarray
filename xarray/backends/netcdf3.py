from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
_specialchars = '_.@+- !"#$%&\\()*,:;<=>?[]^`{|}~'
_reserved_names = {'byte', 'char', 'short', 'ushort', 'int', 'uint', 'int64', 'uint64', 'float', 'real', 'double', 'bool', 'string'}
_nc3_dtype_coercions = {'int64': 'int32', 'uint64': 'int32', 'uint32': 'int32', 'uint16': 'int16', 'uint8': 'int8', 'bool': 'int8'}
STRING_ENCODING = 'utf-8'
COERCION_VALUE_ERROR = "could not safely cast array from {dtype} to {new_dtype}. While it is not always the case, a common reason for this is that xarray has deemed it safest to encode np.datetime64[ns] or np.timedelta64[ns] values with int64 values representing units of 'nanoseconds'. This is either due to the fact that the times are known to require nanosecond precision for an accurate round trip, or that the times are unknown prior to writing due to being contained in a chunked array. Ways to work around this are either to use a backend that supports writing int64 values, or to manually specify the encoding['units'] and encoding['dtype'] (e.g. 'seconds since 1970-01-01' and np.dtype('int32')) on the time variable(s) such that the times can be serialized in a netCDF3 file (note that depending on the situation, however, this latter option may result in an inaccurate round trip)."

def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the dtype conversions as specified by the
    ``_nc3_dtype_coercions`` mapping:
        int64  -> int32
        uint64 -> int32
        uint32 -> int32
        uint16 -> int16
        uint8  -> int8
        bool   -> int8

    Data is checked for equality, or equivalence (non-NaN values) using the
    ``(cast_array == original_array).all()``.
    """
    dtype = arr.dtype
    if dtype.name in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype.name]
        cast_arr = arr.astype(new_dtype)
        if np.array_equal(cast_arr, arr) or (np.isnan(cast_arr) == np.isnan(arr)).all():
            return cast_arr
        else:
            raise ValueError(COERCION_VALUE_ERROR.format(dtype=dtype, new_dtype=new_dtype))
    return arr

def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return c.isalnum() or len(c.encode('utf-8')) > 1

def is_valid_nc3_name(s):
    """Test whether an object can be validly converted to a netCDF-3
    dimension, variable or attribute name

    Earlier versions of the netCDF C-library reference implementation
    enforced a more restricted set of characters in creating new names,
    but permitted reading names containing arbitrary bytes. This
    specification extends the permitted characters in names to include
    multi-byte UTF-8 encoded Unicode and additional printing characters
    from the US-ASCII alphabet. The first character of a name must be
    alphanumeric, a multi-byte UTF-8 character, or '_' (reserved for
    special names with meaning to implementations, such as the
    "_FillValue" attribute). Subsequent characters may also include
    printing special characters, except for '/' which is not allowed in
    names. Names that have trailing space characters are also not
    permitted.
    """
    if not isinstance(s, str):
        return False
    if not s:
        return False
    if s in _reserved_names:
        return False
    if s[-1].isspace():
        return False
    if '/' in s:
        return False
    
    first = s[0]
    if not (_isalnumMUTF8(first) or first == '_'):
        return False
    
    for c in s[1:]:
        if not (_isalnumMUTF8(c) or c in _specialchars):
            return False
    
    return True
