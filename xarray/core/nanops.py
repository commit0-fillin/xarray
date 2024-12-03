from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import astype, count, fillna, isnull, sum_where, where, where_method

def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """
    if axis is None:
        if np.any(mask):
            return np.nan
    else:
        if isinstance(axis, tuple):
            for ax in axis:
                mask = mask.all(axis=ax)
        else:
            mask = mask.all(axis=axis)

        if mask.ndim > 0:
            null_mask = (mask | (np.isnan(result) & (result != np.inf)))
            if min_count > 1:
                count = np.sum(~mask, axis=axis)
                null_mask |= (count < min_count)

            result[null_mask] = np.nan

    return result

def _nan_argminmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanargmin, nanargmax for object arrays. Always return integer
    type
    """
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    result = func(value, axis=axis, **kwargs)

    if isinstance(result, tuple):
        # In case func returns both values and indices
        result = result[0]

    return _maybe_null_out(result, axis, valid_count == 0).astype(int)

def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanmin and nanmax for object array"""
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    result = func(value, axis=axis, **kwargs)
    return _maybe_null_out(result, axis, valid_count == 0)

def _nanmean_ddof_object(ddof, value, axis=None, dtype=None, **kwargs):
    """In house nanmean. ddof argument will be used in _nanvar method"""
    valid_count = count(value, axis=axis)
    value = where(pandas_isnull(value), 0, value)
    
    if dtype is not None:
        value = value.astype(dtype)
    
    the_sum = np.sum(value, axis=axis, **kwargs)
    
    if axis is not None and valid_count.ndim < the_sum.ndim:
        valid_count = valid_count.reshape(valid_count.shape + (1,) * (the_sum.ndim - valid_count.ndim))
    
    return _maybe_null_out(the_sum / (valid_count - ddof), axis, valid_count == 0)
