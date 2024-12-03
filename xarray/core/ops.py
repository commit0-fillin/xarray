"""Define core operations for xarray objects.

TODO(shoyer): rewrite this module, making use of xarray.core.computation,
NumPy's __array_ufunc__ and mixin classes instead of the unintuitive "inject"
functions.
"""
from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
try:
    import bottleneck as bn
    has_bottleneck = True
except ImportError:
    bn = np
    has_bottleneck = False
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'and', 'xor', 'or', 'lshift', 'rshift']
NUMPY_SAME_METHODS = ['item', 'searchsorted']
REDUCE_METHODS = ['all', 'any']
NAN_REDUCE_METHODS = ['max', 'min', 'mean', 'prod', 'sum', 'std', 'var', 'median']
_CUM_DOCSTRING_TEMPLATE = 'Apply `{name}` along some dimension of {cls}.\n\nParameters\n----------\n{extra_args}\nskipna : bool, optional\n    If True, skip missing values (as marked by NaN). By default, only\n    skips missing values for float dtypes; other dtypes either do not\n    have a sentinel missing value (int) or skipna=True has not been\n    implemented (object, datetime64 or timedelta64).\nkeep_attrs : bool, optional\n    If True, the attributes (`attrs`) will be copied from the original\n    object to the new one.  If False (default), the new object will be\n    returned without attributes.\n**kwargs : dict\n    Additional keyword arguments passed on to `{name}`.\n\nReturns\n-------\ncumvalue : {cls}\n    New {cls} object with `{name}` applied to its data along the\n    indicated dimension.\n'
_REDUCE_DOCSTRING_TEMPLATE = "Reduce this {cls}'s data by applying `{name}` along some dimension(s).\n\nParameters\n----------\n{extra_args}{skip_na_docs}{min_count_docs}\nkeep_attrs : bool, optional\n    If True, the attributes (`attrs`) will be copied from the original\n    object to the new one.  If False (default), the new object will be\n    returned without attributes.\n**kwargs : dict\n    Additional keyword arguments passed on to the appropriate array\n    function for calculating `{name}` on this object's data.\n\nReturns\n-------\nreduced : {cls}\n    New {cls} object with `{name}` applied to its data and the\n    indicated dimension(s) removed.\n"
_SKIPNA_DOCSTRING = '\nskipna : bool, optional\n    If True, skip missing values (as marked by NaN). By default, only\n    skips missing values for float dtypes; other dtypes either do not\n    have a sentinel missing value (int) or skipna=True has not been\n    implemented (object, datetime64 or timedelta64).'
_MINCOUNT_DOCSTRING = "\nmin_count : int, default: None\n    The required number of valid values to perform the operation. If\n    fewer than min_count non-NA values are present the result will be\n    NA. Only used if skipna is set to True or defaults to True for the\n    array's dtype. New in version 0.10.8: Added with the default being\n    None. Changed in version 0.17.0: if specified on an integer array\n    and skipna=True, the result will be a float array."

def fillna(data, other, join='left', dataset_join='left'):
    """Fill missing values in this object with data from the other object.
    Follows normal broadcasting and alignment rules.

    Parameters
    ----------
    join : {"outer", "inner", "left", "right"}, optional
        Method for joining the indexes of the passed objects along each
        dimension
        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {"outer", "inner", "left", "right"}, optional
        Method for joining variables of Dataset objects with mismatched
        data variables.
        - "outer": take variables from both Dataset objects
        - "inner": take only overlapped variables
        - "left": take only variables from the first object
        - "right": take only variables from the last object
    """
    from xarray.core.alignment import align
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    
    if isinstance(data, DataArray):
        return data.fillna(other)
    elif isinstance(data, Dataset):
        if isinstance(other, Dataset):
            other_vars = other.data_vars
        else:
            other_vars = {k: other for k in data.data_vars}
        
        aligned_data, aligned_other = align(data, other, join=join, copy=False)
        
        for k in aligned_data.data_vars:
            if k in other_vars:
                aligned_data[k] = aligned_data[k].fillna(aligned_other[k])
        
        return aligned_data
    else:
        raise TypeError("fillna only works with DataArray or Dataset objects")

def where_method(self, cond, other=dtypes.NA):
    """Return elements from `self` or `other` depending on `cond`.

    Parameters
    ----------
    cond : DataArray or Dataset with boolean dtype
        Locations at which to preserve this objects values.
    other : scalar, DataArray or Dataset, optional
        Value to use for locations in this object where ``cond`` is False.
        By default, inserts missing values.

    Returns
    -------
    Same type as caller.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    
    if isinstance(self, DataArray):
        return self.where(cond, other)
    elif isinstance(self, Dataset):
        if isinstance(cond, Dataset):
            coords = self.coords
            cond_vars = cond.data_vars
            other_vars = other.data_vars if isinstance(other, Dataset) else {}

            ds = Dataset(coords=coords)
            for k in self.data_vars:
                cond_k = cond_vars.get(k, cond)
                other_k = other_vars.get(k, other)
                ds[k] = self[k].where(cond_k, other_k)
            return ds
        else:
            return self.where(cond, other)
    else:
        raise TypeError("where_method only works with DataArray or Dataset objects")
NON_INPLACE_OP = {get_op('i' + name): get_op(name) for name in NUM_BINARY_OPS}
argsort = _method_wrapper('argsort')
conj = _method_wrapper('conj')
conjugate = _method_wrapper('conjugate')
round_ = _func_slash_method_wrapper(duck_array_ops.around, name='round')

class IncludeReduceMethods:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, '_reduce_method', None):
            inject_reduce_methods(cls)

class IncludeNumpySameMethods:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        inject_numpy_same(cls)
