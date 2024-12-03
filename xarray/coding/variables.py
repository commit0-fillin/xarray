"""Coders for individual Variable objects."""
from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
if TYPE_CHECKING:
    T_VarTuple = tuple[tuple[Hashable, ...], Any, dict, dict]
    T_Name = Union[Hashable, None]

class SerializationWarning(RuntimeWarning):
    """Warnings about encoding/decoding issues in serialization."""

class VariableCoder:
    """Base class for encoding and decoding transformations on variables.

    We use coders for transforming variables between xarray's data model and
    a format suitable for serialization. For example, coders apply CF
    conventions for how data should be represented in netCDF files.

    Subclasses should implement encode() and decode(), which should satisfy
    the identity ``coder.decode(coder.encode(variable)) == variable``. If any
    options are necessary, they should be implemented as arguments to the
    __init__ method.

    The optional name argument to encode() and decode() exists solely for the
    sake of better error messages, and should correspond to the name of
    variables in the underlying store.
    """

    def encode(self, variable: Variable, name: T_Name=None) -> Variable:
        """Convert an encoded variable to a decoded variable"""
        if name is None:
            name = variable.name
        
        data = variable.data
        attrs = variable.attrs.copy()
        encoding = variable.encoding.copy()

        if "dtype" in encoding:
            data = data.astype(encoding["dtype"])
        
        if "scale_factor" in encoding or "add_offset" in encoding:
            data = (data - encoding.get("add_offset", 0)) / encoding.get("scale_factor", 1)
        
        if "_FillValue" in encoding:
            data = duck_array_ops.where(data == encoding["_FillValue"], np.nan, data)
        
        return Variable(variable.dims, data, attrs, encoding)

    def decode(self, variable: Variable, name: T_Name=None) -> Variable:
        """Convert a decoded variable to an encoded variable"""
        if name is None:
            name = variable.name
        
        data = variable.data
        attrs = variable.attrs.copy()
        encoding = variable.encoding.copy()

        if "dtype" in encoding:
            data = data.astype(encoding["dtype"])
        
        if "scale_factor" in encoding or "add_offset" in encoding:
            data = data * encoding.get("scale_factor", 1) + encoding.get("add_offset", 0)
        
        if "_FillValue" in encoding:
            fill_value = encoding["_FillValue"]
            data = duck_array_ops.where(np.isnan(data), fill_value, data)
        
        return Variable(variable.dims, data, attrs, encoding)

class _ElementwiseFunctionArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Lazily computed array holding values of elemwise-function.

    Do not construct this object directly: call lazy_elemwise_func instead.

    Values are computed upon indexing or coercion to a NumPy array.
    """

    def __init__(self, array, func: Callable, dtype: np.typing.DTypeLike):
        assert not is_chunked_array(array)
        self.array = indexing.as_indexable(array)
        self.func = func
        self._dtype = dtype

    def __getitem__(self, key):
        return type(self)(self.array[key], self.func, self.dtype)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.array!r}, func={self.func!r}, dtype={self.dtype!r})'

class NativeEndiannessArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Decode arrays on the fly from non-native to native endianness

    This is useful for decoding arrays from netCDF3 files (which are all
    big endian) into native endianness, so they can be used with Cython
    functions, such as those found in bottleneck and pandas.

    >>> x = np.arange(5, dtype=">i2")

    >>> x.dtype
    dtype('>i2')

    >>> NativeEndiannessArray(x).dtype
    dtype('int16')

    >>> indexer = indexing.BasicIndexer((slice(None),))
    >>> NativeEndiannessArray(x)[indexer].dtype
    dtype('int16')
    """
    __slots__ = ('array',)

    def __init__(self, array) -> None:
        self.array = indexing.as_indexable(array)

    def __getitem__(self, key) -> np.ndarray:
        return np.asarray(self.array[key], dtype=self.dtype)

class BoolTypeArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Decode arrays on the fly from integer to boolean datatype

    This is useful for decoding boolean arrays from integer typed netCDF
    variables.

    >>> x = np.array([1, 0, 1, 1, 0], dtype="i1")

    >>> x.dtype
    dtype('int8')

    >>> BoolTypeArray(x).dtype
    dtype('bool')

    >>> indexer = indexing.BasicIndexer((slice(None),))
    >>> BoolTypeArray(x)[indexer].dtype
    dtype('bool')
    """
    __slots__ = ('array',)

    def __init__(self, array) -> None:
        self.array = indexing.as_indexable(array)

    def __getitem__(self, key) -> np.ndarray:
        return np.asarray(self.array[key], dtype=self.dtype)

def lazy_elemwise_func(array, func: Callable, dtype: np.typing.DTypeLike):
    """Lazily apply an element-wise function to an array.
    Parameters
    ----------
    array : any valid value of Variable._data
    func : callable
        Function to apply to indexed slices of an array. For use with dask,
        this should be a pickle-able object.
    dtype : coercible to np.dtype
        Dtype for the result of this function.

    Returns
    -------
    Either a dask.array.Array or _ElementwiseFunctionArray.
    """
    if is_duck_dask_array(array):
        import dask.array as da
        return da.map_overlap(func, array, dtype=dtype)
    else:
        return _ElementwiseFunctionArray(array, func, dtype)

def pop_to(source: MutableMapping, dest: MutableMapping, key: Hashable, name: T_Name=None) -> Any:
    """
    A convenience function which pops a key k from source to dest.
    None values are not passed on.  If k already exists in dest an
    error is raised.
    """
    value = source.pop(key, None)
    if value is not None:
        if key in dest:
            raise ValueError(f"'{key}' already exists in destination")
        dest[key] = value
    return value

def _apply_mask(data: np.ndarray, encoded_fill_values: list, decoded_fill_value: Any, dtype: np.typing.DTypeLike) -> np.ndarray:
    """Mask all matching values in a NumPy arrays."""
    if not encoded_fill_values:
        return data

    condition = False
    for fill_value in encoded_fill_values:
        condition |= data == fill_value

    return np.where(condition, decoded_fill_value, data).astype(dtype)

def _check_fill_values(attrs, name, dtype):
    """Check _FillValue and missing_value if available.

    Return dictionary with raw fill values and set with encoded fill values.

    Issue SerializationWarning if appropriate.
    """
    fill_values = {}
    encoded_fill_values = set()

    for attr_name in ['_FillValue', 'missing_value']:
        value = attrs.get(attr_name)
        if value is not None:
            fill_values[attr_name] = value
            if np.array(value).dtype.kind != 'O':
                encoded_fill_values.add(np.array(value).astype(dtype).item())

    if len(fill_values) == 2:
        if fill_values['_FillValue'] != fill_values['missing_value']:
            warnings.warn(
                f"Variable '{name}' has multiple fill values {fill_values}, "
                "but encoding can only support one. Prioritizing _FillValue.",
                SerializationWarning,
                stacklevel=3,
            )

    return fill_values, encoded_fill_values

class CFMaskCoder(VariableCoder):
    """Mask or unmask fill values according to CF conventions."""

def _choose_float_dtype(dtype: np.dtype, mapping: MutableMapping) -> type[np.floating[Any]]:
    """Return a float dtype that can losslessly represent `dtype` values."""
    if np.issubdtype(dtype, np.floating):
        return dtype.type
    elif np.issubdtype(dtype, np.integer):
        return np.float64 if dtype.itemsize > 4 else np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

class CFScaleOffsetCoder(VariableCoder):
    """Scale and offset variables according to CF conventions.

    Follows the formula:
        decode_values = encoded_values * scale_factor + add_offset
    """

class UnsignedIntegerCoder(VariableCoder):
    pass

class DefaultFillvalueCoder(VariableCoder):
    """Encode default _FillValue if needed."""

class BooleanCoder(VariableCoder):
    """Code boolean values."""

class EndianCoder(VariableCoder):
    """Decode Endianness to native."""

class NonStringCoder(VariableCoder):
    """Encode NonString variables if dtypes differ."""

class ObjectVLenStringCoder(VariableCoder):
    pass

class NativeEnumCoder(VariableCoder):
    """Encode Enum into variable dtype metadata."""
