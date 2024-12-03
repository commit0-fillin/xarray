from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Generic, cast
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from xarray.core.types import DTypeLikeSave, T_ExtensionArray
HANDLED_EXTENSION_ARRAY_FUNCTIONS: dict[Callable, Callable] = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for PandasExtensionArray objects."""
    def decorator(func):
        HANDLED_EXTENSION_ARRAY_FUNCTIONS[numpy_function] = func
        return func
    return decorator

class PandasExtensionArray(Generic[T_ExtensionArray]):
    array: T_ExtensionArray

    def __init__(self, array: T_ExtensionArray):
        """NEP-18 compliant wrapper for pandas extension arrays.

        Parameters
        ----------
        array : T_ExtensionArray
            The array to be wrapped upon e.g,. :py:class:`xarray.Variable` creation.
        ```
        """
        if not isinstance(array, pd.api.extensions.ExtensionArray):
            raise TypeError(f'{array} is not an pandas ExtensionArray.')
        self.array = array

    def __array_function__(self, func, types, args, kwargs):

        def replace_duck_with_extension_array(args) -> list:
            args_as_list = list(args)
            for index, value in enumerate(args_as_list):
                if isinstance(value, PandasExtensionArray):
                    args_as_list[index] = value.array
                elif isinstance(value, tuple):
                    args_as_list[index] = tuple(replace_duck_with_extension_array(value))
                elif isinstance(value, list):
                    args_as_list[index] = replace_duck_with_extension_array(value)
            return args_as_list
        args = tuple(replace_duck_with_extension_array(args))
        if func not in HANDLED_EXTENSION_ARRAY_FUNCTIONS:
            return func(*args, **kwargs)
        res = HANDLED_EXTENSION_ARRAY_FUNCTIONS[func](*args, **kwargs)
        if is_extension_array_dtype(res):
            return type(self)[type(res)](res)
        return res

    def __array_ufunc__(ufunc, method, *inputs, **kwargs):
        return ufunc(*inputs, **kwargs)

    def __repr__(self):
        return f'{type(self)}(array={repr(self.array)})'

    def __getattr__(self, attr: str) -> object:
        return getattr(self.array, attr)

    def __getitem__(self, key) -> PandasExtensionArray[T_ExtensionArray]:
        item = self.array[key]
        if is_extension_array_dtype(item):
            return type(self)(item)
        if np.isscalar(item):
            return type(self)(type(self.array)([item]))
        return item

    def __setitem__(self, key, val):
        self.array[key] = val

    def __eq__(self, other):
        if isinstance(other, PandasExtensionArray):
            return self.array == other.array
        return self.array == other

    def __ne__(self, other):
        return ~(self == other)

    def __len__(self):
        return len(self.array)
