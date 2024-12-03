from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import is_extension_array_dtype
import xarray as xr
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.extension_array import PandasExtensionArray
from xarray.core.indexing import BasicIndexer, OuterIndexer, PandasIndexingAdapter, VectorizedIndexer, as_indexable
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import OrderedSet, _default, consolidate_dask_from_array_kwargs, decode_numpy_dict_values, drop_dims_from_indexers, either_dict_or_kwargs, emit_user_level_warning, ensure_us_time_resolution, infix_dims, is_dict_like, is_duck_array, is_duck_dask_array, maybe_coerce_to_str
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
from xarray.util.deprecation_helpers import deprecate_dims
NON_NUMPY_SUPPORTED_ARRAY_TYPES = (indexing.ExplicitlyIndexed, pd.Index, pd.api.extensions.ExtensionArray)
BASIC_INDEXING_TYPES = integer_types + (slice,)
if TYPE_CHECKING:
    from xarray.core.types import Dims, ErrorOptionsWithWarn, PadModeOptions, PadReflectOptions, QuantileMethods, Self, T_Chunks, T_DuckArray
    from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint
NON_NANOSECOND_WARNING = 'Converting non-nanosecond precision {case} values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.'

class MissingDimensionsError(ValueError):
    """Error class used when we can't safely guess a dimension name."""

def as_variable(obj: T_DuckArray | Any, name=None, auto_convert: bool=True) -> Variable | IndexVariable:
    """Convert an object into a Variable.

    Parameters
    ----------
    obj : object
        Object to convert into a Variable.

        - If the object is already a Variable, return a shallow copy.
        - Otherwise, if the object has 'dims' and 'data' attributes, convert
          it into a new Variable.
        - If all else fails, attempt to convert the object into a Variable by
          unpacking it into the arguments for creating a new Variable.
    name : str, optional
        If provided:

        - `obj` can be a 1D array, which is assumed to label coordinate values
          along a dimension of this given name.
        - Variables with name matching one of their dimensions are converted
          into `IndexVariable` objects.
    auto_convert : bool, optional
        For internal use only! If True, convert a "dimension" variable into
        an IndexVariable object (deprecated).

    Returns
    -------
    var : Variable
        The newly created variable.

    """
    pass

def _maybe_wrap_data(data):
    """
    Put pandas.Index and numpy.ndarray arguments in adapter objects to ensure
    they can be indexed properly.

    NumpyArrayAdapter, PandasIndexingAdapter and LazilyIndexedArray should
    all pass through unmodified.
    """
    pass

def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64, according to the pandas convention. For the time
    being, convert any non-nanosecond precision DatetimeIndex or TimedeltaIndex
    objects to nanosecond precision.  While pandas is relaxing this in version
    2.0.0, in xarray we will need to make sure we are ready to handle
    non-nanosecond precision datetimes or timedeltas in our code before allowing
    such values to pass through unchanged.  Converting to nanosecond precision
    through pandas.Series objects ensures that datetimes and timedeltas are
    within the valid date range for ns precision, as pandas will raise an error
    if they are not.
    """
    pass

def _possibly_convert_datetime_or_timedelta_index(data):
    """For the time being, convert any non-nanosecond precision DatetimeIndex or
    TimedeltaIndex objects to nanosecond precision.  While pandas is relaxing
    this in version 2.0.0, in xarray we will need to make sure we are ready to
    handle non-nanosecond precision datetimes or timedeltas in our code
    before allowing such values to pass through unchanged."""
    pass

def as_compatible_data(data: T_DuckArray | ArrayLike, fastpath: bool=False) -> T_DuckArray:
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    pass

def _as_array_or_item(data):
    """Return the given values as a numpy array, or as an individual item if
    it's a 0d datetime64 or timedelta64 array.

    Importantly, this function does not copy data if it is already an ndarray -
    otherwise, it will not be possible to update Variable values in place.

    This function mostly exists because 0-dimensional ndarrays with
    dtype=datetime64 are broken :(
    https://github.com/numpy/numpy/issues/4337
    https://github.com/numpy/numpy/issues/7619

    TODO: remove this (replace with np.asarray) once these issues are fixed
    """
    pass

class Variable(NamedArray, AbstractArray, VariableArithmetic):
    """A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single Array. A single Variable object is not fully
    described outside the context of its parent Dataset (if you want such a
    fully described object, use a DataArray instead).

    The main functional difference between Variables and numpy arrays is that
    numerical operations on Variables implement array broadcasting by dimension
    name. For example, adding an Variable with dimensions `('time',)` to
    another Variable with dimensions `('space',)` results in a new Variable
    with dimensions `('time', 'space')`. Furthermore, numpy reduce operations
    like ``mean`` or ``sum`` are overwritten to take a "dimension" argument
    instead of an "axis".

    Variables are light-weight objects used as the building block for datasets.
    They are more primitive objects, so operations with them provide marginally
    higher performance than using DataArrays. However, manipulating data in the
    form of a Dataset or DataArray should almost always be preferred, because
    they can use more complete metadata in context of coordinate labels.
    """
    __slots__ = ('_dims', '_data', '_attrs', '_encoding')

    def __init__(self, dims, data: T_DuckArray | ArrayLike, attrs=None, encoding=None, fastpath=False):
        """
        Parameters
        ----------
        dims : str or sequence of str
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions.
        data : array_like
            Data array which supports numpy-like data access.
        attrs : dict_like or None, optional
            Attributes to assign to the new variable. If None (default), an
            empty attribute dictionary is initialized.
        encoding : dict_like or None, optional
            Dictionary specifying how to encode this array's data into a
            serialized format like netCDF4. Currently used keys (for netCDF)
            include '_FillValue', 'scale_factor', 'add_offset' and 'dtype'.
            Well-behaved code to serialize a Variable should ignore
            unrecognized encoding items.
        """
        super().__init__(dims=dims, data=as_compatible_data(data, fastpath=fastpath), attrs=attrs)
        self._encoding = None
        if encoding is not None:
            self.encoding = encoding

    @property
    def data(self):
        """
        The Variable's data as an array. The underlying array type
        (e.g. dask, sparse, pint) is preserved.

        See Also
        --------
        Variable.to_numpy
        Variable.as_numpy
        Variable.values
        """
        pass

    def astype(self, dtype, *, order=None, casting=None, subok=None, copy=None, keep_attrs=True) -> Self:
        """
        Copy of the Variable object, with data cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout order of the result. ‘C’ means C order,
            ‘F’ means Fortran order, ‘A’ means ‘F’ order if all the arrays are
            Fortran contiguous, ‘C’ order otherwise, and ‘K’ means as close to
            the order the array elements appear in memory as possible.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.
        keep_attrs : bool, optional
            By default, astype keeps attributes. Set to False to remove
            attributes in the returned object.

        Returns
        -------
        out : same as object
            New object with data cast to the specified type.

        Notes
        -----
        The ``order``, ``casting``, ``subok`` and ``copy`` arguments are only passed
        through to the ``astype`` method of the underlying array when a value
        different than ``None`` is supplied.
        Make sure to only supply these arguments if the underlying array class
        supports them.

        See Also
        --------
        numpy.ndarray.astype
        dask.array.Array.astype
        sparse.COO.astype
        """
        pass

    @property
    def values(self):
        """The variable's data as a numpy.ndarray"""
        pass

    def to_base_variable(self) -> Variable:
        """Return this variable as a base xarray.Variable"""
        pass
    to_variable = utils.alias(to_base_variable, 'to_variable')

    def to_index_variable(self) -> IndexVariable:
        """Return this variable as an xarray.IndexVariable"""
        pass
    to_coord = utils.alias(to_index_variable, 'to_coord')

    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index"""
        pass

    def to_dict(self, data: bool | str='list', encoding: bool=False) -> dict[str, Any]:
        """Dictionary representation of variable."""
        pass

    def _broadcast_indexes(self, key):
        """Prepare an indexing key for an indexing operation.

        Parameters
        ----------
        key : int, slice, array-like, dict or tuple of integer, slice and array-like
            Any valid input for indexing.

        Returns
        -------
        dims : tuple
            Dimension of the resultant variable.
        indexers : IndexingTuple subclass
            Tuple of integer, array-like, or slices to use when indexing
            self._data. The type of this argument indicates the type of
            indexing to perform, either basic, outer or vectorized.
        new_order : Optional[Sequence[int]]
            Optional reordering to do on the result of indexing. If not None,
            the first len(new_order) indexing should be moved to these
            positions.
        """
        pass

    def _validate_indexers(self, key):
        """Make sanity checks"""
        pass

    def __getitem__(self, key) -> Self:
        """Return a new Variable object whose contents are consistent with
        getting the provided key from the underlying data.

        NB. __getitem__ and __setitem__ implement xarray-style indexing,
        where if keys are unlabeled arrays, we index the array orthogonally
        with them. If keys are labeled array (such as Variables), they are
        broadcasted with our usual scheme and then the array is indexed with
        the broadcasted key, like numpy's fancy indexing.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        dims, indexer, new_order = self._broadcast_indexes(key)
        indexable = as_indexable(self._data)
        data = indexing.apply_indexer(indexable, indexer)
        if new_order:
            data = np.moveaxis(data, range(len(new_order)), new_order)
        return self._finalize_indexing_result(dims, data)

    def _finalize_indexing_result(self, dims, data) -> Self:
        """Used by IndexVariable to return IndexVariable objects when possible."""
        pass

    def _getitem_with_mask(self, key, fill_value=dtypes.NA):
        """Index this Variable with -1 remapped to fill_value."""
        pass

    def __setitem__(self, key, value):
        """__setitem__ is overloaded to access the underlying numpy values with
        orthogonal indexing.

        See __getitem__ for more details.
        """
        dims, index_tuple, new_order = self._broadcast_indexes(key)
        if not isinstance(value, Variable):
            value = as_compatible_data(value)
            if value.ndim > len(dims):
                raise ValueError(f'shape mismatch: value array of shape {value.shape} could not be broadcast to indexing result with {len(dims)} dimensions')
            if value.ndim == 0:
                value = Variable((), value)
            else:
                value = Variable(dims[-value.ndim:], value)
        value = value.set_dims(dims).data
        if new_order:
            value = duck_array_ops.asarray(value)
            value = value[(len(dims) - value.ndim) * (np.newaxis,) + (Ellipsis,)]
            value = np.moveaxis(value, new_order, range(len(new_order)))
        indexable = as_indexable(self._data)
        indexing.set_with_indexer(indexable, index_tuple, value)

    @property
    def encoding(self) -> dict[Any, Any]:
        """Dictionary of encodings on this variable."""
        pass

    def drop_encoding(self) -> Self:
        """Return a new Variable without encoding."""
        pass

    def load(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return this variable.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        pass

    def compute(self, **kwargs):
        """Manually trigger loading of this variable's data from disk or a
        remote source into memory and return a new variable. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        pass

    def isel(self, indexers: Mapping[Any, Any] | None=None, missing_dims: ErrorOptionsWithWarn='raise', **indexers_kwargs: Any) -> Self:
        """Return a new array indexed along the specified dimension(s).

        Parameters
        ----------
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        obj : Array object
            A new Array with the selected data and dimensions. In general,
            the new variable's data will be a view of this variable's data,
            unless numpy fancy indexing was triggered by using an array
            indexer, in which case the data will be a copy.
        """
        pass

    def squeeze(self, dim=None):
        """Return a new object with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.

        Returns
        -------
        squeezed : same type as caller
            This object, but with with all or a subset of the dimensions of
            length 1 removed.

        See Also
        --------
        numpy.squeeze
        """
        pass

    def shift(self, shifts=None, fill_value=dtypes.NA, **shifts_kwargs):
        """
        Return a new Variable with shifted data.

        Parameters
        ----------
        shifts : mapping of the form {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value : scalar, optional
            Value to use for newly missing values
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but shifted data.
        """
        pass

    def pad(self, pad_width: Mapping[Any, int | tuple[int, int]] | None=None, mode: PadModeOptions='constant', stat_length: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, constant_values: float | tuple[float, float] | Mapping[Any, tuple[float, float]] | None=None, end_values: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, reflect_type: PadReflectOptions=None, keep_attrs: bool | None=None, **pad_width_kwargs: Any):
        """
        Return a new Variable with padded data.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : str, default: "constant"
            See numpy / Dask docs
        stat_length : int, tuple or mapping of hashable to tuple
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
        constant_values : scalar, tuple or mapping of hashable to tuple
            Used in 'constant'.  The values to set the padded values for each
            axis.
        end_values : scalar, tuple or mapping of hashable to tuple
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
        reflect_type : {"even", "odd"}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **pad_width_kwargs
            One of pad_width or pad_width_kwargs must be provided.

        Returns
        -------
        padded : Variable
            Variable with the same dimensions and attributes but padded data.
        """
        pass

    def roll(self, shifts=None, **shifts_kwargs):
        """
        Return a new Variable with rolld data.

        Parameters
        ----------
        shifts : mapping of hashable to int
            Integer offset to roll along each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Variable
            Variable with the same dimensions and attributes but rolled data.
        """
        pass

    @deprecate_dims
    def transpose(self, *dim: Hashable | ellipsis, missing_dims: ErrorOptionsWithWarn='raise') -> Self:
        """Return a new Variable object with transposed dimensions.

        Parameters
        ----------
        *dim : Hashable, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Variable:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        transposed : Variable
            The returned object has transposed data and dimensions with the
            same attributes as the original.

        Notes
        -----
        This operation returns a view of this variable's data. It is
        lazy for dask-backed Variables but not for numpy-backed Variables.

        See Also
        --------
        numpy.transpose
        """
        pass

    @deprecate_dims
    def set_dims(self, dim, shape=None):
        """Return a new variable with given set of dimensions.
        This method might be used to attach new dimension(s) to variable.

        When possible, this operation does not copy this variable's data.

        Parameters
        ----------
        dim : str or sequence of str or dict
            Dimensions to include on the new variable. If a dict, values are
            used to provide the sizes of new dimensions; otherwise, new
            dimensions are inserted with length 1.

        Returns
        -------
        Variable
        """
        pass

    @partial(deprecate_dims, old_name='dimensions')
    def stack(self, dim=None, **dim_kwargs):
        """
        Stack any number of existing dim into a single new dimension.

        New dim will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Parameters
        ----------
        dim : mapping of hashable to tuple of hashable
            Mapping of form new_name=(dim1, dim2, ...) describing the
            names of new dim, and the existing dim that
            they replace.
        **dim_kwargs
            The keyword arguments form of ``dim``.
            One of dim or dim_kwargs must be provided.

        Returns
        -------
        stacked : Variable
            Variable with the same attributes but stacked data.

        See Also
        --------
        Variable.unstack
        """
        pass

    def _unstack_once_full(self, dim: Mapping[Any, int], old_dim: Hashable) -> Self:
        """
        Unstacks the variable without needing an index.

        Unlike `_unstack_once`, this function requires the existing dimension to
        contain the full product of the new dimensions.
        """
        pass

    def _unstack_once(self, index: pd.MultiIndex, dim: Hashable, fill_value=dtypes.NA, sparse: bool=False) -> Self:
        """
        Unstacks this variable given an index to unstack and the name of the
        dimension to which the index refers.
        """
        pass

    @partial(deprecate_dims, old_name='dimensions')
    def unstack(self, dim=None, **dim_kwargs):
        """
        Unstack an existing dimension into multiple new dimensions.

        New dimensions will be added at the end, and the order of the data
        along each new dimension will be in contiguous (C) order.

        Note that unlike ``DataArray.unstack`` and ``Dataset.unstack``, this
        method requires the existing dimension to contain the full product of
        the new dimensions.

        Parameters
        ----------
        dim : mapping of hashable to mapping of hashable to int
            Mapping of the form old_dim={dim1: size1, ...} describing the
            names of existing dimensions, and the new dimensions and sizes
            that they map to.
        **dim_kwargs
            The keyword arguments form of ``dim``.
            One of dim or dim_kwargs must be provided.

        Returns
        -------
        unstacked : Variable
            Variable with the same attributes but unstacked data.

        See Also
        --------
        Variable.stack
        DataArray.unstack
        Dataset.unstack
        """
        pass

    def clip(self, min=None, max=None):
        """
        Return an array whose values are limited to ``[min, max]``.
        At least one of max or min must be given.

        Refer to `numpy.clip` for full documentation.

        See Also
        --------
        numpy.clip : equivalent function
        """
        pass

    def reduce(self, func: Callable[..., Any], dim: Dims=None, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, **kwargs) -> Variable:
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default `func` is
            applied over all dimensions.
        axis : int or Sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        pass

    @classmethod
    def concat(cls, variables, dim='concat_dim', positions=None, shortcut=False, combine_attrs='override'):
        """Concatenate variables along a new or existing dimension.

        Parameters
        ----------
        variables : iterable of Variable
            Arrays to stack together. Each variable is expected to have
            matching dimensions and shape except for along the stacked
            dimension.
        dim : str or DataArray, optional
            Name of the dimension to stack along. This can either be a new
            dimension name, in which case it is added along axis=0, or an
            existing dimension name, in which case the location of the
            dimension is unchanged. Where to insert the new dimension is
            determined by the first variable.
        positions : None or list of array-like, optional
            List of integer arrays which specifies the integer positions to
            which to assign each dataset along the concatenated dimension.
            If not supplied, objects are concatenated in the provided order.
        shortcut : bool, optional
            This option is used internally to speed-up groupby operations.
            If `shortcut` is True, some checks of internal consistency between
            arrays to concatenate are skipped.
        combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                          "override"}, default: "override"
            String indicating how to combine attrs of the objects being merged:

            - "drop": empty attrs on returned Dataset.
            - "identical": all attrs must be the same on every object.
            - "no_conflicts": attrs from all objects are combined, any that have
              the same name must also have the same value.
            - "drop_conflicts": attrs from all objects are combined, any that have
              the same name but different values are dropped.
            - "override": skip comparing and copy attrs from the first dataset to
              the result.

        Returns
        -------
        stacked : Variable
            Concatenated Variable formed by stacking all the supplied variables
            along the given dimension.
        """
        pass

    def equals(self, other, equiv=duck_array_ops.array_equiv):
        """True if two Variables have the same dimensions and values;
        otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for Variables
        does element-wise comparisons (like numpy.ndarrays).
        """
        pass

    def broadcast_equals(self, other, equiv=duck_array_ops.array_equiv):
        """True if two Variables have the values after being broadcast against
        each other; otherwise False.

        Variables can still be equal (like pandas objects) if they have NaN
        values in the same locations.
        """
        pass

    def identical(self, other, equiv=duck_array_ops.array_equiv):
        """Like equals, but also checks attributes."""
        pass

    def no_conflicts(self, other, equiv=duck_array_ops.array_notnull_equiv):
        """True if the intersection of two Variable's non-null data is
        equal; otherwise false.

        Variables can thus still be equal if there are locations where either,
        or both, contain NaN values.
        """
        pass

    def quantile(self, q: ArrayLike, dim: str | Sequence[Hashable] | None=None, method: QuantileMethods='linear', keep_attrs: bool | None=None, skipna: bool | None=None, interpolation: QuantileMethods | None=None) -> Self:
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float or sequence of float
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        method : str, default: "linear"
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points. The options sorted by their R
            type as summarized in the H&F paper [1]_ are:

                1. "inverted_cdf"
                2. "averaged_inverted_cdf"
                3. "closest_observation"
                4. "interpolated_inverted_cdf"
                5. "hazen"
                6. "weibull"
                7. "linear"  (default)
                8. "median_unbiased"
                9. "normal_unbiased"

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument
            was previously called "interpolation", renamed in accordance with numpy
            version 1.22.0.

        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
            dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanquantile, pandas.Series.quantile, Dataset.quantile
        DataArray.quantile

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        pass

    def rank(self, dim, pct=False):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If `pct`, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.

        Returns
        -------
        ranked : Variable

        See Also
        --------
        Dataset.rank, DataArray.rank
        """
        pass

    def rolling_window(self, dim, window, window_dim, center=False, fill_value=dtypes.NA):
        """
        Make a rolling_window along dim and add a new_dim to the last place.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rolling_window.
            For nd-rolling, should be list of dimensions.
        window : int
            Window size of the rolling
            For nd-rolling, should be list of integers.
        window_dim : str
            New name of the window dimension.
            For nd-rolling, should be list of strings.
        center : bool, default: False
            If True, pad fill_value for both ends. Otherwise, pad in the head
            of the axis.
        fill_value
            value to be filled.

        Returns
        -------
        Variable that is a view of the original array with a added dimension of
        size w.
        The return dim: self.dims + (window_dim, )
        The return shape: self.shape + (window, )

        Examples
        --------
        >>> v = Variable(("a", "b"), np.arange(8).reshape((2, 4)))
        >>> v.rolling_window("b", 3, "window_dim")
        <xarray.Variable (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])

        >>> v.rolling_window("b", 3, "window_dim", center=True)
        <xarray.Variable (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        """
        pass

    def coarsen(self, windows, func, boundary='exact', side='left', keep_attrs=None, **kwargs):
        """
        Apply reduction function.
        """
        pass

    def coarsen_reshape(self, windows, boundary, side):
        """
        Construct a reshaped-array for coarsen
        """
        pass

    def isnull(self, keep_attrs: bool | None=None):
        """Test each value in the array for whether it is a missing value.

        Returns
        -------
        isnull : Variable
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.isnull

        Examples
        --------
        >>> var = xr.Variable("x", [1, np.nan, 3])
        >>> var
        <xarray.Variable (x: 3)> Size: 24B
        array([ 1., nan,  3.])
        >>> var.isnull()
        <xarray.Variable (x: 3)> Size: 3B
        array([False,  True, False])
        """
        pass

    def notnull(self, keep_attrs: bool | None=None):
        """Test each value in the array for whether it is not a missing value.

        Returns
        -------
        notnull : Variable
            Same type and shape as object, but the dtype of the data is bool.

        See Also
        --------
        pandas.notnull

        Examples
        --------
        >>> var = xr.Variable("x", [1, np.nan, 3])
        >>> var
        <xarray.Variable (x: 3)> Size: 24B
        array([ 1., nan,  3.])
        >>> var.notnull()
        <xarray.Variable (x: 3)> Size: 3B
        array([ True, False,  True])
        """
        pass

    @property
    def imag(self) -> Variable:
        """
        The imaginary part of the variable.

        See Also
        --------
        numpy.ndarray.imag
        """
        pass

    @property
    def real(self) -> Variable:
        """
        The real part of the variable.

        See Also
        --------
        numpy.ndarray.real
        """
        pass

    def __array_wrap__(self, obj, context=None):
        return Variable(self.dims, obj)

    def _to_numeric(self, offset=None, datetime_unit=None, dtype=float):
        """A (private) method to convert datetime array to numeric dtype
        See duck_array_ops.datetime_to_numeric
        """
        pass

    def _unravel_argminmax(self, argminmax: str, dim: Dims, axis: int | None, keep_attrs: bool | None, skipna: bool | None) -> Variable | dict[Hashable, Variable]:
        """Apply argmin or argmax over one or more dimensions, returning the result as a
        dict of DataArray that can be passed directly to isel.
        """
        pass

    def argmin(self, dim: Dims=None, axis: int | None=None, keep_attrs: bool | None=None, skipna: bool | None=None) -> Variable | dict[Hashable, Variable]:
        """Index or indices of the minimum of the Variable over one or more dimensions.
        If a sequence is passed to 'dim', then result returned as dict of Variables,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a Variable with dtype int.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : "...", str, Iterable of Hashable or None, optional
            The dimensions over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Variable or dict of Variable

        See Also
        --------
        DataArray.argmin, DataArray.idxmin
        """
        pass

    def argmax(self, dim: Dims=None, axis: int | None=None, keep_attrs: bool | None=None, skipna: bool | None=None) -> Variable | dict[Hashable, Variable]:
        """Index or indices of the maximum of the Variable over one or more dimensions.
        If a sequence is passed to 'dim', then result returned as dict of Variables,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a Variable with dtype int.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : "...", str, Iterable of Hashable or None, optional
            The dimensions over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Variable or dict of Variable

        See Also
        --------
        DataArray.argmax, DataArray.idxmax
        """
        pass

    def _as_sparse(self, sparse_format=_default, fill_value=_default) -> Variable:
        """
        Use sparse-array as backend.
        """
        pass

    def _to_dense(self) -> Variable:
        """
        Change backend from sparse to np.array.
        """
        pass

    def chunk(self, chunks: T_Chunks={}, name: str | None=None, lock: bool | None=None, inline_array: bool | None=None, chunked_array_type: str | ChunkManagerEntrypoint[Any] | None=None, from_array_kwargs: Any=None, **chunks_kwargs: Any) -> Self:
        """Coerce this array's data into a dask array with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple or dict, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        name : str, optional
            Used to generate the name for this array in the internal dask
            graph. Does not need not be unique.
        lock : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        inline_array : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        chunked_array_type: str, optional
            Which chunked array type to coerce this datasets' arrays to.
            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntrypoint` system.
            Experimental API that should not be relied upon.
        from_array_kwargs: dict, optional
            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
            For example, with dask as the default chunked array type, this method would pass additional kwargs
            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
        **chunks_kwargs : {dim: chunks, ...}, optional
            The keyword arguments form of ``chunks``.
            One of chunks or chunks_kwargs must be provided.

        Returns
        -------
        chunked : xarray.Variable

        See Also
        --------
        Variable.chunks
        Variable.chunksizes
        xarray.unify_chunks
        dask.array.from_array
        """
        pass

class IndexVariable(Variable):
    """Wrapper for accommodating a pandas.Index in an xarray.Variable.

    IndexVariable preserve loaded values in the form of a pandas.Index instead
    of a NumPy array. Hence, their values are immutable and must always be one-
    dimensional.

    They also have a name property, which is the name of their sole dimension
    unless another name is given.
    """
    __slots__ = ()
    _data: PandasIndexingAdapter

    def __init__(self, dims, data, attrs=None, encoding=None, fastpath=False):
        super().__init__(dims, data, attrs, encoding, fastpath)
        if self.ndim != 1:
            raise ValueError(f'{type(self).__name__} objects must be 1-dimensional')
        if not isinstance(self._data, PandasIndexingAdapter):
            self._data = PandasIndexingAdapter(self._data)

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._dims, self._data.array, self._attrs or None))

    def __setitem__(self, key, value):
        raise TypeError(f'{type(self).__name__} values cannot be modified')

    @classmethod
    def concat(cls, variables, dim='concat_dim', positions=None, shortcut=False, combine_attrs='override'):
        """Specialized version of Variable.concat for IndexVariable objects.

        This exists because we want to avoid converting Index objects to NumPy
        arrays, if possible.
        """
        pass

    def copy(self, deep: bool=True, data: T_DuckArray | ArrayLike | None=None):
        """Returns a copy of this object.

        `deep` is ignored since data is stored in the form of
        pandas.Index, which is already immutable. Dimensions, attributes
        and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: True
            Deep is ignored when data is given. Whether the data array is
            loaded into memory and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.

        Returns
        -------
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.
        """
        pass

    def to_index_variable(self) -> IndexVariable:
        """Return this variable as an xarray.IndexVariable"""
        pass
    to_coord = utils.alias(to_index_variable, 'to_coord')

    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index"""
        pass

    @property
    def level_names(self) -> list[str] | None:
        """Return MultiIndex level names or None if this IndexVariable has no
        MultiIndex.
        """
        pass

    def get_level_variable(self, level):
        """Return a new IndexVariable from a given MultiIndex level."""
        pass

def _broadcast_compat_variables(*variables):
    """Create broadcast compatible variables, with the same dimensions.

    Unlike the result of broadcast_variables(), some variables may have
    dimensions of size 1 instead of the size of the broadcast dimension.
    """
    pass

def broadcast_variables(*variables: Variable) -> tuple[Variable, ...]:
    """Given any number of variables, return variables with matching dimensions
    and broadcast data.

    The data on the returned variables will be a view of the data on the
    corresponding original arrays, but dimensions will be reordered and
    inserted so that both broadcast arrays have the same dimensions. The new
    dimensions are sorted in order of appearance in the first variable's
    dimensions followed by the second variable's dimensions.
    """
    pass

def concat(variables, dim='concat_dim', positions=None, shortcut=False, combine_attrs='override'):
    """Concatenate variables along a new or existing dimension.

    Parameters
    ----------
    variables : iterable of Variable
        Arrays to stack together. Each variable is expected to have
        matching dimensions and shape except for along the stacked
        dimension.
    dim : str or DataArray, optional
        Name of the dimension to stack along. This can either be a new
        dimension name, in which case it is added along axis=0, or an
        existing dimension name, in which case the location of the
        dimension is unchanged. Where to insert the new dimension is
        determined by the first variable.
    positions : None or list of array-like, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    shortcut : bool, optional
        This option is used internally to speed-up groupby operations.
        If `shortcut` is True, some checks of internal consistency between
        arrays to concatenate are skipped.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"}, default: "override"
        String indicating how to combine attrs of the objects being merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

    Returns
    -------
    stacked : Variable
        Concatenated Variable formed by stacking all the supplied variables
        along the given dimension.
    """
    if not variables:
        raise ValueError("No variables were supplied to concat")

    first_var = variables[0]
    new_dim = dim not in first_var.dims

    if new_dim:
        concat_dim = dim
        dim_axis = 0
    else:
        concat_dim = first_var.dims.index(dim)
        dim_axis = first_var.get_axis_num(dim)

    variables = [var.expand_dims({dim: 1}) if new_dim and dim not in var.dims else var for var in variables]

    if not shortcut:
        variables = align_variables(*variables)

    data = duck_array_ops.concatenate([var.data for var in variables], axis=dim_axis)

    if positions is not None:
        indices = nputils.inverse_permutation(np.concatenate(positions))
        data = duck_array_ops.take(data, indices, axis=dim_axis)

    if new_dim:
        dims = (dim,) + first_var.dims
    else:
        dims = first_var.dims

    attrs = merge_attrs([var.attrs for var in variables], combine_attrs)

    return Variable(dims, data, attrs)

def calculate_dimensions(variables: Mapping[Any, Variable]) -> dict[Hashable, int]:
    """Calculate the dimensions corresponding to a set of variables.

    Returns dictionary mapping from dimension names to sizes. Raises ValueError
    if any of the dimension sizes conflict.
    """
    pass
