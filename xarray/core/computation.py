"""
Functions for applying functions that act on arrays to xarray's labeled data.
"""
from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
if TYPE_CHECKING:
    from xarray.core.coordinates import Coordinates
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import CombineAttrsOptions, JoinOptions
    MissingCoreDimOptions = Literal['raise', 'copy', 'drop']
_NO_FILL_VALUE = utils.ReprObject('<no-fill-value>')
_DEFAULT_NAME = utils.ReprObject('<default-name>')
_JOINS_WITHOUT_FILL_VALUES = frozenset({'inner', 'exact'})

def _first_of_type(args, kind):
    """Return either first object of type 'kind' or raise if not found."""
    for arg in args:
        if isinstance(arg, kind):
            return arg
    raise ValueError(f"No object of type {kind} found in arguments")

def _all_of_type(args, kind):
    """Return all objects of type 'kind'"""
    return [arg for arg in args if isinstance(arg, kind)]

class _UFuncSignature:
    """Core dimensions signature for a given function.

    Based on the signature provided by generalized ufuncs in NumPy.

    Attributes
    ----------
    input_core_dims : tuple[tuple]
        Core dimension names on each input variable.
    output_core_dims : tuple[tuple]
        Core dimension names on each output variable.
    """
    __slots__ = ('input_core_dims', 'output_core_dims', '_all_input_core_dims', '_all_output_core_dims', '_all_core_dims')

    def __init__(self, input_core_dims, output_core_dims=((),)):
        self.input_core_dims = tuple((tuple(a) for a in input_core_dims))
        self.output_core_dims = tuple((tuple(a) for a in output_core_dims))
        self._all_input_core_dims = None
        self._all_output_core_dims = None
        self._all_core_dims = None

    def __eq__(self, other):
        try:
            return self.input_core_dims == other.input_core_dims and self.output_core_dims == other.output_core_dims
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f'{type(self).__name__}({list(self.input_core_dims)!r}, {list(self.output_core_dims)!r})'

    def __str__(self):
        lhs = ','.join(('({})'.format(','.join(dims)) for dims in self.input_core_dims))
        rhs = ','.join(('({})'.format(','.join(dims)) for dims in self.output_core_dims))
        return f'{lhs}->{rhs}'

    def to_gufunc_string(self, exclude_dims=frozenset()):
        """Create an equivalent signature string for a NumPy gufunc.

        Unlike __str__, handles dimensions that don't map to Python
        identifiers.

        Also creates unique names for input_core_dims contained in exclude_dims.
        """
        def _make_dim_string(dims):
            return ','.join(f'd{i}' if d in exclude_dims else d
                            for i, d in enumerate(dims))

        input_dims = [_make_dim_string(dims) for dims in self.input_core_dims]
        output_dims = [_make_dim_string(dims) for dims in self.output_core_dims]

        return f"({'),('.join(input_dims)})->({'),('.join(output_dims)})"

def build_output_coords_and_indexes(args: Iterable[Any], signature: _UFuncSignature, exclude_dims: Set=frozenset(), combine_attrs: CombineAttrsOptions='override') -> tuple[list[dict[Any, Variable]], list[dict[Any, Index]]]:
    """Build output coordinates and indexes for an operation.

    Parameters
    ----------
    args : Iterable
        List of raw operation arguments. Any valid types for xarray operations
        are OK, e.g., scalars, Variable, DataArray, Dataset.
    signature : _UfuncSignature
        Core dimensions signature for the operation.
    exclude_dims : set, optional
        Dimensions excluded from the operation. Coordinates along these
        dimensions are dropped.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "drop"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    Returns
    -------
    Dictionaries of Variable and Index objects with merged coordinates.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.merge import merge_coords

    xarray_objs = [arg for arg in args if isinstance(arg, (DataArray, Dataset))]
    if not xarray_objs:
        return [], []

    coords = []
    for obj in xarray_objs:
        if isinstance(obj, DataArray):
            coords.append(obj.coords)
        else:  # Dataset
            coords.append(obj.coords)

    merged_coords = merge_coords(coords, combine_attrs=combine_attrs)
    merged_indexes = {dim: merged_coords[dim].to_index() for dim in merged_coords if dim not in exclude_dims}

    return [merged_coords], [merged_indexes]

def apply_dataarray_vfunc(func, *args, signature: _UFuncSignature, join: JoinOptions='inner', exclude_dims=frozenset(), keep_attrs='override') -> tuple[DataArray, ...] | DataArray:
    """Apply a variable level function over DataArray, Variable and/or ndarray
    objects.
    """
    pass
_JOINERS: dict[str, Callable] = {'inner': ordered_set_intersection, 'outer': ordered_set_union, 'left': operator.itemgetter(0), 'right': operator.itemgetter(-1), 'exact': assert_and_return_exact_match}

def _check_core_dims(signature, variable_args, name):
    """
    Check if an arg has all the core dims required by the signature.

    Slightly awkward design, of returning the error message. But we want to
    give a detailed error message, which requires inspecting the variable in
    the inner loop.
    """
    pass

def apply_dict_of_variables_vfunc(func, *args, signature: _UFuncSignature, join='inner', fill_value=None, on_missing_core_dim: MissingCoreDimOptions='raise'):
    """Apply a variable level function over dicts of DataArray, DataArray,
    Variable and ndarray objects.
    """
    pass

def _fast_dataset(variables: dict[Hashable, Variable], coord_variables: Mapping[Hashable, Variable], indexes: dict[Hashable, Index]) -> Dataset:
    """Create a dataset as quickly as possible.

    Beware: the `variables` dict is modified INPLACE.
    """
    pass

def apply_dataset_vfunc(func, *args, signature: _UFuncSignature, join='inner', dataset_join='exact', fill_value=_NO_FILL_VALUE, exclude_dims=frozenset(), keep_attrs='override', on_missing_core_dim: MissingCoreDimOptions='raise') -> Dataset | tuple[Dataset, ...]:
    """Apply a variable level function over Dataset, dict of DataArray,
    DataArray, Variable and/or ndarray objects.
    """
    pass

def _iter_over_selections(obj, dim, values):
    """Iterate over selections of an xarray object in the provided order."""
    pass

def apply_groupby_func(func, *args):
    """Apply a dataset or datarray level function over GroupBy, Dataset,
    DataArray, Variable and/or ndarray objects.
    """
    pass
SLICE_NONE = slice(None)

def apply_variable_ufunc(func, *args, signature: _UFuncSignature, exclude_dims=frozenset(), dask='forbidden', output_dtypes=None, vectorize=False, keep_attrs='override', dask_gufunc_kwargs=None) -> Variable | tuple[Variable, ...]:
    """Apply a ndarray level function over Variable and/or ndarray objects."""
    pass

def apply_array_ufunc(func, *args, dask='forbidden'):
    """Apply a ndarray level function over ndarray objects."""
    pass

def apply_ufunc(func: Callable, *args: Any, input_core_dims: Sequence[Sequence] | None=None, output_core_dims: Sequence[Sequence] | None=((),), exclude_dims: Set=frozenset(), vectorize: bool=False, join: JoinOptions='exact', dataset_join: str='exact', dataset_fill_value: object=_NO_FILL_VALUE, keep_attrs: bool | str | None=None, kwargs: Mapping | None=None, dask: Literal['forbidden', 'allowed', 'parallelized']='forbidden', output_dtypes: Sequence | None=None, output_sizes: Mapping[Any, int] | None=None, meta: Any=None, dask_gufunc_kwargs: dict[str, Any] | None=None, on_missing_core_dim: MissingCoreDimOptions='raise') -> Any:
    """Apply a vectorized function for unlabeled arrays on xarray objects.

    The function will be mapped over the data variable(s) of the input
    arguments using xarray's standard rules for labeled computation, including
    alignment, broadcasting, looping over GroupBy/Dataset variables, and
    merging of coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays
        (``.data``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs, you
        must set ``output_core_dims`` as well.
    *args : Dataset, DataArray, DataArrayGroupBy, DatasetGroupBy, Variable,         numpy.ndarray, dask.array.Array or scalar
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    input_core_dims : sequence of sequence, optional
        List of the same length as ``args`` giving the list of core dimensions
        on each input argument that should not be broadcast. By default, we
        assume there are no core dimensions on any input arguments.

        For example, ``input_core_dims=[[], ['time']]`` indicates that all
        dimensions on the first argument and all dimensions other than 'time'
        on the second argument should be broadcast.

        Core dimensions are automatically moved to the last axes of input
        variables before applying ``func``, which facilitates using NumPy style
        generalized ufuncs [2]_.
    output_core_dims : list of tuple, optional
        List of the same length as the number of output arguments from
        ``func``, giving the list of core dimensions on each output that were
        not broadcast on the inputs. By default, we assume that ``func``
        outputs exactly one array, with axes corresponding to each broadcast
        dimension.

        Core dimensions are assumed to appear as the last dimensions of each
        output in the provided order.
    exclude_dims : set, optional
        Core dimensions on the inputs to exclude from alignment and
        broadcasting entirely. Any input coordinates along these dimensions
        will be dropped. Each excluded dimension must also appear in
        ``input_core_dims`` for at least one argument. Only dimensions listed
        here are allowed to change size between input and output objects.
    vectorize : bool, optional
        If True, then assume ``func`` only takes arrays defined over core
        dimensions as input and vectorize it automatically with
        :py:func:`numpy.vectorize`. This option exists for convenience, but is
        almost always slower than supplying a pre-vectorized function.
    join : {"outer", "inner", "left", "right", "exact"}, default: "exact"
        Method for joining the indexes of the passed objects along each
        dimension, and the variables of Dataset objects with mismatched
        data variables:

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {"outer", "inner", "left", "right", "exact"}, default: "exact"
        Method for joining variables of Dataset objects with mismatched
        data variables.

        - 'outer': take variables from both Dataset objects
        - 'inner': take only overlapped variables
        - 'left': take only variables from the first object
        - 'right': take only variables from the last object
        - 'exact': data variables on all Dataset objects must match exactly
    dataset_fill_value : optional
        Value used in place of missing variables on Dataset inputs when the
        datasets do not share the exact same ``data_vars``. Required if
        ``dataset_join not in {'inner', 'exact'}``, otherwise ignored.
    keep_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", "override"} or bool, optional
        - 'drop' or False: empty attrs on returned xarray object.
        - 'identical': all attrs must be the same on every object.
        - 'no_conflicts': attrs from all objects are combined, any that have the same name must also have the same value.
        - 'drop_conflicts': attrs from all objects are combined, any that have the same name but different values are dropped.
        - 'override' or True: skip comparing and copy attrs from the first object to the result.
    kwargs : dict, optional
        Optional keyword arguments passed directly on to call ``func``.
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays:

        - 'forbidden' (default): raise an error if a dask array is encountered.
        - 'allowed': pass dask arrays directly on to ``func``. Prefer this option if
          ``func`` natively supports dask arrays.
        - 'parallelized': automatically parallelize ``func`` if any of the
          inputs are a dask array by using :py:func:`dask.array.apply_gufunc`. Multiple output
          arguments are supported. Only use this option if ``func`` does not natively
          support dask arrays (e.g. converts them to numpy arrays).
    dask_gufunc_kwargs : dict, optional
        Optional keyword arguments passed to :py:func:`dask.array.apply_gufunc` if
        dask='parallelized'. Possible keywords are ``output_sizes``, ``allow_rechunk``
        and ``meta``.
    output_dtypes : list of dtype, optional
        Optional list of output dtypes. Only used if ``dask='parallelized'`` or
        ``vectorize=True``.
    output_sizes : dict, optional
        Optional mapping from dimension names to sizes for outputs. Only used
        if dask='parallelized' and new dimensions (not found on inputs) appear
        on outputs. ``output_sizes`` should be given in the ``dask_gufunc_kwargs``
        parameter. It will be removed as direct parameter in a future version.
    meta : optional
        Size-0 object representing the type of array wrapped by dask array. Passed on to
        :py:func:`dask.array.apply_gufunc`. ``meta`` should be given in the
        ``dask_gufunc_kwargs`` parameter . It will be removed as direct parameter
        a future version.
    on_missing_core_dim : {"raise", "copy", "drop"}, default: "raise"
        How to handle missing core dimensions on input variables.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    Notes
    -----
    This function is designed for the more common case where ``func`` can work on numpy
    arrays. If ``func`` needs to manipulate a whole xarray object subset to each block
    it is possible to use :py:func:`xarray.map_blocks`.

    Note that due to the overhead :py:func:`xarray.map_blocks` is considerably slower than ``apply_ufunc``.

    Examples
    --------
    Calculate the vector magnitude of two arguments:

    >>> def magnitude(a, b):
    ...     func = lambda x, y: np.sqrt(x**2 + y**2)
    ...     return xr.apply_ufunc(func, a, b)
    ...

    You can now apply ``magnitude()`` to :py:class:`DataArray` and :py:class:`Dataset`
    objects, with automatically preserved dimensions and coordinates, e.g.,

    >>> array = xr.DataArray([1, 2, 3], coords=[("x", [0.1, 0.2, 0.3])])
    >>> magnitude(array, -array)
    <xarray.DataArray (x: 3)> Size: 24B
    array([1.41421356, 2.82842712, 4.24264069])
    Coordinates:
      * x        (x) float64 24B 0.1 0.2 0.3

    Plain scalars, numpy arrays and a mix of these with xarray objects is also
    supported:

    >>> magnitude(3, 4)
    np.float64(5.0)
    >>> magnitude(3, np.array([0, 4]))
    array([3., 5.])
    >>> magnitude(array, 0)
    <xarray.DataArray (x: 3)> Size: 24B
    array([1., 2., 3.])
    Coordinates:
      * x        (x) float64 24B 0.1 0.2 0.3

    Other examples of how you could use ``apply_ufunc`` to write functions to
    (very nearly) replicate existing xarray functionality:

    Compute the mean (``.mean``) over one dimension:

    >>> def mean(obj, dim):
    ...     # note: apply always moves core dimensions to the end
    ...     return apply_ufunc(
    ...         np.mean, obj, input_core_dims=[[dim]], kwargs={"axis": -1}
    ...     )
    ...

    Inner product over a specific dimension (like :py:func:`dot`):

    >>> def _inner(x, y):
    ...     result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
    ...     return result[..., 0, 0]
    ...
    >>> def inner_product(a, b, dim):
    ...     return apply_ufunc(_inner, a, b, input_core_dims=[[dim], [dim]])
    ...

    Stack objects along a new dimension (like :py:func:`concat`):

    >>> def stack(objects, dim, new_coord):
    ...     # note: this version does not stack coordinates
    ...     func = lambda *x: np.stack(x, axis=-1)
    ...     result = apply_ufunc(
    ...         func,
    ...         *objects,
    ...         output_core_dims=[[dim]],
    ...         join="outer",
    ...         dataset_fill_value=np.nan
    ...     )
    ...     result[dim] = new_coord
    ...     return result
    ...

    If your function is not vectorized but can be applied only to core
    dimensions, you can use ``vectorize=True`` to turn into a vectorized
    function. This wraps :py:func:`numpy.vectorize`, so the operation isn't
    terribly fast. Here we'll use it to calculate the distance between
    empirical samples from two probability distributions, using a scipy
    function that needs to be applied to vectors:

    >>> import scipy.stats
    >>> def earth_mover_distance(first_samples, second_samples, dim="ensemble"):
    ...     return apply_ufunc(
    ...         scipy.stats.wasserstein_distance,
    ...         first_samples,
    ...         second_samples,
    ...         input_core_dims=[[dim], [dim]],
    ...         vectorize=True,
    ...     )
    ...

    Most of NumPy's builtin functions already broadcast their inputs
    appropriately for use in ``apply_ufunc``. You may find helper functions such as
    :py:func:`numpy.broadcast_arrays` helpful in writing your function. ``apply_ufunc`` also
    works well with :py:func:`numba.vectorize` and :py:func:`numba.guvectorize`.

    See Also
    --------
    numpy.broadcast_arrays
    numba.vectorize
    numba.guvectorize
    dask.array.apply_gufunc
    xarray.map_blocks

    Notes
    -----
    :ref:`dask.automatic-parallelization`
        User guide describing :py:func:`apply_ufunc` and :py:func:`map_blocks`.

    :doc:`xarray-tutorial:advanced/apply_ufunc/apply_ufunc`
        Advanced Tutorial on applying numpy function using :py:func:`apply_ufunc`

    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html
    .. [2] https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html
    """
    pass

def cov(da_a: T_DataArray, da_b: T_DataArray, dim: Dims=None, ddof: int=1, weights: T_DataArray | None=None) -> T_DataArray:
    """
    Compute covariance between two DataArray objects along a shared dimension.

    Parameters
    ----------
    da_a : DataArray
        Array to compute.
    da_b : DataArray
        Array to compute.
    dim : str, iterable of hashable, "..." or None, optional
        The dimension along which the covariance will be computed
    ddof : int, default: 1
        If ddof=1, covariance is normalized by N-1, giving an unbiased estimate,
        else normalization is by N.
    weights : DataArray, optional
        Array of weights.

    Returns
    -------
    covariance : DataArray

    See Also
    --------
    pandas.Series.cov : corresponding pandas function
    xarray.corr : respective function to calculate correlation

    Examples
    --------
    >>> from xarray import DataArray
    >>> da_a = DataArray(
    ...     np.array([[1, 2, 3], [0.1, 0.2, 0.3], [3.2, 0.6, 1.8]]),
    ...     dims=("space", "time"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...         ("time", pd.date_range("2000-01-01", freq="1D", periods=3)),
    ...     ],
    ... )
    >>> da_a
    <xarray.DataArray (space: 3, time: 3)> Size: 72B
    array([[1. , 2. , 3. ],
           [0.1, 0.2, 0.3],
           [3.2, 0.6, 1.8]])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    >>> da_b = DataArray(
    ...     np.array([[0.2, 0.4, 0.6], [15, 10, 5], [3.2, 0.6, 1.8]]),
    ...     dims=("space", "time"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...         ("time", pd.date_range("2000-01-01", freq="1D", periods=3)),
    ...     ],
    ... )
    >>> da_b
    <xarray.DataArray (space: 3, time: 3)> Size: 72B
    array([[ 0.2,  0.4,  0.6],
           [15. , 10. ,  5. ],
           [ 3.2,  0.6,  1.8]])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    >>> xr.cov(da_a, da_b)
    <xarray.DataArray ()> Size: 8B
    array(-3.53055556)
    >>> xr.cov(da_a, da_b, dim="time")
    <xarray.DataArray (space: 3)> Size: 24B
    array([ 0.2       , -0.5       ,  1.69333333])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
    >>> weights = DataArray(
    ...     [4, 2, 1],
    ...     dims=("space"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...     ],
    ... )
    >>> weights
    <xarray.DataArray (space: 3)> Size: 24B
    array([4, 2, 1])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
    >>> xr.cov(da_a, da_b, dim="space", weights=weights)
    <xarray.DataArray (time: 3)> Size: 24B
    array([-4.69346939, -4.49632653, -3.37959184])
    Coordinates:
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    """
    pass

def corr(da_a: T_DataArray, da_b: T_DataArray, dim: Dims=None, weights: T_DataArray | None=None) -> T_DataArray:
    """
    Compute the Pearson correlation coefficient between
    two DataArray objects along a shared dimension.

    Parameters
    ----------
    da_a : DataArray
        Array to compute.
    da_b : DataArray
        Array to compute.
    dim : str, iterable of hashable, "..." or None, optional
        The dimension along which the correlation will be computed
    weights : DataArray, optional
        Array of weights.

    Returns
    -------
    correlation: DataArray

    See Also
    --------
    pandas.Series.corr : corresponding pandas function
    xarray.cov : underlying covariance function

    Examples
    --------
    >>> from xarray import DataArray
    >>> da_a = DataArray(
    ...     np.array([[1, 2, 3], [0.1, 0.2, 0.3], [3.2, 0.6, 1.8]]),
    ...     dims=("space", "time"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...         ("time", pd.date_range("2000-01-01", freq="1D", periods=3)),
    ...     ],
    ... )
    >>> da_a
    <xarray.DataArray (space: 3, time: 3)> Size: 72B
    array([[1. , 2. , 3. ],
           [0.1, 0.2, 0.3],
           [3.2, 0.6, 1.8]])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    >>> da_b = DataArray(
    ...     np.array([[0.2, 0.4, 0.6], [15, 10, 5], [3.2, 0.6, 1.8]]),
    ...     dims=("space", "time"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...         ("time", pd.date_range("2000-01-01", freq="1D", periods=3)),
    ...     ],
    ... )
    >>> da_b
    <xarray.DataArray (space: 3, time: 3)> Size: 72B
    array([[ 0.2,  0.4,  0.6],
           [15. , 10. ,  5. ],
           [ 3.2,  0.6,  1.8]])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    >>> xr.corr(da_a, da_b)
    <xarray.DataArray ()> Size: 8B
    array(-0.57087777)
    >>> xr.corr(da_a, da_b, dim="time")
    <xarray.DataArray (space: 3)> Size: 24B
    array([ 1., -1.,  1.])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
    >>> weights = DataArray(
    ...     [4, 2, 1],
    ...     dims=("space"),
    ...     coords=[
    ...         ("space", ["IA", "IL", "IN"]),
    ...     ],
    ... )
    >>> weights
    <xarray.DataArray (space: 3)> Size: 24B
    array([4, 2, 1])
    Coordinates:
      * space    (space) <U2 24B 'IA' 'IL' 'IN'
    >>> xr.corr(da_a, da_b, dim="space", weights=weights)
    <xarray.DataArray (time: 3)> Size: 24B
    array([-0.50240504, -0.83215028, -0.99057446])
    Coordinates:
      * time     (time) datetime64[ns] 24B 2000-01-01 2000-01-02 2000-01-03
    """
    pass

def _cov_corr(da_a: T_DataArray, da_b: T_DataArray, weights: T_DataArray | None=None, dim: Dims=None, ddof: int=0, method: Literal['cov', 'corr', None]=None) -> T_DataArray:
    """
    Internal method for xr.cov() and xr.corr() so only have to
    sanitize the input arrays once and we don't repeat code.
    """
    pass

def cross(a: DataArray | Variable, b: DataArray | Variable, *, dim: Hashable) -> DataArray | Variable:
    """
    Compute the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector
    perpendicular to both `a` and `b`. The vectors in `a` and `b` are
    defined by the values along the dimension `dim` and can have sizes
    1, 2 or 3. Where the size of either `a` or `b` is
    1 or 2, the remaining components of the input vector is assumed to
    be zero and the cross product calculated accordingly. In cases where
    both input vectors have dimension 2, the z-component of the cross
    product is returned.

    Parameters
    ----------
    a, b : DataArray or Variable
        Components of the first and second vector(s).
    dim : hashable
        The dimension along which the cross product will be computed.
        Must be available in both vectors.

    Examples
    --------
    Vector cross-product with 3 dimensions:

    >>> a = xr.DataArray([1, 2, 3])
    >>> b = xr.DataArray([4, 5, 6])
    >>> xr.cross(a, b, dim="dim_0")
    <xarray.DataArray (dim_0: 3)> Size: 24B
    array([-3,  6, -3])
    Dimensions without coordinates: dim_0

    Vector cross-product with 3 dimensions but zeros at the last axis
    yields the same results as with 2 dimensions:

    >>> a = xr.DataArray([1, 2, 0])
    >>> b = xr.DataArray([4, 5, 0])
    >>> xr.cross(a, b, dim="dim_0")
    <xarray.DataArray (dim_0: 3)> Size: 24B
    array([ 0,  0, -3])
    Dimensions without coordinates: dim_0

    Multiple vector cross-products. Note that the direction of the
    cross product vector is defined by the right-hand rule:

    >>> a = xr.DataArray(
    ...     [[1, 2, 3], [4, 5, 6]],
    ...     dims=("time", "cartesian"),
    ...     coords=dict(
    ...         time=(["time"], [0, 1]),
    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
    ...     ),
    ... )
    >>> b = xr.DataArray(
    ...     [[4, 5, 6], [1, 2, 3]],
    ...     dims=("time", "cartesian"),
    ...     coords=dict(
    ...         time=(["time"], [0, 1]),
    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
    ...     ),
    ... )
    >>> xr.cross(a, b, dim="cartesian")
    <xarray.DataArray (time: 2, cartesian: 3)> Size: 48B
    array([[-3,  6, -3],
           [ 3, -6,  3]])
    Coordinates:
      * time       (time) int64 16B 0 1
      * cartesian  (cartesian) <U1 12B 'x' 'y' 'z'

    Cross can be called on Datasets by converting to DataArrays and later
    back to a Dataset:

    >>> ds_a = xr.Dataset(dict(x=("dim_0", [1]), y=("dim_0", [2]), z=("dim_0", [3])))
    >>> ds_b = xr.Dataset(dict(x=("dim_0", [4]), y=("dim_0", [5]), z=("dim_0", [6])))
    >>> c = xr.cross(
    ...     ds_a.to_dataarray("cartesian"),
    ...     ds_b.to_dataarray("cartesian"),
    ...     dim="cartesian",
    ... )
    >>> c.to_dataset(dim="cartesian")
    <xarray.Dataset> Size: 24B
    Dimensions:  (dim_0: 1)
    Dimensions without coordinates: dim_0
    Data variables:
        x        (dim_0) int64 8B -3
        y        (dim_0) int64 8B 6
        z        (dim_0) int64 8B -3

    See Also
    --------
    numpy.cross : Corresponding numpy function
    """
    pass

@deprecate_dims
def dot(*arrays, dim: Dims=None, **kwargs: Any):
    """Generalized dot product for xarray objects. Like ``np.einsum``, but
    provides a simpler interface based on array dimension names.

    Parameters
    ----------
    *arrays : DataArray or Variable
        Arrays to compute.
    dim : str, iterable of hashable, "..." or None, optional
        Which dimensions to sum over. Ellipsis ('...') sums over all dimensions.
        If not specified, then all the common dimensions are summed over.
    **kwargs : dict
        Additional keyword arguments passed to ``numpy.einsum`` or
        ``dask.array.einsum``

    Returns
    -------
    DataArray

    See Also
    --------
    numpy.einsum
    dask.array.einsum
    opt_einsum.contract

    Notes
    -----
    We recommend installing the optional ``opt_einsum`` package, or alternatively passing ``optimize=True``,
    which is passed through to ``np.einsum``, and works for most array backends.

    Examples
    --------
    >>> da_a = xr.DataArray(np.arange(3 * 2).reshape(3, 2), dims=["a", "b"])
    >>> da_b = xr.DataArray(np.arange(3 * 2 * 2).reshape(3, 2, 2), dims=["a", "b", "c"])
    >>> da_c = xr.DataArray(np.arange(2 * 3).reshape(2, 3), dims=["c", "d"])

    >>> da_a
    <xarray.DataArray (a: 3, b: 2)> Size: 48B
    array([[0, 1],
           [2, 3],
           [4, 5]])
    Dimensions without coordinates: a, b

    >>> da_b
    <xarray.DataArray (a: 3, b: 2, c: 2)> Size: 96B
    array([[[ 0,  1],
            [ 2,  3]],
    <BLANKLINE>
           [[ 4,  5],
            [ 6,  7]],
    <BLANKLINE>
           [[ 8,  9],
            [10, 11]]])
    Dimensions without coordinates: a, b, c

    >>> da_c
    <xarray.DataArray (c: 2, d: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Dimensions without coordinates: c, d

    >>> xr.dot(da_a, da_b, dim=["a", "b"])
    <xarray.DataArray (c: 2)> Size: 16B
    array([110, 125])
    Dimensions without coordinates: c

    >>> xr.dot(da_a, da_b, dim=["a"])
    <xarray.DataArray (b: 2, c: 2)> Size: 32B
    array([[40, 46],
           [70, 79]])
    Dimensions without coordinates: b, c

    >>> xr.dot(da_a, da_b, da_c, dim=["b", "c"])
    <xarray.DataArray (a: 3, d: 3)> Size: 72B
    array([[  9,  14,  19],
           [ 93, 150, 207],
           [273, 446, 619]])
    Dimensions without coordinates: a, d

    >>> xr.dot(da_a, da_b)
    <xarray.DataArray (c: 2)> Size: 16B
    array([110, 125])
    Dimensions without coordinates: c

    >>> xr.dot(da_a, da_b, dim=...)
    <xarray.DataArray ()> Size: 8B
    array(235)
    """
    pass

def where(cond, x, y, keep_attrs=None):
    """Return elements from `x` or `y` depending on `cond`.

    Performs xarray-like broadcasting across input arguments.

    All dimension coordinates on `x` and `y`  must be aligned with each
    other and with `cond`.

    Parameters
    ----------
    cond : scalar, array, Variable, DataArray or Dataset
        When True, return values from `x`, otherwise returns values from `y`.
    x : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is True
    y : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is False
    keep_attrs : bool or str or callable, optional
        How to treat attrs. If True, keep the attrs of `x`.

    Returns
    -------
    Dataset, DataArray, Variable or array
        In priority order: Dataset, DataArray, Variable or array, whichever
        type appears as an input argument.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     0.1 * np.arange(10),
    ...     dims=["lat"],
    ...     coords={"lat": np.arange(10)},
    ...     name="sst",
    ... )
    >>> x
    <xarray.DataArray 'sst' (lat: 10)> Size: 80B
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Coordinates:
      * lat      (lat) int64 80B 0 1 2 3 4 5 6 7 8 9

    >>> xr.where(x < 0.5, x, x * 100)
    <xarray.DataArray 'sst' (lat: 10)> Size: 80B
    array([ 0. ,  0.1,  0.2,  0.3,  0.4, 50. , 60. , 70. , 80. , 90. ])
    Coordinates:
      * lat      (lat) int64 80B 0 1 2 3 4 5 6 7 8 9

    >>> y = xr.DataArray(
    ...     0.1 * np.arange(9).reshape(3, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": np.arange(3), "lon": 10 + np.arange(3)},
    ...     name="sst",
    ... )
    >>> y
    <xarray.DataArray 'sst' (lat: 3, lon: 3)> Size: 72B
    array([[0. , 0.1, 0.2],
           [0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8]])
    Coordinates:
      * lat      (lat) int64 24B 0 1 2
      * lon      (lon) int64 24B 10 11 12

    >>> xr.where(y.lat < 1, y, -1)
    <xarray.DataArray (lat: 3, lon: 3)> Size: 72B
    array([[ 0. ,  0.1,  0.2],
           [-1. , -1. , -1. ],
           [-1. , -1. , -1. ]])
    Coordinates:
      * lat      (lat) int64 24B 0 1 2
      * lon      (lon) int64 24B 10 11 12

    >>> cond = xr.DataArray([True, False], dims=["x"])
    >>> x = xr.DataArray([1, 2], dims=["y"])
    >>> xr.where(cond, x, 0)
    <xarray.DataArray (x: 2, y: 2)> Size: 32B
    array([[1, 2],
           [0, 0]])
    Dimensions without coordinates: x, y

    See Also
    --------
    numpy.where : corresponding numpy function
    Dataset.where, DataArray.where :
        equivalent methods
    """
    pass

def polyval(coord: Dataset | DataArray, coeffs: Dataset | DataArray, degree_dim: Hashable='degree') -> Dataset | DataArray:
    """Evaluate a polynomial at specific values

    Parameters
    ----------
    coord : DataArray or Dataset
        Values at which to evaluate the polynomial.
    coeffs : DataArray or Dataset
        Coefficients of the polynomial.
    degree_dim : Hashable, default: "degree"
        Name of the polynomial degree dimension in `coeffs`.

    Returns
    -------
    DataArray or Dataset
        Evaluated polynomial.

    See Also
    --------
    xarray.DataArray.polyfit
    numpy.polynomial.polynomial.polyval
    """
    pass

def _ensure_numeric(data: Dataset | DataArray) -> Dataset | DataArray:
    """Converts all datetime64 variables to float64

    Parameters
    ----------
    data : DataArray or Dataset
        Variables with possible datetime dtypes.

    Returns
    -------
    DataArray or Dataset
        Variables with datetime64 dtypes converted to float64.
    """
    pass

def _calc_idxminmax(*, array, func: Callable, dim: Hashable | None=None, skipna: bool | None=None, fill_value: Any=dtypes.NA, keep_attrs: bool | None=None):
    """Apply common operations for idxmin and idxmax."""
    pass
_T = TypeVar('_T', bound=Union['Dataset', 'DataArray'])
_U = TypeVar('_U', bound=Union['Dataset', 'DataArray'])
_V = TypeVar('_V', bound=Union['Dataset', 'DataArray'])

def unify_chunks(*objects: Dataset | DataArray) -> tuple[Dataset | DataArray, ...]:
    """
    Given any number of Dataset and/or DataArray objects, returns
    new objects with unified chunk size along all chunked dimensions.

    Returns
    -------
    unified (DataArray or Dataset) – Tuple of objects with the same type as
    *objects with consistent chunk sizes for all dask-array variables

    See Also
    --------
    dask.array.core.unify_chunks
    """
    pass
