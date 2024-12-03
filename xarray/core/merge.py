from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.alignment import deep_align
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, create_default_index_implicit, filter_indexes_from_coords, indexes_equal
from xarray.core.utils import Frozen, compat_dict_union, dict_equiv, equivalent
from xarray.core.variable import Variable, as_variable, calculate_dimensions
if TYPE_CHECKING:
    from xarray.core.coordinates import Coordinates
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import CombineAttrsOptions, CompatOptions, JoinOptions
    DimsLike = Union[Hashable, Sequence[Hashable]]
    ArrayLike = Any
    VariableLike = Union[ArrayLike, tuple[DimsLike, ArrayLike], tuple[DimsLike, ArrayLike, Mapping], tuple[DimsLike, ArrayLike, Mapping, Mapping]]
    XarrayValue = Union[DataArray, Variable, VariableLike]
    DatasetLike = Union[Dataset, Coordinates, Mapping[Any, XarrayValue]]
    CoercibleValue = Union[XarrayValue, pd.Series, pd.DataFrame]
    CoercibleMapping = Union[Dataset, Mapping[Any, CoercibleValue]]
PANDAS_TYPES = (pd.Series, pd.DataFrame)
_VALID_COMPAT = Frozen({'identical': 0, 'equals': 1, 'broadcast_equals': 2, 'minimal': 3, 'no_conflicts': 4, 'override': 5})

class Context:
    """object carrying the information of a call"""

    def __init__(self, func):
        self.func = func

def broadcast_dimension_size(variables: list[Variable]) -> dict[Hashable, int]:
    """Extract dimension sizes from a dictionary of variables.

    Raises ValueError if any dimensions have different sizes.
    """
    dims = {}
    for var in variables:
        for dim, size in var.sizes.items():
            if dim in dims:
                if dims[dim] != size:
                    raise ValueError(f"Inconsistent dimension size for {dim}: "
                                     f"found both {dims[dim]} and {size}")
            else:
                dims[dim] = size
    return dims

class MergeError(ValueError):
    """Error class for merge failures due to incompatible arguments."""

def unique_variable(name: Hashable, variables: list[Variable], compat: CompatOptions='broadcast_equals', equals: bool | None=None) -> Variable:
    """Return the unique variable from a list of variables or raise MergeError.

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        Type of equality check to use.
    equals : None or bool, optional
        corresponding to result of compat test

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
    """
    if len(variables) == 1:
        return variables[0]

    if compat == 'override':
        return variables[0]

    first_var = variables[0]
    for var in variables[1:]:
        if compat == 'broadcast_equals':
            if not first_var.broadcast_equals(var):
                raise MergeError(f"Conflicting values for variable {name}")
        elif compat == 'equals':
            if not first_var.equals(var):
                raise MergeError(f"Conflicting values for variable {name}")
        elif compat == 'identical':
            if not first_var.identical(var):
                raise MergeError(f"Conflicting values or attributes for variable {name}")
        elif compat == 'no_conflicts':
            if not first_var.no_conflicts(var):
                raise MergeError(f"Conflicting non-null values for variable {name}")
        else:
            raise ValueError(f"Unsupported compat option: {compat}")

    return first_var
MergeElement = tuple[Variable, Optional[Index]]

def _assert_prioritized_valid(grouped: dict[Hashable, list[MergeElement]], prioritized: Mapping[Any, MergeElement]) -> None:
    """Make sure that elements given in prioritized will not corrupt any
    index given in grouped.
    """
    for name, (variable, index) in prioritized.items():
        if name in grouped:
            grouped_variable, grouped_index = grouped[name][0]
            if index is not None and grouped_index is not None:
                if not indexes_equal(index, grouped_index):
                    raise MergeError(f"Incompatible indexes for variable {name}")
            elif index is not None or grouped_index is not None:
                raise MergeError(f"Inconsistent presence of index for variable {name}")

def merge_collected(grouped: dict[Any, list[MergeElement]], prioritized: Mapping[Any, MergeElement] | None=None, compat: CompatOptions='minimal', combine_attrs: CombineAttrsOptions='override', equals: dict[Any, bool] | None=None) -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge dicts of variables, while resolving conflicts appropriately.

    Parameters
    ----------
    grouped : mapping
    prioritized : mapping
    compat : str
        Type of equality check to use when checking for conflicts.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                     "override"} or callable, default: "override"
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
    equals : mapping, optional
        corresponding to result of compat test

    Returns
    -------
    Dict with keys taken by the union of keys on list_of_mappings,
    and Variable values corresponding to those that should be found on the
    merged result.
    """
    if prioritized is None:
        prioritized = {}
    
    _assert_prioritized_valid(grouped, prioritized)
    
    result_vars = {}
    result_indexes = {}
    
    for name, elements in grouped.items():
        if name in prioritized:
            result_vars[name], result_indexes[name] = prioritized[name]
        else:
            variables = [elem[0] for elem in elements]
            variable = unique_variable(name, variables, compat)
            
            indexes = [elem[1] for elem in elements if elem[1] is not None]
            index = indexes[0] if indexes else None
            
            result_vars[name] = variable
            if index is not None:
                result_indexes[name] = index
    
    # Combine attributes
    if callable(combine_attrs):
        attrs = combine_attrs([var.attrs for var in result_vars.values()], Context(merge_collected))
    elif combine_attrs == "drop":
        attrs = {}
    elif combine_attrs == "identical":
        attrs = result_vars[next(iter(result_vars))].attrs
        for var in result_vars.values():
            if not dict_equiv(var.attrs, attrs):
                raise MergeError("Conflicting attributes")
    elif combine_attrs == "no_conflicts":
        attrs = {}
        for var in result_vars.values():
            for k, v in var.attrs.items():
                if k in attrs and attrs[k] != v:
                    raise MergeError(f"Conflicting attribute {k}")
                attrs[k] = v
    elif combine_attrs == "drop_conflicts":
        attrs = {}
        for var in result_vars.values():
            for k, v in var.attrs.items():
                if k not in attrs:
                    attrs[k] = v
    elif combine_attrs == "override":
        attrs = result_vars[next(iter(result_vars))].attrs
    else:
        raise ValueError(f"Unsupported combine_attrs option: {combine_attrs}")
    
    for var in result_vars.values():
        var.attrs = attrs
    
    return result_vars, result_indexes

def collect_variables_and_indexes(list_of_mappings: Iterable[DatasetLike], indexes: Mapping[Any, Any] | None=None) -> dict[Hashable, list[MergeElement]]:
    """Collect variables and indexes from list of mappings of xarray objects.

    Mappings can be Dataset or Coordinates objects, in which case both
    variables and indexes are extracted from it.

    It can also have values of one of the following types:
    - an xarray.Variable
    - a tuple `(dims, data[, attrs[, encoding]])` that can be converted in
      an xarray.Variable
    - or an xarray.DataArray

    If a mapping of indexes is given, those indexes are assigned to all variables
    with a matching key/name. For dimension variables with no matching index, a
    default (pandas) index is assigned. DataArray indexes that don't match mapping
    keys are also extracted.

    """
    pass

def collect_from_coordinates(list_of_coords: list[Coordinates]) -> dict[Hashable, list[MergeElement]]:
    """Collect variables and indexes to be merged from Coordinate objects."""
    pass

def merge_coordinates_without_align(objects: list[Coordinates], prioritized: Mapping[Any, MergeElement] | None=None, exclude_dims: Set=frozenset(), combine_attrs: CombineAttrsOptions='override') -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge variables/indexes from coordinates without automatic alignments.

    This function is used for merging coordinate from pre-existing xarray
    objects.
    """
    pass

def determine_coords(list_of_mappings: Iterable[DatasetLike]) -> tuple[set[Hashable], set[Hashable]]:
    """Given a list of dicts with xarray object values, identify coordinates.

    Parameters
    ----------
    list_of_mappings : list of dict or list of Dataset
        Of the same form as the arguments to expand_variable_dicts.

    Returns
    -------
    coord_names : set of variable names
    noncoord_names : set of variable names
        All variable found in the input should appear in either the set of
        coordinate or non-coordinate names.
    """
    pass

def coerce_pandas_values(objects: Iterable[CoercibleMapping]) -> list[DatasetLike]:
    """Convert pandas values found in a list of labeled objects.

    Parameters
    ----------
    objects : list of Dataset or mapping
        The mappings may contain any sort of objects coercible to
        xarray.Variables as keys, including pandas objects.

    Returns
    -------
    List of Dataset or dictionary objects. Any inputs or values in the inputs
    that were pandas objects have been converted into native xarray objects.
    """
    pass

def _get_priority_vars_and_indexes(objects: Sequence[DatasetLike], priority_arg: int | None, compat: CompatOptions='equals') -> dict[Hashable, MergeElement]:
    """Extract the priority variable from a list of mappings.

    We need this method because in some cases the priority argument itself
    might have conflicting values (e.g., if it is a dict with two DataArray
    values with conflicting coordinate values).

    Parameters
    ----------
    objects : sequence of dict-like of Variable
        Dictionaries in which to find the priority variables.
    priority_arg : int or None
        Integer object whose variable should take priority.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare non-concatenated variables of the same name for
        potential conflicts. This is passed down to merge.

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset

    Returns
    -------
    A dictionary of variables and associated indexes (if any) to prioritize.
    """
    pass

def merge_coords(objects: Iterable[CoercibleMapping], compat: CompatOptions='minimal', join: JoinOptions='outer', priority_arg: int | None=None, indexes: Mapping[Any, Index] | None=None, fill_value: object=dtypes.NA) -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    pass

def merge_attrs(variable_attrs, combine_attrs, context=None):
    """Combine attributes from different variables according to combine_attrs"""
    pass

class _MergeResult(NamedTuple):
    variables: dict[Hashable, Variable]
    coord_names: set[Hashable]
    dims: dict[Hashable, int]
    indexes: dict[Hashable, Index]
    attrs: dict[Hashable, Any]

def merge_core(objects: Iterable[CoercibleMapping], compat: CompatOptions='broadcast_equals', join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='override', priority_arg: int | None=None, explicit_coords: Iterable[Hashable] | None=None, indexes: Mapping[Any, Any] | None=None, fill_value: object=dtypes.NA, skip_align_args: list[int] | None=None) -> _MergeResult:
    """Core logic for merging labeled objects.

    This is not public API.

    Parameters
    ----------
    objects : list of mapping
        All values must be convertible to labeled arrays.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        Compatibility checks to use when merging variables.
    join : {"outer", "inner", "left", "right"}, optional
        How to combine objects with different indexes.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "override"
        How to combine attributes of objects
    priority_arg : int, optional
        Optional argument in `objects` that takes precedence over the others.
    explicit_coords : set, optional
        An explicit list of variables from `objects` that are coordinates.
    indexes : dict, optional
        Dictionary with values given by xarray.Index objects or anything that
        may be cast to pandas.Index objects.
    fill_value : scalar, optional
        Value to use for newly missing values
    skip_align_args : list of int, optional
        Optional arguments in `objects` that are not included in alignment.

    Returns
    -------
    variables : dict
        Dictionary of Variable objects.
    coord_names : set
        Set of coordinate names.
    dims : dict
        Dictionary mapping from dimension names to sizes.
    attrs : dict
        Dictionary of attributes

    Raises
    ------
    MergeError if the merge cannot be done successfully.
    """
    pass

def merge(objects: Iterable[DataArray | CoercibleMapping], compat: CompatOptions='no_conflicts', join: JoinOptions='outer', fill_value: object=dtypes.NA, combine_attrs: CombineAttrsOptions='override') -> Dataset:
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : iterable of Dataset or iterable of DataArray or iterable of dict-like
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts",               "override", "minimal"}, default: "no_conflicts"
        String indicating how to compare variables of the same name for
        potential conflicts:

        - "identical": all values, dimensions and attributes must be the
          same.
        - "equals": all values and dimensions must be the same.
        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
        - "minimal": drop conflicting coordinates

    join : {"outer", "inner", "left", "right", "exact", "override"}, default: "outer"
        String indicating how to combine differing indexes in objects.

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "override"
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
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     [[1.0, 2.0], [3.0, 5.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ...     name="var1",
    ... )
    >>> y = xr.DataArray(
    ...     [[5.0, 6.0], [7.0, 8.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 150.0]},
    ...     name="var2",
    ... )
    >>> z = xr.DataArray(
    ...     [[0.0, 3.0], [4.0, 9.0]],
    ...     dims=("time", "lon"),
    ...     coords={"time": [30.0, 60.0], "lon": [100.0, 150.0]},
    ...     name="var3",
    ... )

    >>> x
    <xarray.DataArray 'var1' (lat: 2, lon: 2)> Size: 32B
    array([[1., 2.],
           [3., 5.]])
    Coordinates:
      * lat      (lat) float64 16B 35.0 40.0
      * lon      (lon) float64 16B 100.0 120.0

    >>> y
    <xarray.DataArray 'var2' (lat: 2, lon: 2)> Size: 32B
    array([[5., 6.],
           [7., 8.]])
    Coordinates:
      * lat      (lat) float64 16B 35.0 42.0
      * lon      (lon) float64 16B 100.0 150.0

    >>> z
    <xarray.DataArray 'var3' (time: 2, lon: 2)> Size: 32B
    array([[0., 3.],
           [4., 9.]])
    Coordinates:
      * time     (time) float64 16B 30.0 60.0
      * lon      (lon) float64 16B 100.0 150.0

    >>> xr.merge([x, y, z])
    <xarray.Dataset> Size: 256B
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 24B 35.0 40.0 42.0
      * lon      (lon) float64 24B 100.0 120.0 150.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 72B 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 72B 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 48B 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="identical")
    <xarray.Dataset> Size: 256B
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 24B 35.0 40.0 42.0
      * lon      (lon) float64 24B 100.0 120.0 150.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 72B 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 72B 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 48B 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals")
    <xarray.Dataset> Size: 256B
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 24B 35.0 40.0 42.0
      * lon      (lon) float64 24B 100.0 120.0 150.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 72B 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 72B 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 48B 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat="equals", fill_value=-999.0)
    <xarray.Dataset> Size: 256B
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 24B 35.0 40.0 42.0
      * lon      (lon) float64 24B 100.0 120.0 150.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 72B 1.0 2.0 -999.0 3.0 ... -999.0 -999.0 -999.0
        var2     (lat, lon) float64 72B 5.0 -999.0 6.0 -999.0 ... 7.0 -999.0 8.0
        var3     (time, lon) float64 48B 0.0 -999.0 3.0 4.0 -999.0 9.0

    >>> xr.merge([x, y, z], join="override")
    <xarray.Dataset> Size: 144B
    Dimensions:  (lat: 2, lon: 2, time: 2)
    Coordinates:
      * lat      (lat) float64 16B 35.0 40.0
      * lon      (lon) float64 16B 100.0 120.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 32B 1.0 2.0 3.0 5.0
        var2     (lat, lon) float64 32B 5.0 6.0 7.0 8.0
        var3     (time, lon) float64 32B 0.0 3.0 4.0 9.0

    >>> xr.merge([x, y, z], join="inner")
    <xarray.Dataset> Size: 64B
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
      * lat      (lat) float64 8B 35.0
      * lon      (lon) float64 8B 100.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 8B 1.0
        var2     (lat, lon) float64 8B 5.0
        var3     (time, lon) float64 16B 0.0 4.0

    >>> xr.merge([x, y, z], compat="identical", join="inner")
    <xarray.Dataset> Size: 64B
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
      * lat      (lat) float64 8B 35.0
      * lon      (lon) float64 8B 100.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 8B 1.0
        var2     (lat, lon) float64 8B 5.0
        var3     (time, lon) float64 16B 0.0 4.0

    >>> xr.merge([x, y, z], compat="broadcast_equals", join="outer")
    <xarray.Dataset> Size: 256B
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
      * lat      (lat) float64 24B 35.0 40.0 42.0
      * lon      (lon) float64 24B 100.0 120.0 150.0
      * time     (time) float64 16B 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 72B 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 72B 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 48B 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], join="exact")
    Traceback (most recent call last):
    ...
    ValueError: cannot align objects with join='exact' where ...

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    combine_nested
    combine_by_coords
    """
    pass

def dataset_merge_method(dataset: Dataset, other: CoercibleMapping, overwrite_vars: Hashable | Iterable[Hashable], compat: CompatOptions, join: JoinOptions, fill_value: Any, combine_attrs: CombineAttrsOptions) -> _MergeResult:
    """Guts of the Dataset.merge method."""
    pass

def dataset_update_method(dataset: Dataset, other: CoercibleMapping) -> _MergeResult:
    """Guts of the Dataset.update method.

    This drops a duplicated coordinates from `other` if `other` is not an
    `xarray.Dataset`, e.g., if it's a dict with DataArray values (GH2068,
    GH2180).
    """
    pass
