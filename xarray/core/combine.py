from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
if TYPE_CHECKING:
    from xarray.core.types import CombineAttrsOptions, CompatOptions, JoinOptions

def _infer_tile_ids_from_nested_list(entry, current_pos=()):
    """
    Given a list of lists (of lists...) of objects, returns a iterator
    which returns a tuple containing the index of each object in the nested
    list structure as the key, and the object. This can then be called by the
    dict constructor to create a dictionary of the objects organised by their
    position in the original nested list.

    Recursively traverses the given structure, while keeping track of the
    current position. Should work for any type of object which isn't a list.

    Parameters
    ----------
    entry : list[list[obj, obj, ...], ...]
        List of lists of arbitrary depth, containing objects in the order
        they are to be concatenated.
    current_pos : tuple
        Current position in the nested structure.

    Returns
    -------
    combined_tile_ids : dict[tuple(int, ...), obj]
    """
    if isinstance(entry, list):
        for i, item in enumerate(entry):
            yield from _infer_tile_ids_from_nested_list(item, current_pos + (i,))
    else:
        yield (current_pos, entry)

def _check_dimension_depth_tile_ids(combined_tile_ids):
    """
    Check all tuples are the same length, i.e. check that all lists are
    nested to the same depth.
    """
    depths = {len(key) for key in combined_tile_ids.keys()}
    if len(depths) > 1:
        raise ValueError("Inconsistent depths in combined_tile_ids")

def _check_shape_tile_ids(combined_tile_ids):
    """Check all lists along one dimension are same length."""
    if not combined_tile_ids:
        return

    depths = [len(key) for key in combined_tile_ids.keys()]
    max_depth = max(depths)

    for dim in range(max_depth):
        indices = [key[dim] for key in combined_tile_ids.keys() if len(key) > dim]
        if len(set(indices)) != max(indices) + 1:
            raise ValueError(f"Inconsistent lengths along dimension {dim}")

def _combine_nd(combined_ids, concat_dims, data_vars='all', coords='different', compat: CompatOptions='no_conflicts', fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop'):
    """
    Combines an N-dimensional structure of datasets into one by applying a
    series of either concat and merge operations along each dimension.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_ids : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match the length of the tuples used as
        keys in combined_ids. If the string is a dimension name then concat
        along that dimension, if it is None then merge.

    Returns
    -------
    combined_ds : xarray.Dataset
    """
    if len(combined_ids) == 1:
        return next(iter(combined_ids.values()))

    # Recursively combine datasets along each dimension
    def _combine_dim(dim_index, combined_ids):
        if dim_index == len(concat_dims):
            return next(iter(combined_ids.values()))

        dim = concat_dims[dim_index]
        grouped = {}
        for key, ds in combined_ids.items():
            grouped.setdefault(key[:dim_index], []).append(ds)

        if dim is None:
            # Merge datasets
            return merge(
                [_combine_dim(dim_index + 1, {k: v for k, v in combined_ids.items() if k[:dim_index] == key})
                 for key in grouped],
                compat=compat,
                fill_value=fill_value,
                join=join,
                combine_attrs=combine_attrs
            )
        else:
            # Concatenate datasets
            return concat(
                [_combine_dim(dim_index + 1, {k[dim_index + 1:]: v for k, v in combined_ids.items() if k[:dim_index] == key})
                 for key in grouped],
                dim=dim,
                data_vars=data_vars,
                coords=coords,
                compat=compat,
                fill_value=fill_value,
                join=join,
                combine_attrs=combine_attrs
            )

    return _combine_dim(0, combined_ids)

def _combine_1d(datasets, concat_dim, compat: CompatOptions='no_conflicts', data_vars='all', coords='different', fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop'):
    """
    Applies either concat or merge to 1D list of datasets depending on value
    of concat_dim
    """
    if concat_dim is not None:
        return concat(datasets, dim=concat_dim, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    else:
        return merge(datasets, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
DATASET_HYPERCUBE = Union[Dataset, Iterable['DATASET_HYPERCUBE']]

def combine_nested(datasets: DATASET_HYPERCUBE, concat_dim: str | DataArray | None | Sequence[str | DataArray | pd.Index | None], compat: str='no_conflicts', data_vars: str='all', coords: str='different', fill_value: object=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop') -> Dataset:
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort the supplied datasets under any circumstances, so the
    datasets must be passed in the order you wish them to be concatenated. It
    does align coordinates, but different variables on datasets can cause it to
    fail under some scenarios. In complex cases, you may need to clean up your
    data and use concat/merge explicitly.

    To concatenate along multiple dimensions the datasets must be passed as a
    nested list-of-lists, with a depth equal to the length of ``concat_dims``.
    ``combine_nested`` will concatenate along the top-level list first.

    Useful for combining datasets from a set of nested directories, or for
    collecting the output of a simulation parallelized along multiple
    dimensions.

    Parameters
    ----------
    datasets : list or nested list of Dataset
        Dataset objects to combine.
        If concatenation or merging along more than one dimension is desired,
        then datasets must be supplied in a nested list-of-lists.
    concat_dim : str, or list of str, DataArray, Index or None
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        Set ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {"identical", "equals", "broadcast_equals",               "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential merge conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    data_vars : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    coords : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
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
    combined : xarray.Dataset

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each process wrote out to a separate file. A domain which was decomposed
    into 4 parts, 2 each along both the x and y axes, requires organising the
    datasets into a doubly-nested list, e.g:

    >>> x1y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x1y1
    <xarray.Dataset> Size: 64B
    Dimensions:        (x: 2, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 32B 1.764 0.4002 0.9787 2.241
        precipitation  (x, y) float64 32B 1.868 -0.9773 0.9501 -0.1514
    >>> x1y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )


    >>> ds_grid = [[x1y1, x1y2], [x2y1, x2y2]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["x", "y"])
    >>> combined
    <xarray.Dataset> Size: 256B
    Dimensions:        (x: 4, y: 4)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 128B 1.764 0.4002 -0.1032 ... 0.04576 -0.1872
        precipitation  (x, y) float64 128B 1.868 -0.9773 0.761 ... 0.1549 0.3782

    ``combine_nested`` can also be used to explicitly merge datasets with
    different variables. For example if we have 4 datasets, which are divided
    along two times, and contain two different variables, we can pass ``None``
    to ``concat_dim`` to specify the dimension of the nested list over which
    we wish to use ``merge`` instead of ``concat``:

    >>> t1temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t1temp
    <xarray.Dataset> Size: 40B
    Dimensions:      (t: 5)
    Dimensions without coordinates: t
    Data variables:
        temperature  (t) float64 40B -0.8878 -1.981 -0.3479 0.1563 1.23

    >>> t1precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})
    >>> t1precip
    <xarray.Dataset> Size: 40B
    Dimensions:        (t: 5)
    Dimensions without coordinates: t
    Data variables:
        precipitation  (t) float64 40B 1.202 -0.3873 -0.3023 -1.049 -1.42

    >>> t2temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t2precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})


    >>> ds_grid = [[t1temp, t1precip], [t2temp, t2precip]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["t", None])
    >>> combined
    <xarray.Dataset> Size: 160B
    Dimensions:        (t: 10)
    Dimensions without coordinates: t
    Data variables:
        temperature    (t) float64 80B -0.8878 -1.981 -0.3479 ... -0.4381 -1.253
        precipitation  (t) float64 80B 1.202 -0.3873 -0.3023 ... -0.8955 0.3869

    See also
    --------
    concat
    merge
    """
    if not isinstance(concat_dim, (list, tuple)):
        concat_dim = [concat_dim]
    
    if len(concat_dim) == 0:
        raise ValueError("concat_dim must be specified")

    combined_ids = _infer_tile_ids_from_nested_list(datasets)
    _check_dimension_depth_tile_ids(combined_ids)
    _check_shape_tile_ids(combined_ids)

    return _combine_nd(combined_ids, concat_dim, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)

def _combine_single_variable_hypercube(datasets, fill_value=dtypes.NA, data_vars='all', coords='different', compat: CompatOptions='no_conflicts', join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='no_conflicts'):
    """
    Attempt to combine a list of Datasets into a hypercube using their
    coordinates.

    All provided Datasets must belong to a single variable, ie. must be
    assigned the same variable name. This precondition is not checked by this
    function, so the caller is assumed to know what it's doing.

    This function is NOT part of the public API.
    """
    if len(datasets) == 0:
        return Dataset()

    if len(datasets) == 1:
        return datasets[0]

    all_coords = set()
    for ds in datasets:
        all_coords.update(ds.coords)

    concat_dims = []
    for coord in all_coords:
        values = set()
        for ds in datasets:
            if coord in ds.coords:
                values.update(ds.coords[coord].values)
        if len(values) > 1:
            concat_dims.append(coord)

    if not concat_dims:
        return merge(datasets, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)

    combined_ids = {}
    for ds in datasets:
        key = tuple(ds.coords[dim].values[0] if dim in ds.coords else None for dim in concat_dims)
        combined_ids[key] = ds

    return _combine_nd(combined_ids, concat_dims, data_vars=data_vars, coords=coords, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)

def combine_by_coords(data_objects: Iterable[Dataset | DataArray]=[], compat: CompatOptions='no_conflicts', data_vars: Literal['all', 'minimal', 'different'] | list[str]='all', coords: str='different', fill_value: object=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='no_conflicts') -> Dataset | DataArray:
    """

    Attempt to auto-magically combine the given datasets (or data arrays)
    into one by using dimension coordinates.

    This function attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `combine_nested`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    data_objects : Iterable of Datasets or DataArrays
        Data objects to combine.

    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset

    data_vars : {"minimal", "different", "all" or list of str}, optional
        These data variables will be concatenated together:

        - "minimal": Only data variables in which the dimension already
          appears are included.
        - "different": Data variables which are not equal (ignoring
          attributes) across all datasets are also concatenated (as well as
          all for which dimension already appears). Beware: this option may
          load the data payload of data variables into memory if they are not
          already loaded.
        - "all": All data variables will be concatenated.
        - list of str: The listed data variables will be concatenated, in
          addition to the "minimal" data variables.

        If objects are DataArrays, `data_vars` must be "all".
    coords : {"minimal", "different", "all"} or list of str, optional
        As per the "data_vars" kwarg, but for coordinate variables.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values. If None, raises a ValueError if
        the passed Datasets do not create a complete hypercube.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "no_conflicts"
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
    combined : xarray.Dataset or xarray.DataArray
        Will return a Dataset unless all the inputs are unnamed DataArrays, in which case a
        DataArray will be returned.

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> x1 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [0, 1], "x": [10, 20, 30]},
    ... )
    >>> x2 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [10, 20, 30]},
    ... )
    >>> x3 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [40, 50, 60]},
    ... )

    >>> x1
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 0 1
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 48B 10.98 14.3 12.06 10.9 8.473 12.92
        precipitation  (y, x) float64 48B 0.4376 0.8918 0.9637 0.3834 0.7917 0.5289

    >>> x2
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 2 3
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 48B 11.36 18.51 1.421 1.743 0.4044 16.65
        precipitation  (y, x) float64 48B 0.7782 0.87 0.9786 0.7992 0.4615 0.7805

    >>> x3
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 2 3
      * x              (x) int64 24B 40 50 60
    Data variables:
        temperature    (y, x) float64 48B 2.365 12.8 2.867 18.89 10.44 8.293
        precipitation  (y, x) float64 48B 0.2646 0.7742 0.4562 0.5684 0.01879 0.6176

    >>> xr.combine_by_coords([x2, x1])
    <xarray.Dataset> Size: 248B
    Dimensions:        (y: 4, x: 3)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 96B 10.98 14.3 12.06 ... 1.743 0.4044 16.65
        precipitation  (y, x) float64 96B 0.4376 0.8918 0.9637 ... 0.4615 0.7805

    >>> xr.combine_by_coords([x3, x1])
    <xarray.Dataset> Size: 464B
    Dimensions:        (y: 4, x: 6)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 192B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 192B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    >>> xr.combine_by_coords([x3, x1], join="override")
    <xarray.Dataset> Size: 256B
    Dimensions:        (y: 2, x: 6)
    Coordinates:
      * y              (y) int64 16B 0 1
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 96B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 96B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    >>> xr.combine_by_coords([x1, x2, x3])
    <xarray.Dataset> Size: 464B
    Dimensions:        (y: 4, x: 6)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 192B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 192B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    You can also combine DataArray objects, but the behaviour will differ depending on
    whether or not the DataArrays are named. If all DataArrays are named then they will
    be promoted to Datasets before combining, and then the resultant Dataset will be
    returned, e.g.

    >>> named_da1 = xr.DataArray(
    ...     name="a", data=[1.0, 2.0], coords={"x": [0, 1]}, dims="x"
    ... )
    >>> named_da1
    <xarray.DataArray 'a' (x: 2)> Size: 16B
    array([1., 2.])
    Coordinates:
      * x        (x) int64 16B 0 1

    >>> named_da2 = xr.DataArray(
    ...     name="a", data=[3.0, 4.0], coords={"x": [2, 3]}, dims="x"
    ... )
    >>> named_da2
    <xarray.DataArray 'a' (x: 2)> Size: 16B
    array([3., 4.])
    Coordinates:
      * x        (x) int64 16B 2 3

    >>> xr.combine_by_coords([named_da1, named_da2])
    <xarray.Dataset> Size: 64B
    Dimensions:  (x: 4)
    Coordinates:
      * x        (x) int64 32B 0 1 2 3
    Data variables:
        a        (x) float64 32B 1.0 2.0 3.0 4.0

    If all the DataArrays are unnamed, a single DataArray will be returned, e.g.

    >>> unnamed_da1 = xr.DataArray(data=[1.0, 2.0], coords={"x": [0, 1]}, dims="x")
    >>> unnamed_da2 = xr.DataArray(data=[3.0, 4.0], coords={"x": [2, 3]}, dims="x")
    >>> xr.combine_by_coords([unnamed_da1, unnamed_da2])
    <xarray.DataArray (x: 4)> Size: 32B
    array([1., 2., 3., 4.])
    Coordinates:
      * x        (x) int64 32B 0 1 2 3

    Finally, if you attempt to combine a mix of unnamed DataArrays with either named
    DataArrays or Datasets, a ValueError will be raised (as this is an ambiguous operation).
    """
    if not data_objects:
        return Dataset()

    if all(isinstance(obj, DataArray) for obj in data_objects):
        if all(obj.name is not None for obj in data_objects):
            data_objects = [obj.to_dataset() for obj in data_objects]
        elif any(obj.name is not None for obj in data_objects):
            raise ValueError("Cannot combine mix of named and unnamed DataArrays")
        else:
            return _combine_single_variable_hypercube(data_objects, fill_value=fill_value, data_vars=data_vars, coords=coords, compat=compat, join=join, combine_attrs=combine_attrs)

    datasets = [obj if isinstance(obj, Dataset) else obj.to_dataset() for obj in data_objects]

    # Identify the dimensions to concatenate along
    concat_dims = set()
    for ds in datasets:
        for dim in ds.dims:
            if dim in ds.coords and any(not ds.coords[dim].equals(other.coords[dim]) for other in datasets if dim in other.coords):
                concat_dims.add(dim)

    # Sort datasets based on coordinate values
    def sort_key(ds):
        return tuple(ds.coords[dim].values[0] if dim in ds.coords else np.nan for dim in concat_dims)

    sorted_datasets = sorted(datasets, key=sort_key)

    # Combine datasets
    combined = _combine_nd(
        {tuple(ds.coords[dim].values[0] if dim in ds.coords else None for dim in concat_dims): ds for ds in sorted_datasets},
        list(concat_dims),
        data_vars=data_vars,
        coords=coords,
        compat=compat,
        fill_value=fill_value,
        join=join,
        combine_attrs=combine_attrs
    )

    return combined
