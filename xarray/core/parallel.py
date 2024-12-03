from __future__ import annotations
import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict
import numpy as np
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.merge import merge
from xarray.core.utils import is_dask_collection
from xarray.core.variable import Variable
if TYPE_CHECKING:
    from xarray.core.types import T_Xarray

class ExpectedDict(TypedDict):
    shapes: dict[Hashable, int]
    coords: set[Hashable]
    data_vars: set[Hashable]
    indexes: dict[Hashable, Index]

def make_meta(obj):
    """If obj is a DataArray or Dataset, return a new object of the same type and with
    the same variables and dtypes, but where all variables have size 0 and numpy
    backend.
    If obj is neither a DataArray nor Dataset, return it unaltered.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    if isinstance(obj, (DataArray, Dataset)):
        if isinstance(obj, DataArray):
            data = np.array([], dtype=obj.dtype)
            coords = {k: make_meta(v) for k, v in obj.coords.items()}
            return DataArray(data, coords=coords, dims=obj.dims, name=obj.name, attrs=obj.attrs)
        else:
            data_vars = {k: make_meta(v) for k, v in obj.data_vars.items()}
            coords = {k: make_meta(v) for k, v in obj.coords.items()}
            return Dataset(data_vars, coords=coords, attrs=obj.attrs)
    return obj

def infer_template(func: Callable[..., T_Xarray], obj: DataArray | Dataset, *args, **kwargs) -> T_Xarray:
    """Infer return object by running the function on meta objects."""
    meta_obj = make_meta(obj)
    meta_args = [make_meta(arg) if isinstance(arg, (DataArray, Dataset)) else arg for arg in args]
    meta_kwargs = {k: make_meta(v) if isinstance(v, (DataArray, Dataset)) else v for k, v in kwargs.items()}
    return func(meta_obj, *meta_args, **meta_kwargs)

def make_dict(x: DataArray | Dataset) -> dict[Hashable, Any]:
    """Map variable name to numpy(-like) data
    (Dataset.to_dict() is too complicated).
    """
    if isinstance(x, DataArray):
        return {x.name: x.data}
    else:
        return {k: v.data for k, v in x.variables.items()}

def subset_dataset_to_block(graph: dict, gname: str, dataset: Dataset, input_chunk_bounds, chunk_index):
    """
    Creates a task that subsets an xarray dataset to a block determined by chunk_index.
    Block extents are determined by input_chunk_bounds.
    Also subtasks that subset the constituent variables of a dataset.
    """
    from dask.highlevelgraph import HighLevelGraph

    def subset_variable(var, input_chunk_bounds, chunk_index):
        indexers = {dim: slice(input_chunk_bounds[dim][i], input_chunk_bounds[dim][i+1])
                    for dim, i in zip(var.dims, chunk_index)}
        return var.isel(indexers)

    var_tasks = {}
    for name, var in dataset.variables.items():
        var_name = f'{gname}-{name}'
        var_tasks[var_name] = (subset_variable, var, input_chunk_bounds, chunk_index)

    def subset_dataset(variables):
        return Dataset(variables)

    ds_name = f'{gname}-subset'
    ds_task = (subset_dataset, var_tasks)

    graph.update(HighLevelGraph.from_collections(ds_name, ds_task, dependencies=var_tasks))
    return ds_name

def map_blocks(func: Callable[..., T_Xarray], obj: DataArray | Dataset, args: Sequence[Any]=(), kwargs: Mapping[str, Any] | None=None, template: DataArray | Dataset | None=None) -> T_Xarray:
    """Apply a function to each block of a DataArray or Dataset.

    .. warning::
        This function is experimental and its signature may change.

    Parameters
    ----------
    func : callable
        User-provided function that accepts a DataArray or Dataset as its first
        parameter ``obj``. The function will receive a subset or 'block' of ``obj`` (see below),
        corresponding to one chunk along each chunked dimension. ``func`` will be
        executed as ``func(subset_obj, *subset_args, **kwargs)``.

        This function must return either a single DataArray or a single Dataset.

        This function cannot add a new chunked dimension.
    obj : DataArray, Dataset
        Passed to the function as its first argument, one block at a time.
    args : sequence
        Passed to func after unpacking and subsetting any xarray objects by blocks.
        xarray objects in args must be aligned with obj, otherwise an error is raised.
    kwargs : mapping
        Passed verbatim to func after unpacking. xarray objects, if any, will not be
        subset to blocks. Passing dask collections in kwargs is not allowed.
    template : DataArray or Dataset, optional
        xarray object representing the final result after compute is called. If not provided,
        the function will be first run on mocked-up data, that looks like ``obj`` but
        has sizes 0, to determine properties of the returned object such as dtype,
        variable names, attributes, new dimensions and new indexes (if any).
        ``template`` must be provided if the function changes the size of existing dimensions.
        When provided, ``attrs`` on variables in `template` are copied over to the result. Any
        ``attrs`` set by ``func`` will be ignored.

    Returns
    -------
    obj : same as obj
        A single DataArray or Dataset with dask backend, reassembled from the outputs of the
        function.

    Notes
    -----
    This function is designed for when ``func`` needs to manipulate a whole xarray object
    subset to each block. Each block is loaded into memory. In the more common case where
    ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.

    If none of the variables in ``obj`` is backed by dask arrays, calling this function is
    equivalent to calling ``func(obj, *args, **kwargs)``.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks
    xarray.DataArray.map_blocks

    Examples
    --------
    Calculate an anomaly from climatology using ``.groupby()``. Using
    ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
    its indices, and its methods like ``.groupby()``.

    >>> def calculate_anomaly(da, groupby_type="time.month"):
    ...     gb = da.groupby(groupby_type)
    ...     clim = gb.mean(dim="time")
    ...     return gb - clim
    ...
    >>> time = xr.cftime_range("1990-01", "1992-01", freq="ME")
    >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
    >>> np.random.seed(123)
    >>> array = xr.DataArray(
    ...     np.random.rand(len(time)),
    ...     dims=["time"],
    ...     coords={"time": time, "month": month},
    ... ).chunk()
    >>> array.map_blocks(calculate_anomaly, template=array).compute()
    <xarray.DataArray (time: 24)> Size: 192B
    array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,
            0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,
           -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,
            0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,
            0.07673453,  0.22865714,  0.19063865, -0.0590131 ])
    Coordinates:
      * time     (time) object 192B 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
        month    (time) int64 192B 1 2 3 4 5 6 7 8 9 10 ... 3 4 5 6 7 8 9 10 11 12

    Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
    to the function being applied in ``xr.map_blocks()``:

    >>> array.map_blocks(
    ...     calculate_anomaly,
    ...     kwargs={"groupby_type": "time.year"},
    ...     template=array,
    ... )  # doctest: +ELLIPSIS
    <xarray.DataArray (time: 24)> Size: 192B
    dask.array<<this-array>-calculate_anomaly, shape=(24,), dtype=float64, chunksize=(24,), chunktype=numpy.ndarray>
    Coordinates:
      * time     (time) object 192B 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
        month    (time) int64 192B dask.array<chunksize=(24,), meta=np.ndarray>
    """
    pass
