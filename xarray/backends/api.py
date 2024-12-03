from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Union, cast, overload
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import AbstractDataStore, ArrayWriter, _find_absolute_paths, _normalize_path
from xarray.backends.locks import _get_scheduler
from xarray.core import indexing
from xarray.core.combine import _infer_concat_order_from_positions, _nested_combine, combine_by_coords
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import NetcdfWriteModes, ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None
    from io import BufferedIOBase
    from xarray.backends.common import BackendEntrypoint
    from xarray.core.types import CombineAttrsOptions, CompatOptions, JoinOptions, NestedSequence, T_Chunks
    T_NetcdfEngine = Literal['netcdf4', 'scipy', 'h5netcdf']
    T_Engine = Union[T_NetcdfEngine, Literal['pydap', 'zarr'], type[BackendEntrypoint], str, None]
    T_NetcdfTypes = Literal['NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC']
    from xarray.core.datatree import DataTree
DATAARRAY_NAME = '__xarray_dataarray_name__'
DATAARRAY_VARIABLE = '__xarray_dataarray_variable__'
ENGINES = {'netcdf4': backends.NetCDF4DataStore.open, 'scipy': backends.ScipyDataStore, 'pydap': backends.PydapDataStore.open, 'h5netcdf': backends.H5NetCDFStore.open, 'zarr': backends.ZarrStore.open_group}

def _validate_dataset_names(dataset: Dataset) -> None:
    """DataArray.name and Dataset keys must be a string or None"""
    for k in dataset.variables:
        if not isinstance(k, str) and k is not None:
            raise ValueError("Dataset variable names must be strings or None")
    if dataset.name is not None and not isinstance(dataset.name, str):
        raise ValueError("Dataset name must be a string or None")

def _validate_attrs(dataset, invalid_netcdf=False):
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray, a list/tuple of numbers/strings, or a numpy.bool_.

    Notes
    -----
    A numpy.bool_ is only allowed when using the h5netcdf engine with
    `invalid_netcdf=True`.
    """
    def check_attr(name, value):
        if not isinstance(name, str):
            raise ValueError("Attribute names must be strings")
        if isinstance(value, (str, numbers.Number, np.ndarray)):
            return
        if isinstance(value, (list, tuple)):
            if all(isinstance(v, (str, numbers.Number)) for v in value):
                return
        if invalid_netcdf and isinstance(value, np.bool_):
            return
        raise ValueError(f"Invalid attribute value for {name}: {value}")

    for k, v in dataset.attrs.items():
        check_attr(k, v)

def _finalize_store(write, store):
    """Finalize this store by explicitly syncing and closing"""
    if write:
        if hasattr(store, 'sync'):
            store.sync()
        if hasattr(store, 'close'):
            store.close()

def load_dataset(filename_or_obj, **kwargs) -> Dataset:
    """Open, load into memory, and close a Dataset from a file or file-like
    object.

    This is a thin wrapper around :py:meth:`~xarray.open_dataset`. It differs
    from `open_dataset` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataset` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataset`. See that documentation for further details.

    Returns
    -------
    dataset : Dataset
        The newly created Dataset.

    See Also
    --------
    open_dataset
    """
    with open_dataset(filename_or_obj, **kwargs) as ds:
        return ds.load()

def load_dataarray(filename_or_obj, **kwargs):
    """Open, load into memory, and close a DataArray from a file or file-like
    object containing a single data variable.

    This is a thin wrapper around :py:meth:`~xarray.open_dataarray`. It differs
    from `open_dataarray` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataarray` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataarray`. See that documentation for further details.

    Returns
    -------
    datarray : DataArray
        The newly created DataArray.

    See Also
    --------
    open_dataarray
    """
    with open_dataarray(filename_or_obj, **kwargs) as da:
        return da.load()

def open_dataset(filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, engine: T_Engine=None, chunks: T_Chunks=None, cache: bool | None=None, decode_cf: bool | None=None, mask_and_scale: bool | Mapping[str, bool] | None=None, decode_times: bool | Mapping[str, bool] | None=None, decode_timedelta: bool | Mapping[str, bool] | None=None, use_cftime: bool | Mapping[str, bool] | None=None, concat_characters: bool | Mapping[str, bool] | None=None, decode_coords: Literal['coordinates', 'all'] | bool | None=None, drop_variables: str | Iterable[str] | None=None, inline_array: bool=False, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None, backend_kwargs: dict[str, Any] | None=None, **kwargs) -> Dataset:
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None}        , installed backend         or subclass of xarray.backends.BackendEntrypoint, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4". A custom backend class (a subclass of ``BackendEntrypoint``)
        can also be used.
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the data into dask arrays.

        - ``chunks="auto"`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using the engine's preferred chunk
          size, generally identical to the format's chunk size. If not available, a
          single chunk for all arrays.

        See dask chunking for more details.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool or dict-like, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    decode_times : bool or dict-like, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool or dict-like, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    use_cftime: bool or dict-like, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    concat_characters : bool or dict-like, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.

        Only existing variables can be set as coordinates. Missing variables
        will be silently ignored.
    drop_variables: str or iterable of str, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    inline_array: bool, default: False
        How to include the array in the dask task graph.
        By default(``inline_array=False``) the array is included in a task by
        itself, and each chunk refers to that task by its key. With
        ``inline_array=True``, Dask will instead inline the array directly
        in the values of the task graph. See :py:func:`dask.array.from_array`.
    chunked_array_type: str, optional
        Which chunked array type to coerce this datasets' arrays to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "scipy".

        See engine open function for kwargs accepted by each specific engine.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.

    See Also
    --------
    open_mfdataset
    """
    pass

def open_dataarray(filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, engine: T_Engine | None=None, chunks: T_Chunks | None=None, cache: bool | None=None, decode_cf: bool | None=None, mask_and_scale: bool | None=None, decode_times: bool | None=None, decode_timedelta: bool | None=None, use_cftime: bool | None=None, concat_characters: bool | None=None, decode_coords: Literal['coordinates', 'all'] | bool | None=None, drop_variables: str | Iterable[str] | None=None, inline_array: bool=False, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None, backend_kwargs: dict[str, Any] | None=None, **kwargs) -> DataArray:
    """Open an DataArray from a file or file-like object containing a single
    data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None}        , installed backend         or subclass of xarray.backends.BackendEntrypoint, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    chunks : int, dict, 'auto' or None, default: None
        If provided, used to load the data into dask arrays.

        - ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using engine preferred chunks if
          exposed by the backend, otherwise with a single chunk for all arrays.

        See dask chunking for more details.

    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.

        Only existing variables can be set as coordinates. Missing variables
        will be silently ignored.
    drop_variables: str or iterable of str, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    inline_array: bool, default: False
        How to include the array in the dask task graph.
        By default(``inline_array=False``) the array is included in a task by
        itself, and each chunk refers to that task by its key. With
        ``inline_array=True``, Dask will instead inline the array directly
        in the values of the task graph. See :py:func:`dask.array.from_array`.
    chunked_array_type: str, optional
        Which chunked array type to coerce the underlying data array to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEnetryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        For example if :py:func:`dask.array.Array` objects are used for chunking, additional kwargs will be passed
        to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "scipy".

        See engine open function for kwargs accepted by each specific engine.

    Notes
    -----
    This is designed to be fully compatible with `DataArray.to_netcdf`. Saving
    using `DataArray.to_netcdf` and then loading with this function will
    produce an identical result.

    All parameters are passed directly to `xarray.open_dataset`. See that
    documentation for further details.

    See also
    --------
    open_dataset
    """
    pass

def open_datatree(filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, engine: T_Engine=None, **kwargs) -> DataTree:
    """
    Open and decode a DataTree from a file or file-like object, creating one tree node for each group in the file.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file or Zarr store.
    engine : str, optional
        Xarray backend engine to use. Valid options include `{"netcdf4", "h5netcdf", "zarr"}`.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`~xarray.open_dataset` for each group.
    Returns
    -------
    xarray.DataTree
    """
    pass

def open_mfdataset(paths: str | NestedSequence[str | os.PathLike], chunks: T_Chunks | None=None, concat_dim: str | DataArray | Index | Sequence[str] | Sequence[DataArray] | Sequence[Index] | None=None, compat: CompatOptions='no_conflicts', preprocess: Callable[[Dataset], Dataset] | None=None, engine: T_Engine | None=None, data_vars: Literal['all', 'minimal', 'different'] | list[str]='all', coords='different', combine: Literal['by_coords', 'nested']='by_coords', parallel: bool=False, join: JoinOptions='outer', attrs_file: str | os.PathLike | None=None, combine_attrs: CombineAttrsOptions='override', **kwargs) -> Dataset:
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default ``combine='by_coords'``
    will be used. Requires dask to be installed. See documentation for
    details on dask [1]_. Global attributes from the ``attrs_file`` are used
    for the combined dataset.

    Parameters
    ----------
    paths : str or nested sequence of paths
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int, dict, 'auto' or None, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_. This argument is evaluated
        on a per-file basis, so chunk sizes that span multiple files will be ignored.
    concat_dim : str, DataArray, Index or a Sequence of these or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='nested'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        ``xarray.merge``.
    combine : {"by_coords", "nested"}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. Default is to use ``xarray.combine_by_coords``.
    compat : {"identical", "equals", "broadcast_equals",               "no_conflicts", "override"}, default: "no_conflicts"
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

         * "broadcast_equals": all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * "equals": all values and dimensions must be the same.
         * "identical": all values, dimensions and attributes must be the
           same.
         * "no_conflicts": only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
         * "override": skip comparing and pick variable from first dataset

    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None}        , installed backend         or subclass of xarray.backends.BackendEntrypoint, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    data_vars : {"minimal", "different", "all"} or list of str, default: "all"
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    coords : {"minimal", "different", "all"} or list of str, optional
        These coordinate variables will be concatenated together:
         * "minimal": Only coordinates in which the dimension already appears
           are included.
         * "different": Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * "all": All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the "minimal" coordinates.
    parallel : bool, default: False
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {"outer", "inner", "left", "right", "exact", "override"}, default: "outer"
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
    attrs_file : str or path-like, optional
        Path of the file used to read global attributes from.
        By default global attributes are read from the first file provided,
        with wildcard matches sorted by filename.
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
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`. For an
        overview of some of the possible options, see the documentation of
        :py:func:`xarray.open_dataset`

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    ``open_mfdataset`` opens files with read-only access. When you modify values
    of a Dataset, even one linked to files on disk, only the in-memory copy you
    are manipulating in xarray is modified: the original file on disk is never
    touched.

    See Also
    --------
    combine_by_coords
    combine_nested
    open_dataset

    Examples
    --------
    A user might want to pass additional arguments into ``preprocess`` when
    applying some operation to many individual files that are being opened. One route
    to do this is through the use of ``functools.partial``.

    >>> from functools import partial
    >>> def _preprocess(x, lon_bnds, lat_bnds):
    ...     return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
    ...
    >>> lon_bnds, lat_bnds = (-110, -105), (40, 45)
    >>> partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    >>> ds = xr.open_mfdataset(
    ...     "file_*.nc", concat_dim="time", preprocess=partial_func
    ... )  # doctest: +SKIP

    It is also possible to use any argument to ``open_dataset`` together
    with ``open_mfdataset``, such as for example ``drop_variables``:

    >>> ds = xr.open_mfdataset(
    ...     "file.nc", drop_variables=["varname_1", "varname_2"]  # any list of vars
    ... )  # doctest: +SKIP

    References
    ----------

    .. [1] https://docs.xarray.dev/en/stable/dask.html
    .. [2] https://docs.xarray.dev/en/stable/dask.html#chunking-and-performance
    """
    pass
WRITEABLE_STORES: dict[T_NetcdfEngine, Callable] = {'netcdf4': backends.NetCDF4DataStore.open, 'scipy': backends.ScipyDataStore, 'h5netcdf': backends.H5NetCDFStore.open}

def to_netcdf(dataset: Dataset, path_or_file: str | os.PathLike | None=None, mode: NetcdfWriteModes='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: bool=True, multifile: bool=False, invalid_netcdf: bool=False) -> tuple[ArrayWriter, AbstractDataStore] | bytes | Delayed | None:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    pass

def dump_to_store(dataset, store, writer=None, encoder=None, encoding=None, unlimited_dims=None):
    """Store dataset contents to a backends.*DataStore object."""
    pass

def save_mfdataset(datasets, paths, mode='w', format=None, groups=None, engine=None, compute=True, **kwargs):
    """Write multiple datasets to disk as netCDF files simultaneously.

    This function is intended for use with datasets consisting of dask.array
    objects, in which case it can write the multiple datasets to disk
    simultaneously using a shared thread pool.

    When not using dask, it is no different than calling ``to_netcdf``
    repeatedly.

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets to save.
    paths : list of str or list of path-like objects
        List of paths to which to save each corresponding dataset.
    mode : {"w", "a"}, optional
        Write ("w") or append ("a") mode. If mode="w", any existing file at
        these locations will be overwritten.
    format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT",               "NETCDF3_CLASSIC"}, optional
        File format for the resulting netCDF file:

        * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
          features.
        * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
          netCDF 3 compatible API features.
        * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
          which fully supports 2+ GB files, but is only compatible with
          clients linked against netCDF version 3.6.0 or later.
        * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
          handle 2+ GB files very well.

        All formats are supported by the netCDF4-python library.
        scipy.io.netcdf only supports the last two formats.

        The default format is NETCDF4 if you are saving a file to disk and
        have the netCDF4-python library available. Otherwise, xarray falls
        back to using scipy to write netCDF files and defaults to the
        NETCDF3_64BIT format (scipy does not support netCDF4).
    groups : list of str, optional
        Paths to the netCDF4 group in each corresponding file to which to save
        datasets (only works for format="NETCDF4"). The groups will be created
        if necessary.
    engine : {"netcdf4", "scipy", "h5netcdf"}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, with a
        preference for "netcdf4" if writing to a file on disk.
        See `Dataset.to_netcdf` for additional information.
    compute : bool
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.
    **kwargs : dict, optional
        Additional arguments are passed along to ``to_netcdf``.

    Examples
    --------
    Save a dataset into one netCDF per year of data:

    >>> ds = xr.Dataset(
    ...     {"a": ("time", np.linspace(0, 1, 48))},
    ...     coords={"time": pd.date_range("2010-01-01", freq="ME", periods=48)},
    ... )
    >>> ds
    <xarray.Dataset> Size: 768B
    Dimensions:  (time: 48)
    Coordinates:
      * time     (time) datetime64[ns] 384B 2010-01-31 2010-02-28 ... 2013-12-31
    Data variables:
        a        (time) float64 384B 0.0 0.02128 0.04255 ... 0.9574 0.9787 1.0
    >>> years, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    pass

def to_zarr(dataset: Dataset, store: MutableMapping | str | os.PathLike[str] | None=None, chunk_store: MutableMapping | str | os.PathLike | None=None, mode: ZarrWriteModes | None=None, synchronizer=None, group: str | None=None, encoding: Mapping | None=None, *, compute: bool=True, consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice | Literal['auto']] | Literal['auto'] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None, write_empty_chunks: bool | None=None, chunkmanager_store_kwargs: dict[str, Any] | None=None) -> backends.ZarrStore | Delayed:
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """
    pass
