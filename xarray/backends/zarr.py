from __future__ import annotations
import json
import os
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from xarray import coding, conventions
from xarray.backends.common import BACKEND_ENTRYPOINTS, AbstractWritableDataStore, BackendArray, BackendEntrypoint, _encode_variable_name, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import FrozenDict, HiddenKeyDict, close_on_error
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree
DIMENSION_KEY = '_ARRAY_DIMENSIONS'

def encode_zarr_attr_value(value):
    """
    Encode a attribute value as something that can be serialized as json

    Many xarray datasets / variables have numpy arrays and values. This
    function handles encoding / decoding of such items.

    ndarray -> list
    scalar array -> scalar
    other -> other (no change)
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    return value

class ZarrArrayWrapper(BackendArray):
    __slots__ = ('dtype', 'shape', '_array')

    def __init__(self, zarr_array):
        self._array = zarr_array
        self.shape = self._array.shape
        if self._array.filters is not None and any([filt.codec_id == 'vlen-utf8' for filt in self._array.filters]):
            dtype = coding.strings.create_vlen_dtype(str)
        else:
            dtype = self._array.dtype
        self.dtype = dtype

    def __getitem__(self, key):
        array = self._array
        if isinstance(key, indexing.BasicIndexer):
            method = self._getitem
        elif isinstance(key, indexing.VectorizedIndexer):
            method = self._vindex
        elif isinstance(key, indexing.OuterIndexer):
            method = self._oindex
        return indexing.explicit_indexing_adapter(key, array.shape, indexing.IndexingSupport.VECTORIZED, method)

def _determine_zarr_chunks(enc_chunks, var_chunks, ndim, name, safe_chunks):
    """
    Given encoding chunks (possibly None or []) and variable chunks
    (possibly None or []).
    """
    pass

def extract_zarr_variable_encoding(variable, raise_on_invalid=False, name=None, safe_chunks=True):
    """
    Extract zarr encoding dictionary from xarray Variable

    Parameters
    ----------
    variable : Variable
    raise_on_invalid : bool, optional

    Returns
    -------
    encoding : dict
        Zarr encoding for `variable`
    """
    pass

def encode_zarr_variable(var, needs_copy=True, name=None):
    """
    Converts an Variable into an Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : Variable
        A variable which has been encoded as described above.
    """
    pass

def _validate_datatypes_for_zarr_append(vname, existing_var, new_var):
    """If variable exists in the store, confirm dtype of the data to append is compatible with
    existing dtype.
    """
    pass

def _put_attrs(zarr_obj, attrs):
    """Raise a more informative error message for invalid attrs."""
    pass

class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr"""
    __slots__ = ('zarr_group', '_append_dim', '_consolidate_on_close', '_group', '_mode', '_read_only', '_synchronizer', '_write_region', '_safe_chunks', '_write_empty', '_close_store_on_close')

    def __init__(self, zarr_group, mode=None, consolidate_on_close=False, append_dim=None, write_region=None, safe_chunks=True, write_empty: bool | None=None, close_store_on_close: bool=False):
        self.zarr_group = zarr_group
        self._read_only = self.zarr_group.read_only
        self._synchronizer = self.zarr_group.synchronizer
        self._group = self.zarr_group.path
        self._mode = mode
        self._consolidate_on_close = consolidate_on_close
        self._append_dim = append_dim
        self._write_region = write_region
        self._safe_chunks = safe_chunks
        self._write_empty = write_empty
        self._close_store_on_close = close_store_on_close

    def store(self, variables, attributes, check_encoding_set=frozenset(), writer=None, unlimited_dims=None):
        """
        Top level method for putting data on this store, this method:
          - encodes variables/attributes
          - sets dimensions
          - sets variables

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
            dimension on which the zarray will be appended
            only needed in append mode
        """
        pass

    def set_variables(self, variables, check_encoding_set, writer, unlimited_dims=None):
        """
        This provides a centralized method to set the variables on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        pass

def open_zarr(store, group=None, synchronizer=None, chunks='auto', decode_cf=True, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables=None, consolidated=None, overwrite_encoded_chunks=False, chunk_store=None, storage_options=None, decode_timedelta=None, use_cftime=None, zarr_version=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None, **kwargs):
    """Load and decode a dataset from a Zarr store.

    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute or must have NCZarr format.

    Parameters
    ----------
    store : MutableMapping or str
        A MutableMapping where a Zarr Group has been stored or a path to a
        directory in file system where a Zarr DirectoryStore has been stored.
    synchronizer : object, optional
        Array synchronizer provided to zarr
    group : str, optional
        Group path. (a.k.a. `path` in zarr terminology.)
    chunks : int, dict, 'auto' or None, default: 'auto'
        If provided, used to load the data into dask arrays.

        - ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
          engine preferred chunks.
        - ``chunks=None`` skips using dask, which is generally faster for
          small arrays.
        - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the data with dask using engine preferred chunks if
          exposed by the backend, otherwise with a single chunk for all arrays.

        See dask chunking for more details.
    overwrite_encoded_chunks : bool, optional
        Whether to drop the zarr chunks encoded for each variable when a
        dataset is loaded with specified chunk sizes (default: False)
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
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    drop_variables : str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.
        By default (`consolidate=None`), attempts to read consolidated metadata,
        falling back to read non-consolidated metadata if that fails.

        When the experimental ``zarr_version=3``, ``consolidated`` must be
        either be ``None`` or ``False``.
    chunk_store : MutableMapping, optional
        A separate Zarr store only for chunk data.
    storage_options : dict, optional
        Any additional parameters for the storage backend (ignored for local
        paths).
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds'}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
    zarr_version : int or None, optional
        The desired zarr spec version to target (currently 2 or 3). The default
        of None will attempt to determine the zarr version from ``store`` when
        possible, otherwise defaulting to 2.
    chunked_array_type: str, optional
        Which chunked array type to coerce this datasets' arrays to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        Defaults to {'manager': 'dask'}, meaning additional kwargs will be passed eventually to
        :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_dataset
    open_mfdataset

    References
    ----------
    http://zarr.readthedocs.io/
    """
    pass

class ZarrBackendEntrypoint(BackendEntrypoint):
    """
    Backend for ".zarr" files based on the zarr package.

    For more information about the underlying library, visit:
    https://zarr.readthedocs.io/en/stable

    See Also
    --------
    backends.ZarrStore
    """
    description = 'Open zarr files (.zarr) using zarr in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.ZarrBackendEntrypoint.html'
BACKEND_ENTRYPOINTS['zarr'] = ('zarr', ZarrBackendEntrypoint)
