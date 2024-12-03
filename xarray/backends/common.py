from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree
    from xarray.core.types import NestedSequence
logger = logging.getLogger(__name__)
NONE_VAR_NAME = '__values__'

def _normalize_path(path):
    """
    Normalize pathlikes to string.

    Parameters
    ----------
    path :
        Path to file.

    Examples
    --------
    >>> from pathlib import Path

    >>> directory = Path(xr.backends.common.__file__).parent
    >>> paths_path = Path(directory).joinpath("comm*n.py")
    >>> paths_str = xr.backends.common._normalize_path(paths_path)
    >>> print([type(p) for p in (paths_str,)])
    [<class 'str'>]
    """
    return str(path)

def _find_absolute_paths(paths: str | os.PathLike | NestedSequence[str | os.PathLike], **kwargs) -> list[str]:
    """
    Find absolute paths from the pattern.

    Parameters
    ----------
    paths :
        Path(s) to file(s). Can include wildcards like * .
    **kwargs :
        Extra kwargs. Mainly for fsspec.

    Examples
    --------
    >>> from pathlib import Path

    >>> directory = Path(xr.backends.common.__file__).parent
    >>> paths = str(Path(directory).joinpath("comm*n.py"))  # Find common with wildcard
    >>> paths = xr.backends.common._find_absolute_paths(paths)
    >>> [Path(p).name for p in paths]
    ['common.py']
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    
    absolute_paths = []
    for path in paths:
        normalized_path = _normalize_path(path)
        if is_remote_uri(normalized_path):
            absolute_paths.append(normalized_path)
        else:
            absolute_paths.extend(glob(os.path.abspath(normalized_path), recursive=True))
    
    return absolute_paths

def find_root_and_group(ds):
    """Find the root and group name of a netCDF4/h5netcdf dataset."""
    while ds.parent is not None:
        ds = ds.parent
    group = ds.path
    return ds, group

def robust_getitem(array, key, catch=Exception, max_retries=6, initial_delay=500):
    """
    Robustly index an array, using retry logic with exponential backoff if any
    of the errors ``catch`` are raised. The initial_delay is measured in ms.

    With the default settings, the maximum delay will be in the range of 32-64
    seconds.
    """
    for n in range(max_retries):
        try:
            return array[key]
        except catch:
            if n == max_retries - 1:
                raise
            delay = initial_delay * 2**n
            time.sleep(delay / 1000.0)

class BackendArray(NdimSizeLenMixin, indexing.ExplicitlyIndexed):
    __slots__ = ()

class AbstractDataStore:
    __slots__ = ()

    def load(self):
        """
        This loads the variables and attributes simultaneously.
        A centralized loading function makes it easier to create
        data stores that do automatic encoding/decoding.

        For example::

            class SuffixAppendingDataStore(AbstractDataStore):

                def load(self):
                    variables, attributes = AbstractDataStore.load(self)
                    variables = {'%s_suffix' % k: v
                                 for k, v in variables.items()}
                    attributes = {'%s_suffix' % k: v
                                  for k, v in attributes.items()}
                    return variables, attributes

        This function will be called anytime variables or attributes
        are requested, so care should be taken to make sure its fast.
        """
        raise NotImplementedError("Abstract method")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class ArrayWriter:
    __slots__ = ('sources', 'targets', 'regions', 'lock')

    def __init__(self, lock=None):
        self.sources = []
        self.targets = []
        self.regions = []
        self.lock = lock

class AbstractWritableDataStore(AbstractDataStore):
    __slots__ = ()

    def encode(self, variables, attributes):
        """
        Encode the variables and attributes in this store

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs

        Returns
        -------
        variables : dict-like
        attributes : dict-like

        """
        encoded_variables = {k: self.encode_variable(v) for k, v in variables.items()}
        encoded_attributes = {k: self.encode_attribute(v) for k, v in attributes.items()}
        return encoded_variables, encoded_attributes

    def encode_variable(self, v):
        """encode one variable"""
        return v

    def encode_attribute(self, a):
        """encode one attribute"""
        return a

    def store_dataset(self, dataset):
        """
        in stores, variables are all variables AND coordinates
        in xarray.Dataset variables are variables NOT coordinates,
        so here we pass the whole dataset in instead of doing
        dataset.variables
        """
        self.store(dataset.variables, dataset.attrs)

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
        """
        variables, attributes = self.encode(variables, attributes)
        self.set_dimensions(variables, unlimited_dims)
        self.set_attributes(attributes)
        self.set_variables(variables, check_encoding_set, writer, unlimited_dims)

    def set_attributes(self, attributes):
        """
        This provides a centralized method to set the dataset attributes on the
        data store.

        Parameters
        ----------
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        """
        raise NotImplementedError("Abstract method")

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
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        raise NotImplementedError("Abstract method")

    def set_dimensions(self, variables, unlimited_dims=None):
        """
        This provides a centralized method to set the dimensions on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        raise NotImplementedError("Abstract method")

class WritableCFDataStore(AbstractWritableDataStore):
    __slots__ = ()

class BackendEntrypoint:
    """
    ``BackendEntrypoint`` is a class container and it is the main interface
    for the backend plugins, see :ref:`RST backend_entrypoint`.
    It shall implement:

    - ``open_dataset`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~xarray.Dataset`.
      It shall take in input at least ``filename_or_obj`` argument and
      ``drop_variables`` keyword argument.
      For more details see :ref:`RST open_dataset`.
    - ``guess_can_open`` method: it shall return ``True`` if the backend is able to open
      ``filename_or_obj``, ``False`` otherwise. The implementation of this
      method is not mandatory.
    - ``open_datatree`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~datatree.DataTree`.
      It shall take in input at least ``filename_or_obj`` argument. The
      implementation of this method is not mandatory.  For more details see
      <reference to open_datatree documentation>.

    Attributes
    ----------

    open_dataset_parameters : tuple, default: None
        A list of ``open_dataset`` method parameters.
        The setting of this attribute is not mandatory.
    description : str, default: ""
        A short string describing the engine.
        The setting of this attribute is not mandatory.
    url : str, default: ""
        A string with the URL to the backend's documentation.
        The setting of this attribute is not mandatory.
    """
    open_dataset_parameters: ClassVar[tuple | None] = None
    description: ClassVar[str] = ''
    url: ClassVar[str] = ''

    def __repr__(self) -> str:
        txt = f'<{type(self).__name__}>'
        if self.description:
            txt += f'\n  {self.description}'
        if self.url:
            txt += f'\n  Learn more at {self.url}'
        return txt

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, drop_variables: str | Iterable[str] | None=None, **kwargs: Any) -> Dataset:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """
        return False

    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs: Any) -> DataTree:
        """
        Backend open_datatree method used by Xarray in :py:func:`~xarray.open_datatree`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
BACKEND_ENTRYPOINTS: dict[str, tuple[str | None, type[BackendEntrypoint]]] = {}
