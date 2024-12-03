from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
from packaging.version import Version
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available
if TYPE_CHECKING:
    from xarray.namedarray._typing import T_Chunks, _DType_co, _NormalizedChunks, duckarray
    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray[Any, Any]
dask_available = module_available('dask')

class DaskManager(ChunkManagerEntrypoint['DaskArray']):
    array_cls: type[DaskArray]
    available: bool = dask_available

    def __init__(self) -> None:
        from dask.array import Array
        self.array_cls = Array

    def normalize_chunks(self, chunks: T_Chunks | _NormalizedChunks, shape: tuple[int, ...] | None=None, limit: int | None=None, dtype: _DType_co | None=None, previous_chunks: _NormalizedChunks | None=None) -> _NormalizedChunks:
        """
        Normalize given chunking pattern into an explicit tuple of tuples representation.

        Parameters
        ----------
        chunks : tuple, int, dict, or string
            The chunks to be normalized.
        shape : tuple of ints, optional
            The shape of the array
        limit : int, optional
            The maximum block size to target in bytes,
            if freedom is given to choose
        dtype : np.dtype, optional
            The dtype of the array
        previous_chunks : tuple of tuples of ints, optional
            Chunks from a previous array that we should use for inspiration when
            rechunking dimensions automatically.

        Returns
        -------
        normalized_chunks : tuple of tuples of ints
            The normalized chunks.
        """
        import dask.array as da
        return da.core.normalize_chunks(chunks, shape, limit, dtype, previous_chunks)
