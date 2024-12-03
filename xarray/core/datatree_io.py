from __future__ import annotations
from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal, get_args
from xarray.core.datatree import DataTree
from xarray.core.types import NetcdfWriteModes, ZarrWriteModes
T_DataTreeNetcdfEngine = Literal['netcdf4', 'h5netcdf']
T_DataTreeNetcdfTypes = Literal['NETCDF4']

def _datatree_to_netcdf(dt: DataTree, filepath: str | PathLike, mode: NetcdfWriteModes='w', encoding: Mapping[str, Any] | None=None, unlimited_dims: Mapping | None=None, format: T_DataTreeNetcdfTypes | None=None, engine: T_DataTreeNetcdfEngine | None=None, group: str | None=None, compute: bool=True, **kwargs):
    """This function creates an appropriate datastore for writing a datatree to
    disk as a netCDF file.

    See `DataTree.to_netcdf` for full API docs.
    """
    from xarray.backends.api import to_netcdf

    # Convert DataTree to a nested dictionary of Datasets
    datasets = {}
    for path, node in dt.to_dict().items():
        if path == '/':
            path = ''
        datasets[path] = node

    # Write the root dataset
    root_ds = datasets.pop('', None)
    if root_ds is not None:
        to_netcdf(root_ds, filepath, mode=mode, encoding=encoding, unlimited_dims=unlimited_dims,
                  format=format, engine=engine, group=group, compute=compute, **kwargs)

    # Write the child datasets as groups
    for path, ds in datasets.items():
        group_path = f"/{path}" if path.startswith('/') else path
        to_netcdf(ds, filepath, mode='a', encoding=encoding, unlimited_dims=unlimited_dims,
                  format=format, engine=engine, group=group_path, compute=compute, **kwargs)

    return None

def _datatree_to_zarr(dt: DataTree, store: MutableMapping | str | PathLike[str], mode: ZarrWriteModes='w-', encoding: Mapping[str, Any] | None=None, consolidated: bool=True, group: str | None=None, compute: Literal[True]=True, **kwargs):
    """This function creates an appropriate datastore for writing a datatree
    to a zarr store.

    See `DataTree.to_zarr` for full API docs.
    """
    from xarray.backends.api import to_zarr
    import zarr

    # Convert DataTree to a nested dictionary of Datasets
    datasets = dt.to_dict()

    # Write the root dataset
    root_ds = datasets.pop('/', None)
    if root_ds is not None:
        to_zarr(root_ds, store, mode=mode, encoding=encoding, consolidated=False,
                group=group, compute=compute, **kwargs)

    # Write the child datasets as groups
    for path, ds in datasets.items():
        group_path = f"{group}/{path}" if group else path
        to_zarr(ds, store, mode='a', encoding=encoding, consolidated=False,
                group=group_path, compute=compute, **kwargs)

    # Consolidate metadata if requested
    if consolidated:
        zarr.consolidate_metadata(store)

    return None
