"""
Useful for:

* users learning xarray
* building tutorials in the documentation.

"""
from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING
import numpy as np
from xarray.backends.api import open_dataset as _open_dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
if TYPE_CHECKING:
    from xarray.backends.api import T_Engine
_default_cache_dir_name = 'xarray_tutorial_data'
base_url = 'https://github.com/pydata/xarray-data'
version = 'master'
external_urls = {}
file_formats = {'air_temperature': 3, 'air_temperature_gradient': 4, 'ASE_ice_velocity': 4, 'basin_mask': 4, 'ersstv5': 4, 'rasm': 3, 'ROMS_example': 4, 'tiny': 3, 'eraint_uvz': 3}

def open_dataset(name: str, cache: bool=True, cache_dir: None | str | os.PathLike=None, *, engine: T_Engine=None, **kws) -> Dataset:
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"ASE_ice_velocity"``: MEaSUREs InSAR-Based Ice Velocity of the Amundsen Sea Embayment, Antarctica, Version 1
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    tutorial.load_dataset
    open_dataset
    load_dataset
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir_name
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    file_format = file_formats.get(name, 3)
    dataset_id = f"{base_url}/raw/{version}/{name}.{file_format}"
    local_file = os.path.join(cache_dir, f"{name}.{file_format}")
    
    if os.path.exists(local_file):
        ds = xr.open_dataset(local_file, engine=engine, **kws)
    else:
        try:
            import pooch
        except ImportError:
            raise ImportError("pooch is required to download datasets. Please install it using pip or conda.")
        
        # Download the file
        pooch.retrieve(
            url=dataset_id,
            known_hash=None,
            path=cache_dir,
            fname=f"{name}.{file_format}",
        )
        ds = xr.open_dataset(local_file, engine=engine, **kws)
    
    if not cache:
        ds = ds.load()
        os.remove(local_file)
    
    return ds

def load_dataset(*args, **kwargs) -> Dataset:
    """
    Open, load into memory, and close a dataset from the online repository
    (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    tutorial.open_dataset
    open_dataset
    load_dataset
    """
    ds = open_dataset(*args, **kwargs)
    ds.load()
    return ds

def scatter_example_dataset(*, seed: None | int=None) -> Dataset:
    """
    Create an example dataset.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generation.
    """
    rng = np.random.default_rng(seed)
    
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Generate random data
    Z = rng.normal(size=(100, 100))
    
    # Create a Dataset
    ds = xr.Dataset(
        data_vars={
            "Z": (("x", "y"), Z),
        },
        coords={
            "x": x,
            "y": y,
        },
    )
    
    # Add some metadata
    ds.attrs["title"] = "Example Scatter Dataset"
    ds.attrs["description"] = "A randomly generated dataset for demonstration purposes"
    
    return ds
