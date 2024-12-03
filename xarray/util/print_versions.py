"""Utility functions for printing version information."""
import importlib
import locale
import os
import platform
import struct
import subprocess
import sys

def get_sys_info():
    """Returns system information as a dict"""
    blob = []
    
    # Python and OS information
    blob.extend([
        ("python", sys.version.replace("\n", " ")),
        ("python-bits", struct.calcsize("P") * 8),
        ("OS", platform.platform()),
        ("OS-release", platform.release()),
        ("machine", platform.machine()),
        ("processor", platform.processor()),
        ("byteorder", sys.byteorder),
        ("LC_ALL", os.environ.get("LC_ALL", "None")),
        ("LANG", os.environ.get("LANG", "None")),
        ("LOCALE", ".".join(locale.getlocale())),
    ])

    return dict(blob)

def show_versions(file=sys.stdout):
    """print the versions of xarray and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    print("\nSYSTEM INFO", file=file)
    print("-----------", file=file)
    for k, v in sys_info.items():
        print(f"{k:>20}: {v}", file=file)

    print("\nPACKAGE VERSIONS", file=file)
    print("----------------", file=file)
    packages = [
        "xarray",
        "pandas",
        "numpy",
        "scipy",
        "netCDF4",
        "pydap",
        "h5netcdf",
        "h5py",
        "Nio",
        "zarr",
        "cftime",
        "nc_time_axis",
        "matplotlib",
        "cartopy",
        "seaborn",
        "dask",
        "distributed",
        "bottleneck",
        "sparse",
        "numbagg",
        "pint",
        "setuptools",
        "pip",
        "conda",
    ]

    for modname in packages:
        try:
            mod = importlib.import_module(modname)
            version = getattr(mod, "__version__", "installed (no version available)")
        except ImportError:
            version = "not installed"
        print(f"{modname:>20}: {version}", file=file)

    print("\nC EXTENSIONS", file=file)
    print("------------", file=file)
    try:
        import xarray._primitives
        print(f"{'xarray._primitives':>20}: {'built' if xarray._primitives.compiled else 'not built'}", file=file)
    except ImportError:
        print(f"{'xarray._primitives':>20}: not available", file=file)
if __name__ == '__main__':
    show_versions()
