from __future__ import annotations
from xarray.core import dtypes, nputils

def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    import dask.array as da
    import numpy as np

    if min_count is None:
        min_count = window

    def wrapped_func(x):
        return moving_func(x, window, min_count=min_count, axis=axis)

    if isinstance(a, da.Array):
        return da.map_overlap(wrapped_func, a, depth={axis: window - 1}, boundary='reflect')
    else:
        return wrapped_func(a)

def push(array, n, axis):
    """
    Dask-aware bottleneck.push
    """
    import dask.array as da
    import numpy as np

    def push_func(x):
        return np.roll(x, n, axis=axis)

    if isinstance(array, da.Array):
        return da.map_overlap(push_func, array, depth={axis: abs(n)}, boundary='reflect')
    else:
        return push_func(array)
