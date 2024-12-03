"""String formatting routines for __repr__.
"""
from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.datatree_render import RenderDataTree
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.iterators import LevelOrderIter
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
if TYPE_CHECKING:
    from xarray.core.coordinates import AbstractCoordinates
    from xarray.core.datatree import DataTree
UNITS = ('B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')

def pretty_print(x, numchars: int):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = str(x)
    if len(s) > numchars:
        return s[:(numchars - 3)] + '...'
    else:
        return s.ljust(numchars)

def first_n_items(array, n_desired):
    """Returns the first n_desired items of an array"""
    return array[:n_desired]

def last_n_items(array, n_desired):
    """Returns the last n_desired items of an array"""
    return array[-n_desired:]

def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    return [array[-1]] if len(array) > 0 else []

def calc_max_rows_first(max_rows: int) -> int:
    """Calculate the first rows to maintain the max number of rows."""
    return max(1, (max_rows + 1) // 2)

def calc_max_rows_last(max_rows: int) -> int:
    """Calculate the last rows to maintain the max number of rows."""
    return max(1, max_rows // 2)

def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    try:
        timestamp = pd.Timestamp(t)
        return timestamp.isoformat(sep=' ')
    except (ValueError, TypeError):
        return str(t)

def format_timedelta(t, timedelta_format=None):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    if timedelta_format is None:
        timedelta_format = 'auto'
    
    try:
        delta = pd.Timedelta(t)
        if timedelta_format == 'auto':
            return delta.isoformat()
        else:
            return delta.format(timedelta_format)
    except (ValueError, TypeError):
        return str(t)

def format_item(x, timedelta_format=None, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (pd.Timestamp, datetime)):
        return format_timestamp(x)
    elif isinstance(x, (pd.Timedelta, timedelta)):
        return format_timedelta(x, timedelta_format)
    elif isinstance(x, str):
        return repr(x) if quote_strings else x
    elif isinstance(x, (float, np.float_)):
        return f'{x:.4g}'
    else:
        return str(x)

def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    return [format_item(xi) for xi in x]

def format_array_flat(array, max_width: int):
    """Return a formatted string for as many items in the flattened version of
    array that will fit within max_width characters.
    """
    formatted = format_items(array.flat)
    
    cum_len = np.cumsum([len(f) + 2 for f in formatted])
    if cum_len[-1] < max_width:
        return ', '.join(formatted)
    
    num_items = np.argmax(cum_len > max_width)
    return ', '.join(formatted[:num_items] + ['...'])
_KNOWN_TYPE_REPRS = {('numpy', 'ndarray'): 'np.ndarray', ('sparse._coo.core', 'COO'): 'sparse.COO'}

def inline_dask_repr(array):
    """Similar to dask.array.DataArray.__repr__, but without
    redundant information that's already printed by the repr
    function of the xarray wrapper.
    """
    chunksize = tuple(c[0] for c in array.chunks)
    return f'dask.array<chunksize={chunksize}>'

def inline_sparse_repr(array):
    """Similar to sparse.COO.__repr__, but without the redundant shape/dtype."""
    return f'sparse.COO<nnz={array.nnz}, fill_value={array.fill_value}>'

def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    if var._in_memory:
        return format_array_flat(var.data, max_width)
    elif hasattr(var._data, 'name'):
        return f'[{var._data.name}]'
    elif isinstance(var._data, MemoryCachedArray):
        return '[MemoryCachedArray]'
    elif isinstance(var._data, dask.array.Array):
        return inline_dask_repr(var._data)
    elif sparse and isinstance(var._data, sparse.COO):
        return inline_sparse_repr(var._data)
    else:
        return '[{} bytes]'.format(var.nbytes)

def summarize_variable(name: Hashable, var, col_width: int, max_width: int | None=None, is_index: bool=False):
    """Summarize a variable in one line, e.g., for the Dataset.__repr__."""
    if max_width is None:
        max_width = OPTIONS["display_width"]
    first_col = pretty_print(f"  {name}:", col_width)
    if is_index:
        dims_str = f"({var.dims[0]}) "
    else:
        dims_str = f"({', '.join(var.dims)}) "
    front_str = f"{first_col}{dims_str}{var.dtype} "
    if len(front_str) > max_width / 2:
        front_str = f"{first_col}({', '.join(var.dims)}) ... "
    remaining_width = max_width - len(front_str)
    return front_str + inline_variable_array_repr(var, remaining_width)

def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    if col_width is None:
        col_width = max(len(key) + 1, 25)
    first_col = pretty_print(f"    {key}:", col_width)
    
    try:
        summary = repr(value)
    except Exception:
        summary = '(error showing value)'
    
    if len(summary) > 80:
        summary = summary[:77] + '...'
    
    return f"{first_col}{summary}"
EMPTY_REPR = '    *empty*'
data_vars_repr = functools.partial(_mapping_repr, title='Data variables', summarizer=summarize_variable, expand_option_name='display_expand_data_vars')
attrs_repr = functools.partial(_mapping_repr, title='Attributes', summarizer=summarize_attr, expand_option_name='display_expand_attrs')

def _element_formatter(elements: Collection[Hashable], col_width: int, max_rows: int | None=None, delimiter: str=', ') -> str:
    """
    Formats elements for better readability.

    Once it becomes wider than the display width it will create a newline and
    continue indented to col_width.
    Once there are more rows than the maximum displayed rows it will start
    removing rows.

    Parameters
    ----------
    elements : Collection of hashable
        Elements to join together.
    col_width : int
        The width to indent to if a newline has been made.
    max_rows : int, optional
        The maximum number of allowed rows. The default is None.
    delimiter : str, optional
        Delimiter to use between each element. The default is ", ".
    """
    formatted = [str(el) for el in elements]
    lines = []
    current_line = []
    current_width = 0

    for element in formatted:
        if current_width + len(element) + len(delimiter) > OPTIONS["display_width"] - col_width:
            lines.append(delimiter.join(current_line))
            current_line = [' ' * col_width + element]
            current_width = col_width + len(element)
        else:
            current_line.append(element)
            current_width += len(element) + len(delimiter)

    if current_line:
        lines.append(delimiter.join(current_line))

    if max_rows is not None and len(lines) > max_rows:
        half = max_rows // 2
        return '\n'.join(lines[:half] + ['...'] + lines[-half:])
    else:
        return '\n'.join(lines)

def limit_lines(string: str, *, limit: int):
    """
    If the string is more lines than the limit,
    this returns the middle lines replaced by an ellipsis
    """
    lines = string.splitlines()
    if len(lines) <= limit:
        return string
    else:
        half = (limit - 1) // 2
        return '\n'.join(lines[:half] + ['...'] + lines[-half:])

def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    if hasattr(array, 'name') and array.name is not None:
        return f'[{array.name}]'
    if is_duck_array(array):
        return f'[{type(array).__name__}]'
    else:
        return inline_variable_array_repr(array, OPTIONS["display_width"])

def dims_and_coords_repr(ds) -> str:
    """Partial Dataset repr for use inside DataTree inheritance errors."""
    dims = ', '.join(f'{k}: {v}' for k, v in ds.dims.items())
    coords = ', '.join(ds.coords.keys())
    return f"Dimensions: ({dims})\nCoordinates:\n  {coords}"
diff_data_vars_repr = functools.partial(_diff_mapping_repr, title='Data variables', summarizer=summarize_variable)
diff_attrs_repr = functools.partial(_diff_mapping_repr, title='Attributes', summarizer=summarize_attr)

def diff_treestructure(a: DataTree, b: DataTree, require_names_equal: bool) -> str:
    """
    Return a summary of why two trees are not isomorphic.
    If they are isomorphic return an empty string.
    """
    if require_names_equal and a.name != b.name:
        return f"Node names differ: {a.name} != {b.name}"
    
    a_children = set(a.children)
    b_children = set(b.children)
    
    if a_children != b_children:
        only_in_a = a_children - b_children
        only_in_b = b_children - a_children
        diff = []
        if only_in_a:
            diff.append(f"Only in a: {', '.join(only_in_a)}")
        if only_in_b:
            diff.append(f"Only in b: {', '.join(only_in_b)}")
        return '; '.join(diff)
    
    for child in a_children:
        child_diff = diff_treestructure(a[child], b[child], require_names_equal)
        if child_diff:
            return f"In child '{child}': {child_diff}"
    
    return ""

def diff_nodewise_summary(a: DataTree, b: DataTree, compat):
    """Iterates over all corresponding nodes, recording differences between data at each location."""
    summary = []
    for node_a, node_b in zip(LevelOrderIter(a), LevelOrderIter(b)):
        path = node_a.path
        diff = node_a.ds.diff(node_b.ds, compat=compat)
        if diff.dims or diff.data_vars or diff.attrs:
            summary.append(f"At {path}:")
            if diff.dims:
                summary.append(f"  Dimensions: {diff.dims}")
            if diff.data_vars:
                summary.append(f"  Data variables: {list(diff.data_vars)}")
            if diff.attrs:
                summary.append(f"  Attributes: {list(diff.attrs)}")
    return '\n'.join(summary)

def _single_node_repr(node: DataTree) -> str:
    """Information about this node, not including its relationships to other nodes."""
    lines = [
        f"<xarray.DataTree '{node.name or ''}'>",
        dims_and_coords_repr(node.ds),
        f"Data variables: {', '.join(node.ds.data_vars)}",
        f"Attributes: {', '.join(node.ds.attrs)}",
    ]
    return '\n'.join(lines)

def datatree_repr(dt: DataTree):
    """A printable representation of the structure of this entire tree."""
    renderer = RenderDataTree(dt)
    return '\n'.join([
        f"<xarray.DataTree>",
        f"Root: {dt.name or ''}",
        renderer.render(),
    ])

def render_human_readable_nbytes(nbytes: int, /, *, attempt_constant_width: bool=False) -> str:
    """Renders simple human-readable byte count representation

    This is only a quick representation that should not be relied upon for precise needs.

    To get the exact byte count, please use the ``nbytes`` attribute directly.

    Parameters
    ----------
    nbytes
        Byte count
    attempt_constant_width
        For reasonable nbytes sizes, tries to render a fixed-width representation.

    Returns
    -------
        Human-readable representation of the byte count
    """
    if nbytes == 0:
        return '0B'
    
    i = int(math.floor(math.log(nbytes, 1024)))
    p = math.pow(1024, i)
    s = round(nbytes / p, 2)
    
    if attempt_constant_width and s < 100:
        return f'{s:5.2f}{UNITS[i]}'
    else:
        return f'{s:.2f}{UNITS[i]}'
