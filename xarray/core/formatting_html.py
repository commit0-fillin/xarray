from __future__ import annotations
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from typing import TYPE_CHECKING
from xarray.core.formatting import inline_index_repr, inline_variable_array_repr, short_data_repr
from xarray.core.options import _get_boolean_with_default
STATIC_FILES = (('xarray.static.html', 'icons-svg-inline.html'), ('xarray.static.css', 'style.css'))
if TYPE_CHECKING:
    from xarray.core.datatree import DataTree

@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    static_files = {}
    for package, filename in STATIC_FILES:
        with files(package).joinpath(filename).open("r") as f:
            static_files[filename] = f.read()
    return static_files

def short_data_repr_html(array) -> str:
    """Format "data" for DataArray and Variable."""
    if hasattr(array, 'name') and array.name is not None:
        return escape(f'[{array.name}]')
    if is_duck_array(array):
        return escape(f'[{type(array).__name__}]')
    else:
        return escape(inline_variable_array_repr(array, OPTIONS["display_width"]))
coord_section = partial(_mapping_section, name='Coordinates', details_func=summarize_coords, max_items_collapse=25, expand_option_name='display_expand_coords')
datavar_section = partial(_mapping_section, name='Data variables', details_func=summarize_vars, max_items_collapse=15, expand_option_name='display_expand_data_vars')
index_section = partial(_mapping_section, name='Indexes', details_func=summarize_indexes, max_items_collapse=0, expand_option_name='display_expand_indexes')
attr_section = partial(_mapping_section, name='Attributes', details_func=summarize_attrs, max_items_collapse=10, expand_option_name='display_expand_attrs')

def _obj_repr(obj, header_components, sections):
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.
    """
    static_files = _load_static_files()
    icons = static_files['icons-svg-inline.html']
    css = static_files['style.css']

    header = f'<div class="xr-header">{" ".join(header_components)}</div>'
    sections_html = "".join(sections)

    repr_html = f'<pre class="xr-text-repr-fallback">{escape(repr(obj))}</pre>'
    repr_html += f'<div id="%%s" class="xr-wrap">{icons}{header}{sections_html}</div>' % uuid.uuid4()

    return f'<style>{css}</style>' + repr_html
children_section = partial(_mapping_section, name='Groups', details_func=summarize_datatree_children, max_items_collapse=1, expand_option_name='display_expand_groups')

def _wrap_datatree_repr(r: str, end: bool=False) -> str:
    """
    Wrap HTML representation with a tee to the left of it.

    Enclosing HTML tag is a <div> with :code:`display: inline-grid` style.

    Turns:
    [    title    ]
    |   details   |
    |_____________|

    into (A):
    |─ [    title    ]
    |  |   details   |
    |  |_____________|

    or (B):
    └─ [    title    ]
       |   details   |
       |_____________|

    Parameters
    ----------
    r: str
        HTML representation to wrap.
    end: bool
        Specify if the line on the left should continue or end.

        Default is True.

    Returns
    -------
    str
        Wrapped HTML representation.

        Tee color is set to the variable :code:`--xr-border-color`.
    """
    tee = "└" if end else "├"
    return f"""
    <div style="display: inline-grid; grid-template-columns: auto 1fr; grid-column-gap: 0.5em;">
        <div style="color: var(--xr-border-color);">{tee}─</div>
        <div>{r}</div>
    </div>
    """
