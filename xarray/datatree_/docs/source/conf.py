import inspect
import os
import sys
import sphinx_autosummary_accessors
import datatree
cwd = os.getcwd()
parent = os.path.dirname(cwd)
sys.path.insert(0, parent)
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.linkcode', 'sphinx.ext.autosummary', 'sphinx.ext.intersphinx', 'sphinx.ext.extlinks', 'sphinx.ext.napoleon', 'sphinx_copybutton', 'sphinxext.opengraph', 'sphinx_autosummary_accessors', 'IPython.sphinxext.ipython_console_highlighting', 'IPython.sphinxext.ipython_directive', 'nbsphinx', 'sphinxcontrib.srclinks']
extlinks = {'issue': ('https://github.com/xarray-contrib/datatree/issues/%s', 'GH#%s'), 'pull': ('https://github.com/xarray-contrib/datatree/pull/%s', 'GH#%s')}
templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {'sequence': ':term:`sequence`', 'iterable': ':term:`iterable`', 'callable': ':py:func:`callable`', 'dict_like': ':term:`dict-like <mapping>`', 'dict-like': ':term:`dict-like <mapping>`', 'path-like': ':term:`path-like <path-like object>`', 'mapping': ':term:`mapping`', 'file-like': ':term:`file-like <file-like object>`', 'MutableMapping': '~collections.abc.MutableMapping', 'sys.stdout': ':obj:`sys.stdout`', 'timedelta': '~datetime.timedelta', 'string': ':class:`string <str>`', 'array_like': ':term:`array_like`', 'array-like': ':term:`array-like <array_like>`', 'scalar': ':term:`scalar`', 'array': ':term:`array`', 'hashable': ':term:`hashable <name>`', 'color-like': ':py:func:`color-like <matplotlib.colors.is_color_like>`', 'matplotlib colormap name': ':doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`', 'matplotlib axes object': ':py:class:`matplotlib axes object <matplotlib.axes.Axes>`', 'colormap': ':py:class:`colormap <matplotlib.colors.Colormap>`', 'DataArray': '~xarray.DataArray', 'Dataset': '~xarray.Dataset', 'Variable': '~xarray.Variable', 'DatasetGroupBy': '~xarray.core.groupby.DatasetGroupBy', 'DataArrayGroupBy': '~xarray.core.groupby.DataArrayGroupBy', 'ndarray': '~numpy.ndarray', 'MaskedArray': '~numpy.ma.MaskedArray', 'dtype': '~numpy.dtype', 'ComplexWarning': '~numpy.ComplexWarning', 'Index': '~pandas.Index', 'MultiIndex': '~pandas.MultiIndex', 'CategoricalIndex': '~pandas.CategoricalIndex', 'TimedeltaIndex': '~pandas.TimedeltaIndex', 'DatetimeIndex': '~pandas.DatetimeIndex', 'Series': '~pandas.Series', 'DataFrame': '~pandas.DataFrame', 'Categorical': '~pandas.Categorical', 'Path': '~~pathlib.Path', 'pd.Index': '~pandas.Index', 'pd.NaT': '~pandas.NaT'}
source_suffix = '.rst'
master_doc = 'index'
project = 'Datatree'
copyright = '2021 onwards, Tom Nicholas and its Contributors'
author = 'Tom Nicholas'
html_show_sourcelink = True
srclink_project = 'https://github.com/xarray-contrib/datatree'
srclink_branch = 'main'
srclink_src_path = 'docs/source'
version = datatree.__version__
release = datatree.__version__
exclude_patterns = ['_build']
pygments_style = 'sphinx'
intersphinx_mapping = {'python': ('https://docs.python.org/3.8/', None), 'numpy': ('https://numpy.org/doc/stable', None), 'xarray': ('https://xarray.pydata.org/en/stable/', None)}
html_theme = 'sphinx_book_theme'
html_theme_options = {'repository_url': 'https://github.com/xarray-contrib/datatree', 'repository_branch': 'main', 'path_to_docs': 'docs/source', 'use_repository_button': True, 'use_issues_button': True, 'use_edit_page_button': True}
htmlhelp_basename = 'datatree_doc'
latex_elements: dict = {}
latex_documents = [('index', 'datatree.tex', 'Datatree Documentation', author, 'manual')]
man_pages = [('index', 'datatree', 'Datatree Documentation', [author], 1)]
texinfo_documents = [('index', 'datatree', 'Datatree Documentation', author, 'datatree', 'Tree-like hierarchical data structure for xarray.', 'Miscellaneous')]

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(datatree.__file__))

    return f"{srclink_project}/blob/{srclink_branch}/datatree/{fn}{linespec}"
