from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
nc_time_axis_available = module_available('nc_time_axis')
try:
    import cftime
except ImportError:
    cftime = None
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    from numpy.typing import ArrayLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import AspectOptions, ScaleOptions
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt: Any = None
ROBUST_PERCENTILE = 2.0
_MARKERSIZE_RANGE = (18.0, 36.0, 72.0)
_LINEWIDTH_RANGE = (1.5, 1.5, 6.0)

def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    if not filled:
        # non-filled contour plots
        extend = "max"

    if extend == "both":
        ext_n = 2
    elif extend in ["min", "max"]:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = colors.from_levels_and_colors(levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, "name", cmap)

    return new_cmap, cnorm

def _determine_cmap_params(plot_data, vmin=None, vmax=None, cmap=None, center=None, robust=False, extend=None, levels=None, filled=True, norm=None, _is_facetgrid=False):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Parameters
    ----------
    plot_data : Numpy array
        Doesn't handle xarray objects

    Returns
    -------
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize

    calc_data = np.ravel(plot_data[np.isfinite(plot_data)])

    if vmin is None:
        vmin = np.min(calc_data)
    if vmax is None:
        vmax = np.max(calc_data)

    # Choose default extend based on vmin, vmax
    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    # Handle centers
    if center is not None:
        vmin, vmax = _center_to_edge(center, vmin, vmax)
        center = None

    # Handle robust
    if robust:
        vmin, vmax = _robust_minmax(calc_data, ROBUST_PERCENTILE)

    # Handle norm
    if norm is not None:
        vmin = None
        vmax = None

    # Set cmap if none provided
    if cmap is None:
        cmap = OPTIONS["cmap_divergent"] if center is not None else OPTIONS["cmap_sequential"]

    # If cmap is a string, get the colormap from matplotlib
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Handle discrete levels
    if levels is not None:
        levels = _determine_levels(levels, calc_data, vmin, vmax, robust)
        cmap, norm = _build_discrete_cmap(cmap, levels, extend, filled)
    elif isinstance(norm, BoundaryNorm):
        levels = norm.boundaries
    else:
        levels = None

    vmin, vmax = _ensure_finite(vmin, vmax)

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, extend=extend,
                levels=levels, norm=norm, center=center)

def _infer_xy_labels_3d(darray: DataArray | Dataset, x: Hashable | None, y: Hashable | None, rgb: Hashable | None) -> tuple[Hashable, Hashable]:
    """
    Determine x and y labels for showing RGB images.

    Attempts to infer which dimension is RGB/RGBA by size and order of dims.

    """
    dims = list(darray.dims)
    if x is None and y is None:
        assert len(dims) == 3, 'DataArray must be 3D'
        # Assume that the RGB/RGBA axis is the last one
        x, y = dims[0], dims[1]
    elif x is None:
        assert y in dims, f"y {y} must be a dimension of the DataArray"
        x = [d for d in dims if d != y][0]
    elif y is None:
        assert x in dims, f"x {x} must be a dimension of the DataArray"
        y = [d for d in dims if d != x][0]
    else:
        assert x in dims and y in dims, "x and y must be dimensions of the DataArray"

    return x, y

def _infer_xy_labels(darray: DataArray | Dataset, x: Hashable | None, y: Hashable | None, imshow: bool=False, rgb: Hashable | None=None) -> tuple[Hashable, Hashable]:
    """
    Determine x and y labels. For use in _plot2d

    darray must be a 2 dimensional data array, or 3d for imshow only.
    """
    if imshow and darray.ndim == 3:
        return _infer_xy_labels_3d(darray, x, y, rgb)

    if x is None and y is None:
        if darray.ndim != 2:
            raise ValueError('DataArray must be 2D')
        y, x = darray.dims
    elif x is None:
        if y not in darray.dims:
            raise ValueError(f'y {y} must be a dimension of the DataArray')
        x = darray.dims[0] if y == darray.dims[1] else darray.dims[1]
    elif y is None:
        if x not in darray.dims:
            raise ValueError(f'x {x} must be a dimension of the DataArray')
        y = darray.dims[0] if x == darray.dims[1] else darray.dims[1]
    elif any(k not in darray.dims for k in (x, y)):
        raise ValueError('x and y must be dimensions of the DataArray')

    return x, y

def _assert_valid_xy(darray: DataArray | Dataset, xy: Hashable | None, name: str) -> None:
    """
    make sure x and y passed to plotting functions are valid
    """
    if xy is not None:
        if isinstance(darray, DataArray):
            valid_dims = set(darray.dims)
        else:
            valid_dims = set(darray.coords)
        if xy not in valid_dims:
            raise ValueError(f"{name} must be one of {valid_dims}")

def _get_units_from_attrs(da: DataArray) -> str:
    """Extracts and formats the unit/units from a attributes."""
    units = da.attrs.get("units", da.attrs.get("unit", None))
    if units is not None:
        return f" [{units}]"
    else:
        return ""

def label_from_attrs(da: DataArray | None, extra: str='') -> str:
    """Makes informative labels if variable metadata (attrs) follows
    CF conventions."""
    if da is None:
        return ''

    name = da.name
    long_name = da.attrs.get('long_name')
    standard_name = da.attrs.get('standard_name')
    units = _get_units_from_attrs(da)

    if long_name:
        label = long_name
    elif standard_name:
        label = standard_name
    elif name:
        label = name
    else:
        label = ''

    if units:
        label += units

    if extra:
        label += f' {extra}'

    return label

def _interval_to_mid_points(array: Iterable[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
    with the Intervals' mid points.
    """
    return np.array([interval.mid for interval in array])

def _interval_to_bound_points(array: Sequence[pd.Interval]) -> np.ndarray:
    """
    Helper function which returns an array
    with the Intervals' boundaries.
    """
    bounds = [interval.left for interval in array]
    bounds.append(array[-1].right)
    return np.array(bounds)

def _interval_to_double_bound_points(xarray: Iterable[pd.Interval], yarray: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to deal with a xarray consisting of pd.Intervals. Each
    interval is replaced with both boundaries. I.e. the length of xarray
    doubles. yarray is modified so it matches the new shape of xarray.
    """
    xarray2 = []
    yarray2 = []
    for x, y in zip(xarray, yarray):
        xarray2.extend([x.left, x.right])
        yarray2.extend([y, y])
    return np.array(xarray2), np.array(yarray2)

def _resolve_intervals_1dplot(xval: np.ndarray, yval: np.ndarray, kwargs: dict) -> tuple[np.ndarray, np.ndarray, str, str, dict]:
    """
    Helper function to replace the values of x and/or y coordinate arrays
    containing pd.Interval with their mid-points or - for step plots - double
    points which double the length.
    """
    x_is_interval = pd.api.types.is_interval_dtype(xval)
    y_is_interval = pd.api.types.is_interval_dtype(yval)

    if kwargs.get('drawstyle', '').startswith('steps-'):
        if x_is_interval:
            xval, yval = _interval_to_double_bound_points(xval, yval)
        x_suffix = '_bounds'
        y_suffix = ''
    else:
        if x_is_interval:
            xval = _interval_to_mid_points(xval)
        if y_is_interval:
            yval = _interval_to_mid_points(yval)
        x_suffix = '_center'
        y_suffix = '_center'

    return xval, yval, x_suffix, y_suffix, kwargs

def _resolve_intervals_2dplot(val, func_name):
    """
    Helper function to replace the values of a coordinate array containing
    pd.Interval with their mid-points or - for pcolormesh - boundaries which
    increases length by 1.
    """
    if pd.api.types.is_interval_dtype(val):
        if func_name == 'pcolormesh':
            val = _interval_to_bound_points(val)
            suffix = '_bounds'
        else:
            val = _interval_to_mid_points(val)
            suffix = '_center'
    else:
        suffix = ''
    return val, suffix

def _valid_other_type(x: ArrayLike, types: type[object] | tuple[type[object], ...]) -> bool:
    """
    Do all elements of x have a type from types?
    """
    return all(isinstance(el, types) for el in np.ravel(x))

def _valid_numpy_subdtype(x, numpy_types):
    """
    Is any dtype from numpy_types superior to the dtype of x?
    """
    return any(np.issubdtype(x.dtype, numpy_type) for numpy_type in numpy_types)

def _ensure_plottable(*args) -> None:
    """
    Raise exception if there is anything in args that can't be plotted on an
    axis by matplotlib.
    """
    for arg in args:
        if not (
            _valid_numpy_subdtype(arg, [np.floating, np.integer, np.timedelta64, np.datetime64])
            or _valid_other_type(arg, (datetime, date, pd.Timestamp))
        ):
            raise TypeError(
                f"Plotting requires coordinates to be numeric, datetime, or timedelta. "
                f"Instead, we received: {arg.dtype}"
            )

def _update_axes(ax: Axes, xincrease: bool | None, yincrease: bool | None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None) -> None:
    """
    Update axes with provided parameters
    """
    if xincrease is not None:
        ax.invert_xaxis() if not xincrease else None
    if yincrease is not None:
        ax.invert_yaxis() if not yincrease else None
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def _is_monotonic(coord, axis=0):
    """
    >>> _is_monotonic(np.array([0, 1, 2]))
    np.True_
    >>> _is_monotonic(np.array([2, 1, 0]))
    np.True_
    >>> _is_monotonic(np.array([0, 2, 1]))
    np.False_
    """
    if coord.shape[axis] < 2:
        return np.True_
    else:
        n = coord.shape[axis]
        delta_pos = (coord.take(np.arange(1, n), axis=axis) >=
                     coord.take(np.arange(0, n-1), axis=axis))
        delta_neg = (coord.take(np.arange(1, n), axis=axis) <=
                     coord.take(np.arange(0, n-1), axis=axis))
        return np.all(delta_pos) or np.all(delta_neg)

def _infer_interval_breaks(coord, axis=0, scale=None, check_monotonic=False):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    >>> _infer_interval_breaks([[0, 1], [3, 4]], axis=1)
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    >>> _infer_interval_breaks(np.logspace(-2, 2, 5), scale="log")
    array([3.16227766e-03, 3.16227766e-02, 3.16227766e-01, 3.16227766e+00,
           3.16227766e+01, 3.16227766e+02])
    """
    coord = np.asarray(coord)
    if check_monotonic and not _is_monotonic(coord, axis=axis):
        raise ValueError("The coordinate is not monotonic along the specified axis")

    if np.issubdtype(coord.dtype, np.datetime64):
        deltas = np.diff(coord, axis=axis)
        if axis != 0:
            deltas = deltas.T
        median = np.median(deltas)
        return np.concatenate([coord.take([0], axis=axis) - median / 2,
                               coord[..., :-1] + deltas / 2,
                               coord.take([-1], axis=axis) + median / 2], axis=axis)

    if scale == "log":
        return np.power(10, _infer_interval_breaks(np.log10(coord), axis=axis))

    deltas = np.diff(coord, axis=axis)
    if axis != 0:
        deltas = deltas.T
    deltas = np.concatenate([[deltas[0]], deltas, [deltas[-1]]])
    coord_breaks = coord[..., :-1] + deltas / 2
    return np.concatenate([coord.take([0], axis=axis) - deltas[0] / 2,
                           coord_breaks,
                           coord.take([-1], axis=axis) + deltas[-1] / 2], axis=axis)

def _process_cmap_cbar_kwargs(func, data, cmap=None, colors=None, cbar_kwargs: Iterable[tuple[str, Any]] | Mapping[str, Any] | None=None, levels=None, _is_facetgrid=False, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parameters
    ----------
    func : plotting function
    data : ndarray,
        Data values

    Returns
    -------
    cmap_params : dict
    cbar_kwargs : dict
    """
    cmap_params = {}
    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

    if colors is not None:
        if isinstance(colors, str):
            cmap_params['cmap'] = colors
        else:
            cmap_params['colors'] = colors

    if cmap is not None:
        cmap_params['cmap'] = cmap

    if 'contour' in func.__name__:
        # contour functions don't accept 'norm' kwarg, so we'll have to
        # handle the norm separately later
        if kwargs.get('norm') is not None:
            cmap_params['norm'] = kwargs.pop('norm')

        # for contour plots, we need to default vmin/vmax to the range of
        # the data, not just the positive range
        cmap_params.update(_determine_cmap_params(data, vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'),
                                                  cmap=cmap, center=kwargs.get('center'), robust=kwargs.get('robust'),
                                                  extend=kwargs.get('extend'), levels=levels, filled=kwargs.get('filled', True),
                                                  norm=kwargs.get('norm'), _is_facetgrid=_is_facetgrid))
    else:
        cmap_params.update(_determine_cmap_params(data, vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'),
                                                  cmap=cmap, center=kwargs.get('center'), robust=kwargs.get('robust'),
                                                  extend=kwargs.get('extend'), levels=levels, filled=True,
                                                  norm=kwargs.get('norm'), _is_facetgrid=_is_facetgrid))
    return cmap_params, cbar_kwargs

def legend_elements(self, prop='colors', num='auto', fmt=None, func=lambda x: x, **kwargs):
    """
    Create legend handles and labels for a PathCollection.

    Each legend handle is a `.Line2D` representing the Path that was drawn,
    and each label is a string what each Path represents.

    This is useful for obtaining a legend for a `~.Axes.scatter` plot;
    e.g.::

        scatter = plt.scatter([1, 2, 3],  [4, 5, 6],  c=[7, 2, 3])
        plt.legend(*scatter.legend_elements())

    creates three legend elements, one for each color with the numerical
    values passed to *c* as the labels.

    Also see the :ref:`automatedlegendcreation` example.


    Parameters
    ----------
    prop : {"colors", "sizes"}, default: "colors"
        If "colors", the legend handles will show the different colors of
        the collection. If "sizes", the legend will show the different
        sizes. To set both, use *kwargs* to directly edit the `.Line2D`
        properties.
    num : int, None, "auto" (default), array-like, or `~.ticker.Locator`
        Target number of elements to create.
        If None, use all unique elements of the mappable array. If an
        integer, target to use *num* elements in the normed range.
        If *"auto"*, try to determine which option better suits the nature
        of the data.
        The number of created elements may slightly deviate from *num* due
        to a `~.ticker.Locator` being used to find useful locations.
        If a list or array, use exactly those elements for the legend.
        Finally, a `~.ticker.Locator` can be provided.
    fmt : str, `~matplotlib.ticker.Formatter`, or None (default)
        The format or formatter to use for the labels. If a string must be
        a valid input for a `~.StrMethodFormatter`. If None (the default),
        use a `~.ScalarFormatter`.
    func : function, default: ``lambda x: x``
        Function to calculate the labels.  Often the size (or color)
        argument to `~.Axes.scatter` will have been pre-processed by the
        user using a function ``s = f(x)`` to make the markers visible;
        e.g. ``size = np.log10(x)``.  Providing the inverse of this
        function here allows that pre-processing to be inverted, so that
        the legend labels have the correct values; e.g. ``func = lambda
        x: 10**x``.
    **kwargs
        Allowed keyword arguments are *color* and *size*. E.g. it may be
        useful to set the color of the markers if *prop="sizes"* is used;
        similarly to set the size of the markers if *prop="colors"* is
        used. Any further parameters are passed onto the `.Line2D`
        instance. This may be useful to e.g. specify a different
        *markeredgecolor* or *alpha* for the legend handles.

    Returns
    -------
    handles : list of `.Line2D`
        Visual representation of each element of the legend.
    labels : list of str
        The string labels for elements of the legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    handles = []
    labels = []

    if prop == 'colors':
        array = self.get_array()
        if array is None:
            raise ValueError('Collection must have a colormap or array for legend_elements')
        
        cmap = self.get_cmap()
        norm = self.norm if self.norm is not None else Normalize()
        
        if num == 'auto':
            num = min(len(array), 9)
        elif num is None:
            num = len(array)
        
        if isinstance(num, (int, float)):
            boundaries = np.linspace(norm.vmin, norm.vmax, num + 1)
        elif isinstance(num, (list, np.ndarray)):
            boundaries = np.asarray(num)
        else:
            raise ValueError('Invalid value for num')
        
        for i, boundary in enumerate(boundaries[:-1]):
            color = cmap(norm((boundary + boundaries[i+1]) / 2))
            handles.append(Line2D([0], [0], color=color, **kwargs))
            label = func(boundary)
            if fmt is None:
                formatter = ScalarFormatter()
            elif isinstance(fmt, str):
                formatter = StrMethodFormatter(fmt)
            else:
                formatter = fmt
            labels.append(formatter(label))
    
    elif prop == 'sizes':
        sizes = self.get_sizes()
        if sizes is None:
            raise ValueError('Collection must have sizes for legend_elements')
        
        if num == 'auto':
            num = min(len(sizes), 5)
        elif num is None:
            num = len(sizes)
        
        if isinstance(num, (int, float)):
            size_boundaries = np.linspace(min(sizes), max(sizes), num + 1)
        elif isinstance(num, (list, np.ndarray)):
            size_boundaries = np.asarray(num)
        else:
            raise ValueError('Invalid value for num')
        
        for i, size in enumerate(size_boundaries[:-1]):
            handles.append(Line2D([0], [0], marker='o', markersize=np.sqrt(size), **kwargs))
            label = func(size)
            if fmt is None:
                formatter = ScalarFormatter()
            elif isinstance(fmt, str):
                formatter = StrMethodFormatter(fmt)
            else:
                formatter = fmt
            labels.append(formatter(label))
    
    else:
        raise ValueError("prop must be 'colors' or 'sizes'")

    return handles, labels

def _legend_add_subtitle(handles, labels, text):
    """Add a subtitle to legend handles."""
    from matplotlib.lines import Line2D
    
    subtitle = Line2D([], [], color='none', label=text)
    handles.append(subtitle)
    labels.append(text)
    return handles, labels

def _adjust_legend_subtitles(legend):
    """Make invisible-handle "subtitles" entries look more like titles."""
    # Legend title not in rcParams until 3.0
    title_fontsize = plt.rcParams.get('legend.title_fontsize', None)
    for handle, text in zip(legend.legendHandles, legend.texts):
        if not handle.get_visible():
            if title_fontsize is not None:
                text.set_fontsize(title_fontsize)
            else:
                text.set_fontweight('bold')

class _Normalize(Sequence):
    """
    Normalize numerical or categorical values to numerical values.

    The class includes helper methods that simplifies transforming to
    and from normalized values.

    Parameters
    ----------
    data : DataArray
        DataArray to normalize.
    width : Sequence of three numbers, optional
        Normalize the data to these (min, default, max) values.
        The default is None.
    """
    _data: DataArray | None
    _data_unique: np.ndarray
    _data_unique_index: np.ndarray
    _data_unique_inverse: np.ndarray
    _data_is_numeric: bool
    _width: tuple[float, float, float] | None
    __slots__ = ('_data', '_data_unique', '_data_unique_index', '_data_unique_inverse', '_data_is_numeric', '_width')

    def __init__(self, data: DataArray | None, width: tuple[float, float, float] | None=None, _is_facetgrid: bool=False) -> None:
        self._data = data
        self._width = width if not _is_facetgrid else None
        pint_array_type = DuckArrayModule('pint').type
        to_unique = data.to_numpy() if isinstance(data if data is None else data.data, pint_array_type) else data
        data_unique, data_unique_inverse = np.unique(to_unique, return_inverse=True)
        self._data_unique = data_unique
        self._data_unique_index = np.arange(0, data_unique.size)
        self._data_unique_inverse = data_unique_inverse
        self._data_is_numeric = False if data is None else _is_numeric(data)

    def __repr__(self) -> str:
        with np.printoptions(precision=4, suppress=True, threshold=5):
            return f'<_Normalize(data, width={self._width})>\n{self._data_unique} -> {self._values_unique}'

    def __len__(self) -> int:
        return len(self._data_unique)

    def __getitem__(self, key):
        return self._data_unique[key]

    @property
    def data_is_numeric(self) -> bool:
        """
        Check if data is numeric.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).data_is_numeric
        False

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a).data_is_numeric
        True

        >>> # TODO: Datetime should be numeric right?
        >>> a = xr.DataArray(pd.date_range("2000-1-1", periods=4))
        >>> _Normalize(a).data_is_numeric
        False

        # TODO: Timedelta should be numeric right?
        >>> a = xr.DataArray(pd.timedelta_range("-1D", periods=4, freq="D"))
        >>> _Normalize(a).data_is_numeric
        True
        """
        return np.issubdtype(self._data_unique.dtype, np.number) or np.issubdtype(self._data_unique.dtype, np.timedelta64)

    def _calc_widths(self, y: np.ndarray | DataArray) -> np.ndarray | DataArray:
        """
        Normalize the values so they're in between self._width.
        """
        if self._width is None:
            return y
        
        min_width, default_width, max_width = self._width
        y_min, y_max = np.min(y), np.max(y)
        
        if y_min == y_max:
            return np.full_like(y, default_width)
        
        normalized = (y - y_min) / (y_max - y_min)
        return min_width + normalized * (max_width - min_width)

    def _indexes_centered(self, x: np.ndarray | DataArray) -> np.ndarray | DataArray:
        """
        Offset indexes to make sure being in the center of self.levels.
        ["a", "b", "c"] -> [1, 3, 5]
        """
        if self.data_is_numeric:
            return x
        
        indexes = np.arange(len(self._data_unique))
        return indexes * 2 + 1

    @property
    def values(self) -> DataArray | None:
        """
        Return a normalized number array for the unique levels.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).values
        <xarray.DataArray (dim_0: 5)> Size: 40B
        array([3, 1, 1, 3, 5])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 5)> Size: 40B
        array([45., 18., 18., 45., 72.])
        Dimensions without coordinates: dim_0

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([0.5, 0. , 0. , 0.5, 2. , 3. ])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([27., 18., 18., 27., 54., 72.])
        Dimensions without coordinates: dim_0

        >>> _Normalize(a * 0, width=(18, 36, 72)).values
        <xarray.DataArray (dim_0: 6)> Size: 48B
        array([36., 36., 36., 36., 36., 36.])
        Dimensions without coordinates: dim_0

        """
        if self._data is None:
            return None
        
        if self.data_is_numeric:
            values = self._data_unique
        else:
            values = self._indexes_centered(self._data_unique)
        
        values = self._calc_widths(values)
        
        return xr.DataArray(
            values[self._data_unique_inverse],
            dims=self._data.dims,
            coords=self._data.coords,
        )

    @property
    def _values_unique(self) -> np.ndarray | None:
        """
        Return unique values.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a)._values_unique
        array([1, 3, 5])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 45., 72.])

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a)._values_unique
        array([0. , 0.5, 2. , 3. ])

        >>> _Normalize(a, width=(18, 36, 72))._values_unique
        array([18., 27., 54., 72.])
        """
        if self._data is None:
            return None
        
        if self.data_is_numeric:
            values = self._data_unique
        else:
            values = self._indexes_centered(self._data_unique_index)
        
        return self._calc_widths(values)

    @property
    def ticks(self) -> np.ndarray | None:
        """
        Return ticks for plt.colorbar if the data is not numeric.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).ticks
        array([1, 3, 5])
        """
        if self.data_is_numeric:
            return None
        return self._indexes_centered(self._data_unique_index)

    @property
    def levels(self) -> np.ndarray:
        """
        Return discrete levels that will evenly bound self.values.
        ["a", "b", "c"] -> [0, 2, 4, 6]

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).levels
        array([0, 2, 4, 6])
        """
        if self.data_is_numeric:
            return np.concatenate([[-np.inf], self._data_unique, [np.inf]])
        else:
            indexes = np.arange(len(self._data_unique) + 1)
            return indexes * 2

    @property
    def format(self) -> FuncFormatter:
        """
        Return a FuncFormatter that maps self.values elements back to
        the original value as a string. Useful with plt.colorbar.

        Examples
        --------
        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> aa = _Normalize(a, width=(0, 0.5, 1))
        >>> aa._lookup
        0.000000    0.0
        0.166667    0.5
        0.666667    2.0
        1.000000    3.0
        dtype: float64
        >>> aa.format(1)
        '3.0'
        """
        from matplotlib.ticker import FuncFormatter

        def formatter(x, pos):
            if self.data_is_numeric:
                return f"{x:.4g}"
            else:
                index = np.argmin(np.abs(self._values_unique - x))
                return str(self._data_unique[index])

        return FuncFormatter(formatter)

    @property
    def func(self) -> Callable[[Any, None | Any], Any]:
        """
        Return a lambda function that maps self.values elements back to
        the original value as a numpy array. Useful with ax.legend_elements.

        Examples
        --------
        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> aa = _Normalize(a, width=(0, 0.5, 1))
        >>> aa._lookup
        0.000000    0.0
        0.166667    0.5
        0.666667    2.0
        1.000000    3.0
        dtype: float64
        >>> aa.func([0.16, 1])
        array([0.5, 3. ])
        """
        def func(x, pos=None):
            if self.data_is_numeric:
                return x
            else:
                indices = np.argmin(np.abs(self._values_unique[:, np.newaxis] - x), axis=0)
                return self._data_unique[indices]

        return func

def _guess_coords_to_plot(darray: DataArray, coords_to_plot: MutableMapping[str, Hashable | None], kwargs: dict, default_guess: tuple[str, ...]=('x',), ignore_guess_kwargs: tuple[tuple[str, ...], ...]=((),)) -> MutableMapping[str, Hashable]:
    """
    Guess what coords to plot if some of the values in coords_to_plot are None which
    happens when the user has not defined all available ways of visualizing
    the data.

    Parameters
    ----------
    darray : DataArray
        The DataArray to check for available coords.
    coords_to_plot : MutableMapping[str, Hashable]
        Coords defined by the user to plot.
    kwargs : dict
        Extra kwargs that will be sent to matplotlib.
    default_guess : Iterable[str], optional
        Default values and order to retrieve dims if values in dims_plot is
        missing, default: ("x", "hue", "size").
    ignore_guess_kwargs : tuple[tuple[str, ...], ...]
        Matplotlib arguments to ignore.

    Examples
    --------
    >>> ds = xr.tutorial.scatter_example_dataset(seed=42)
    >>> # Only guess x by default:
    >>> xr.plot.utils._guess_coords_to_plot(
    ...     ds.A,
    ...     coords_to_plot={"x": None, "z": None, "hue": None, "size": None},
    ...     kwargs={},
    ... )
    {'x': 'x', 'z': None, 'hue': None, 'size': None}

    >>> # Guess all plot dims with other default values:
    >>> xr.plot.utils._guess_coords_to_plot(
    ...     ds.A,
    ...     coords_to_plot={"x": None, "z": None, "hue": None, "size": None},
    ...     kwargs={},
    ...     default_guess=("x", "hue", "size"),
    ...     ignore_guess_kwargs=((), ("c", "color"), ("s",)),
    ... )
    {'x': 'x', 'z': None, 'hue': 'y', 'size': 'z'}

    >>> # Don't guess ´size´, since the matplotlib kwarg ´s´ has been defined:
    >>> xr.plot.utils._guess_coords_to_plot(
    ...     ds.A,
    ...     coords_to_plot={"x": None, "z": None, "hue": None, "size": None},
    ...     kwargs={"s": 5},
    ...     default_guess=("x", "hue", "size"),
    ...     ignore_guess_kwargs=((), ("c", "color"), ("s",)),
    ... )
    {'x': 'x', 'z': None, 'hue': 'y', 'size': None}

    >>> # Prioritize ´size´ over ´s´:
    >>> xr.plot.utils._guess_coords_to_plot(
    ...     ds.A,
    ...     coords_to_plot={"x": None, "z": None, "hue": None, "size": "x"},
    ...     kwargs={"s": 5},
    ...     default_guess=("x", "hue", "size"),
    ...     ignore_guess_kwargs=((), ("c", "color"), ("s",)),
    ... )
    {'x': 'y', 'z': None, 'hue': 'z', 'size': 'x'}
    """
    available_coords = set(darray.coords)
    guessed_coords = {}

    for coord, value in coords_to_plot.items():
        if value is not None:
            guessed_coords[coord] = value
        elif coord in default_guess:
            for guess in default_guess:
                if guess not in guessed_coords.values() and guess in available_coords:
                    if any(guess in ignore_kwargs for ignore_kwargs in ignore_guess_kwargs):
                        if not any(kwarg in kwargs for kwarg in ignore_kwargs):
                            guessed_coords[coord] = guess
                            break
                    else:
                        guessed_coords[coord] = guess
                        break
        else:
            guessed_coords[coord] = None

    return guessed_coords

def _set_concise_date(ax: Axes, axis: Literal['x', 'y', 'z']='x') -> None: 
    """
    Use ConciseDateFormatter which is meant to improve the
    strings chosen for the ticklabels, and to minimize the
    strings used in those tick labels as much as possible.

    https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html

    Parameters
    ----------
    ax : Axes
        Figure axes.
    axis : Literal["x", "y", "z"], optional
        Which axis to make concise. The default is "x".
    """
    import matplotlib.dates as mdates
    
    if axis == 'x':
        axis = ax.xaxis
    elif axis == 'y':
        axis = ax.yaxis
    elif axis == 'z':
        axis = ax.zaxis
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    locator = axis.get_major_locator()
    formatter = mdates.ConciseDateFormatter(locator)
    axis.set_major_formatter(formatter)
