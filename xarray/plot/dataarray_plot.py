from __future__ import annotations
import functools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, overload
import numpy as np
import pandas as pd
from xarray.core.alignment import broadcast
from xarray.core.concat import concat
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import _LINEWIDTH_RANGE, _MARKERSIZE_RANGE, _add_colorbar, _add_legend, _assert_valid_xy, _determine_guide, _ensure_plottable, _guess_coords_to_plot, _infer_interval_breaks, _infer_xy_labels, _Normalize, _process_cmap_cbar_kwargs, _rescale_imshow_rgb, _resolve_intervals_1dplot, _resolve_intervals_2dplot, _set_concise_date, _update_axes, get_axis, label_from_attrs
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.container import BarContainer
    from matplotlib.contour import QuadContourSet
    from matplotlib.image import AxesImage
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
    from numpy.typing import ArrayLike
    from xarray.core.dataarray import DataArray
    from xarray.core.types import AspectOptions, ExtendOptions, HueStyleOptions, ScaleOptions, T_DataArray
    from xarray.plot.facetgrid import FacetGrid
_styles: dict[str, Any] = {'scatter.edgecolors': 'w'}

def _prepare_plot1d_data(darray: T_DataArray, coords_to_plot: MutableMapping[str, Hashable], plotfunc_name: str | None=None, _is_facetgrid: bool=False) -> dict[str, T_DataArray]:
    """
    Prepare data for usage with plt.scatter.

    Parameters
    ----------
    darray : T_DataArray
        Base DataArray.
    coords_to_plot : MutableMapping[str, Hashable]
        Coords that will be plotted.
    plotfunc_name : str | None
        Name of the plotting function that will be used.

    Returns
    -------
    plts : dict[str, T_DataArray]
        Dict of DataArrays that will be sent to matplotlib.

    Examples
    --------
    >>> # Make sure int coords are plotted:
    >>> a = xr.DataArray(
    ...     data=[1, 2],
    ...     coords={1: ("x", [0, 1], {"units": "s"})},
    ...     dims=("x",),
    ...     name="a",
    ... )
    >>> plts = xr.plot.dataarray_plot._prepare_plot1d_data(
    ...     a, coords_to_plot={"x": 1, "z": None, "hue": None, "size": None}
    ... )
    >>> # Check which coords to plot:
    >>> print({k: v.name for k, v in plts.items()})
    {'y': 'a', 'x': 1}
    """
    plts = {}
    
    # Handle y-axis data
    if darray.name:
        plts['y'] = darray
    else:
        plts['y'] = darray.rename('y')
    
    # Handle x-axis data
    x_coord = coords_to_plot.get('x')
    if x_coord is not None:
        if isinstance(x_coord, str):
            plts['x'] = darray[x_coord]
        else:
            plts['x'] = darray.coords[x_coord]
    else:
        plts['x'] = darray.dims[0]
    
    # Handle hue data
    hue_coord = coords_to_plot.get('hue')
    if hue_coord is not None:
        plts['hue'] = darray.coords[hue_coord]
    
    # Handle size data
    size_coord = coords_to_plot.get('size')
    if size_coord is not None:
        plts['size'] = darray.coords[size_coord]
    
    return plts

    Dimensions      Plotting function

    =============== ===========================
    Dimensions      Plotting function
    =============== ===========================
    1               :py:func:`xarray.plot.line`
    2               :py:func:`xarray.plot.pcolormesh`
    Anything else   :py:func:`xarray.plot.hist`
    =============== ===========================

    Parameters
    ----------
    darray : DataArray
    row : Hashable or None, optional
        If passed, make row faceted plots on this dimension name.
    col : Hashable or None, optional
        If passed, make column faceted plots on this dimension name.
    col_wrap : int or None, optional
        Use together with ``col`` to wrap faceted plots.
    ax : matplotlib axes object, optional
        Axes on which to plot. By default, use the current axes.
        Mutually exclusive with ``size``, ``figsize`` and facets.
    hue : Hashable or None, optional
        If passed, make faceted line plots with hue on this dimension name.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for Matplotlib subplots
        (see :py:meth:`matplotlib:matplotlib.figure.Figure.add_subplot`).
    **kwargs : optional
        Additional keyword arguments for Matplotlib.

    See Also
    --------
    xarray.DataArray.squeeze
    """
    pass

def line(darray: T_DataArray, *args: Any, row: Hashable | None=None, col: Hashable | None=None, figsize: Iterable[float] | None=None, aspect: AspectOptions=None, size: float | None=None, ax: Axes | None=None, hue: Hashable | None=None, x: Hashable | None=None, y: Hashable | None=None, xincrease: bool | None=None, yincrease: bool | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, add_legend: bool=True, _labels: bool=True, **kwargs: Any) -> list[Line3D] | FacetGrid[T_DataArray]:
    """
    Line plot of DataArray values.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.plot`.

    Parameters
    ----------
    darray : DataArray
        Either 1D or 2D. If 2D, one of ``hue``, ``x`` or ``y`` must be provided.
    row : Hashable, optional
        If passed, make row faceted plots on this dimension name.
    col : Hashable, optional
        If passed, make column faceted plots on this dimension name.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : "auto", "equal", scalar or None, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the *width* in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size:
        *height* (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axes on which to plot. By default, the current is used.
        Mutually exclusive with ``size`` and ``figsize``.
    hue : Hashable, optional
        Dimension or coordinate for which you want multiple lines plotted.
        If plotting against a 2D coordinate, ``hue`` must be a dimension.
    x, y : Hashable, optional
        Dimension, coordinate or multi-index level for *x*, *y* axis.
        Only one of these may be specified.
        The other will be used for values from the DataArray on which this
        plot method is called.
    xincrease : bool or None, optional
        Should the values on the *x* axis be increasing from left to right?
        if ``None``, use the default for the Matplotlib function.
    yincrease : bool or None, optional
        Should the values on the *y* axis be increasing from top to bottom?
        if ``None``, use the default for the Matplotlib function.
    xscale, yscale : {'linear', 'symlog', 'log', 'logit'}, optional
        Specifies scaling for the *x*- and *y*-axis, respectively.
    xticks, yticks : array-like, optional
        Specify tick locations for *x*- and *y*-axis.
    xlim, ylim : tuple[float, float], optional
        Specify *x*- and *y*-axis limits.
    add_legend : bool, default: True
        Add legend with *y* axis coordinates (2D inputs only).
    *args, **kwargs : optional
        Additional arguments to :py:func:`matplotlib:matplotlib.pyplot.plot`.

    Returns
    -------
    primitive : list of Line3D or FacetGrid
        When either col or row is given, returns a FacetGrid, otherwise
        a list of matplotlib Line3D objects.
    """
    import matplotlib.pyplot as plt

    # If faceting, use FacetGrid
    if row or col:
        return _easy_facetgrid(darray, row=row, col=col, figsize=figsize, aspect=aspect, size=size, hue=hue, x=x, y=y, **kwargs)

    # Prepare the data
    coords_to_plot = {'x': x, 'y': y, 'hue': hue}
    plts = _prepare_plot1d_data(darray, coords_to_plot)

    # Create or get the axes
    if ax is None:
        if figsize is not None:
            plt.figure(figsize=figsize)
        ax = plt.gca()

    # Plot the data
    lines = ax.plot(plts['x'], plts['y'], *args, **kwargs)

    # Set scales
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    # Set limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Set labels
    if _labels:
        ax.set_xlabel(label_from_attrs(plts['x']))
        ax.set_ylabel(label_from_attrs(plts['y']))

    # Add legend if needed
    if add_legend and hue is not None:
        ax.legend()

    return lines

def step(darray: DataArray, *args: Any, where: Literal['pre', 'post', 'mid']='pre', drawstyle: str | None=None, ds: str | None=None, row: Hashable | None=None, col: Hashable | None=None, **kwargs: Any) -> list[Line3D] | FacetGrid[DataArray]:
    """
    Step plot of DataArray values.

    Similar to :py:func:`matplotlib:matplotlib.pyplot.step`.

    Parameters
    ----------
    where : {'pre', 'post', 'mid'}, default: 'pre'
        Define where the steps should be placed:

        - ``'pre'``: The y value is continued constantly to the left from
          every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
          value ``y[i]``.
        - ``'post'``: The y value is continued constantly to the right from
          every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
          value ``y[i]``.
        - ``'mid'``: Steps occur half-way between the *x* positions.

        Note that this parameter is ignored if one coordinate consists of
        :py:class:`pandas.Interval` values, e.g. as a result of
        :py:func:`xarray.Dataset.groupby_bins`. In this case, the actual
        boundaries of the interval are used.
    drawstyle, ds : str or None, optional
        Additional drawstyle. Only use one of drawstyle and ds.
    row : Hashable, optional
        If passed, make row faceted plots on this dimension name.
    col : Hashable, optional
        If passed, make column faceted plots on this dimension name.
    *args, **kwargs : optional
        Additional arguments for :py:func:`xarray.plot.line`.

    Returns
    -------
    primitive : list of Line3D or FacetGrid
        When either col or row is given, returns a FacetGrid, otherwise
        a list of matplotlib Line3D objects.
    """
    pass

def hist(darray: DataArray, *args: Any, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, xincrease: bool | None=None, yincrease: bool | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, BarContainer | Polygon]:
    """
    Histogram of DataArray.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.hist`.

    Plots *N*-dimensional arrays by first flattening the array.

    Parameters
    ----------
    darray : DataArray
        Can have any number of dimensions.
    figsize : Iterable of float, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : "auto", "equal", scalar or None, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the *width* in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size:
        *height* (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axes on which to plot. By default, use the current axes.
        Mutually exclusive with ``size`` and ``figsize``.
    xincrease : bool or None, optional
        Should the values on the *x* axis be increasing from left to right?
        if ``None``, use the default for the Matplotlib function.
    yincrease : bool or None, optional
        Should the values on the *y* axis be increasing from top to bottom?
        if ``None``, use the default for the Matplotlib function.
    xscale, yscale : {'linear', 'symlog', 'log', 'logit'}, optional
        Specifies scaling for the *x*- and *y*-axis, respectively.
    xticks, yticks : array-like, optional
        Specify tick locations for *x*- and *y*-axis.
    xlim, ylim : tuple[float, float], optional
        Specify *x*- and *y*-axis limits.
    **kwargs : optional
        Additional keyword arguments to :py:func:`matplotlib:matplotlib.pyplot.hist`.

    """
    pass

def _plot1d(plotfunc):
    """Decorator for common 1d plotting logic."""
    pass

def _add_labels(add_labels: bool | Iterable[bool], darrays: Iterable[DataArray | None], suffixes: Iterable[str], ax: Axes) -> None:
    """Set x, y, z labels."""
    pass

@_plot1d
def scatter(xplt: DataArray | None, yplt: DataArray | None, ax: Axes, add_labels: bool | Iterable[bool]=True, **kwargs) -> PathCollection:
    """Scatter variables against each other.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.scatter`.
    """
    pass

def _plot2d(plotfunc):
    """Decorator for common 2d plotting logic."""
    pass

@_plot2d
def imshow(x: np.ndarray, y: np.ndarray, z: np.ma.core.MaskedArray, ax: Axes, **kwargs: Any) -> AxesImage:
    """
    Image plot of 2D DataArray.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.imshow`.

    While other plot methods require the DataArray to be strictly
    two-dimensional, ``imshow`` also accepts a 3D array where some
    dimension can be interpreted as RGB or RGBA color channels and
    allows this dimension to be specified via the kwarg ``rgb=``.

    Unlike :py:func:`matplotlib:matplotlib.pyplot.imshow`, which ignores ``vmin``/``vmax``
    for RGB(A) data,
    xarray *will* use ``vmin`` and ``vmax`` for RGB(A) data
    by applying a single scaling factor and offset to all bands.
    Passing  ``robust=True`` infers ``vmin`` and ``vmax``
    :ref:`in the usual way <robust-plotting>`.
    Additionally the y-axis is not inverted by default, you can
    restore the matplotlib behavior by setting `yincrease=False`.

    .. note::
        This function needs uniformly spaced coordinates to
        properly label the axes. Call :py:meth:`DataArray.plot` to check.

    The pixels are centered on the coordinates. For example, if the coordinate
    value is 3.2, then the pixels for those coordinates will be centered on 3.2.
    """
    pass

@_plot2d
def contour(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axes, **kwargs: Any) -> QuadContourSet:
    """
    Contour plot of 2D DataArray.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.contour`.
    """
    pass

@_plot2d
def contourf(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axes, **kwargs: Any) -> QuadContourSet:
    """
    Filled contour plot of 2D DataArray.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.contourf`.
    """
    pass

@_plot2d
def pcolormesh(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axes, xscale: ScaleOptions | None=None, yscale: ScaleOptions | None=None, infer_intervals=None, **kwargs: Any) -> QuadMesh:
    """
    Pseudocolor plot of 2D DataArray.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.pcolormesh`.
    """
    pass

@_plot2d
def surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axes, **kwargs: Any) -> Poly3DCollection:
    """
    Surface plot of 2D DataArray.

    Wraps :py:meth:`matplotlib:mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`.
    """
    pass
