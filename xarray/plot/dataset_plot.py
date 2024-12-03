from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from xarray.core.alignment import broadcast
from xarray.plot import dataarray_plot
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import _add_colorbar, _get_nice_quiver_magnitude, _infer_meta_data, _process_cmap_cbar_kwargs, get_axis
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.quiver import Quiver
    from numpy.typing import ArrayLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import AspectOptions, ExtendOptions, HueStyleOptions, ScaleOptions
    from xarray.plot.facetgrid import FacetGrid

@_dsplot
def quiver(ds: Dataset, x: Hashable, y: Hashable, ax: Axes, u: Hashable, v: Hashable, **kwargs: Any) -> Quiver:
    """Quiver plot of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.quiver`.
    """
    x_values = ds[x].values
    y_values = ds[y].values
    u_values = ds[u].values
    v_values = ds[v].values

    # Calculate the magnitude of the vectors
    magnitude = _get_nice_quiver_magnitude(u_values, v_values)

    # Normalize the vectors
    u_norm = u_values / magnitude
    v_norm = v_values / magnitude

    return ax.quiver(x_values, y_values, u_norm, v_norm, scale=magnitude, **kwargs)

@_dsplot
def streamplot(ds: Dataset, x: Hashable, y: Hashable, ax: Axes, u: Hashable, v: Hashable, **kwargs: Any) -> LineCollection:
    """Plot streamlines of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.streamplot`.
    """
    x_values = ds[x].values
    y_values = ds[y].values
    u_values = ds[u].values
    v_values = ds[v].values

    # Create a meshgrid for x and y
    X, Y = np.meshgrid(x_values, y_values)

    streamplot = ax.streamplot(X, Y, u_values, v_values, **kwargs)
    return streamplot.lines
F = TypeVar('F', bound=Callable)

def _update_doc_to_dataset(dataarray_plotfunc: Callable) -> Callable[[F], F]:
    """
    Add a common docstring by re-using the DataArray one.

    TODO: Reduce code duplication.

    * The goal is to reduce code duplication by moving all Dataset
      specific plots to the DataArray side and use this thin wrapper to
      handle the conversion between Dataset and DataArray.
    * Improve docstring handling, maybe reword the DataArray versions to
      explain Datasets better.

    Parameters
    ----------
    dataarray_plotfunc : Callable
        Function that returns a finished plot primitive.
    """
    def wrapper(func: F) -> F:
        func.__doc__ = dataarray_plotfunc.__doc__.replace("DataArray", "Dataset")
        return func
    return wrapper

def _temp_dataarray(ds: Dataset, y: Hashable, locals_: dict[str, Any]) -> DataArray:
    """Create a temporary datarray with extra coords."""
    da = ds[y]
    
    # Add extra coordinates
    extra_coords = {}
    for key, value in locals_.items():
        if key in ds and key != y:
            extra_coords[key] = ds[key]
    
    if extra_coords:
        da = da.assign_coords(**extra_coords)
    
    return da

@_update_doc_to_dataset(dataarray_plot.scatter)
def scatter(ds: Dataset, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: ArrayLike | None=None, ylim: ArrayLike | None=None, cmap: str | Colormap | None=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend: ExtendOptions=None, levels: ArrayLike | None=None, **kwargs: Any) -> PathCollection | FacetGrid[DataArray]:
    """Scatter plot Dataset data variables against each other."""
    if x is None or y is None:
        raise ValueError("Both 'x' and 'y' must be specified for Dataset scatter plots")

    # Create a temporary DataArray for x
    x_da = _temp_dataarray(ds, x, locals())

    # Create a temporary DataArray for y
    y_da = _temp_dataarray(ds, y, locals())

    # Align the x and y DataArrays
    x_da, y_da = broadcast(x_da, y_da)

    # Call the DataArray scatter function
    return dataarray_plot.scatter(
        x_da, y=y_da, z=z, hue=hue, hue_style=hue_style, markersize=markersize,
        linewidth=linewidth, figsize=figsize, size=size, aspect=aspect, ax=ax,
        row=row, col=col, col_wrap=col_wrap, xincrease=xincrease, yincrease=yincrease,
        add_legend=add_legend, add_colorbar=add_colorbar, add_labels=add_labels,
        add_title=add_title, subplot_kws=subplot_kws, xscale=xscale, yscale=yscale,
        xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim, cmap=cmap, vmin=vmin,
        vmax=vmax, norm=norm, extend=extend, levels=levels, **kwargs
    )
