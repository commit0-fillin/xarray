from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core._aggregations import DataArrayResampleAggregations, DatasetResampleAggregations
from xarray.core.groupby import DataArrayGroupByBase, DatasetGroupByBase, GroupBy
from xarray.core.types import Dims, InterpOptions, T_Xarray
if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
from xarray.groupers import RESAMPLE_DIM

class Resample(GroupBy[T_Xarray]):
    """An object that extends the `GroupBy` object with additional logic
    for handling specialized re-sampling operations.

    You should create a `Resample` object by using the `DataArray.resample` or
    `Dataset.resample` methods. The dimension along re-sampling

    See Also
    --------
    DataArray.resample
    Dataset.resample

    """

    def __init__(self, *args, dim: Hashable | None=None, resample_dim: Hashable | None=None, **kwargs) -> None:
        if dim == resample_dim:
            raise ValueError(f"Proxy resampling dimension ('{resample_dim}') cannot have the same name as actual dimension ('{dim}')!")
        self._dim = dim
        super().__init__(*args, **kwargs)

    def _drop_coords(self) -> T_Xarray:
        """Drop non-dimension coordinates along the resampled dimension."""
        obj = self._obj
        dim = self._dim
        return obj.drop_vars([c for c in obj.coords if dim in obj[c].dims and c != dim])

    def pad(self, tolerance: float | Iterable[float] | str | None=None) -> T_Xarray:
        """Forward fill new values at up-sampled frequency.

        Parameters
        ----------
        tolerance : float | Iterable[float] | str | None, default: None
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN`` s.

        Returns
        -------
        padded : DataArray or Dataset
        """
        return self._obj.resample(self._dim, self.grouper.freq).pad(tolerance=tolerance)
    ffill = pad

    def backfill(self, tolerance: float | Iterable[float] | str | None=None) -> T_Xarray:
        """Backward fill new values at up-sampled frequency.

        Parameters
        ----------
        tolerance : float | Iterable[float] | str | None, default: None
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN`` s.

        Returns
        -------
        backfilled : DataArray or Dataset
        """
        return self._obj.resample(self._dim, self.grouper.freq).backfill(tolerance=tolerance)
    bfill = backfill

    def nearest(self, tolerance: float | Iterable[float] | str | None=None) -> T_Xarray:
        """Take new values from nearest original coordinate to up-sampled
        frequency coordinates.

        Parameters
        ----------
        tolerance : float | Iterable[float] | str | None, default: None
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN`` s.

        Returns
        -------
        upsampled : DataArray or Dataset
        """
        return self._obj.resample(self._dim, self.grouper.freq).nearest(tolerance=tolerance)

    def interpolate(self, kind: InterpOptions='linear', **kwargs) -> T_Xarray:
        """Interpolate up-sampled data using the original data as knots.

        Parameters
        ----------
        kind : {"linear", "nearest", "zero", "slinear",                 "quadratic", "cubic", "polynomial"}, default: "linear"
            The method used to interpolate. The method should be supported by
            the scipy interpolator:

            - ``interp1d``: {"linear", "nearest", "zero", "slinear",
              "quadratic", "cubic", "polynomial"}
            - ``interpn``: {"linear", "nearest"}

            If ``"polynomial"`` is passed, the ``order`` keyword argument must
            also be provided.

        Returns
        -------
        interpolated : DataArray or Dataset

        See Also
        --------
        DataArray.interp
        Dataset.interp
        scipy.interpolate.interp1d

        """
        return self._interpolate(kind=kind, **kwargs)

    def _interpolate(self, kind='linear', **kwargs) -> T_Xarray:
        """Apply scipy.interpolate.interp1d along resampling dimension."""
        from scipy import interpolate

        obj = self._obj
        dim = self._dim
        index = obj.get_index(dim)
        new_index = pd.date_range(start=index[0], end=index[-1], freq=self.grouper.freq)

        interpolator = interpolate.interp1d(index.astype(float), obj.values, kind=kind, axis=obj.get_axis_num(dim), **kwargs)
        new_values = interpolator(new_index.astype(float))

        return obj.__class__(new_values, coords={dim: new_index, **{k: v for k, v in obj.coords.items() if k != dim}}, attrs=obj.attrs)

class DataArrayResample(Resample['DataArray'], DataArrayGroupByBase, DataArrayResampleAggregations):
    """DataArrayGroupBy object specialized to time resampling operations over a
    specified dimension
    """

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = self._dim
        return self._obj.resample({self._dim: self.grouper.freq}).reduce(func, dim=dim, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)

    def map(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=False, **kwargs: Any) -> DataArray:
        """Apply a function to each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:

            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.

            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        resampled = self._obj.resample({self._dim: self.grouper.freq})
        return resampled.map(func, args=args, shortcut=shortcut, **kwargs)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayResample.map
        """
        return self.map(func, args=args, shortcut=shortcut, **kwargs)

    def asfreq(self) -> DataArray:
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.

        Returns
        -------
        resampled : DataArray
        """
        return self._obj.resample({self._dim: self.grouper.freq}).asfreq()

class DatasetResample(Resample['Dataset'], DatasetGroupByBase, DatasetResampleAggregations):
    """DatasetGroupBy object specialized to resampling a specified dimension"""

    def map(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> Dataset:
        """Apply a function over each Dataset in the groups generated for
        resampling and concatenate them together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        resampled = self._obj.resample({self._dim: self.grouper.freq})
        return resampled.map(func, args=args, shortcut=shortcut, **kwargs)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataSetResample.map
        """
        return self.map(func, args=args, shortcut=shortcut, **kwargs)

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = self._dim
        return self._obj.resample({self._dim: self.grouper.freq}).reduce(func, dim=dim, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)

    def asfreq(self) -> Dataset:
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.

        Returns
        -------
        resampled : Dataset
        """
        return self._obj.resample({self._dim: self.grouper.freq}).asfreq()
