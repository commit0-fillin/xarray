from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
QUANTILE_METHODS = Literal['linear', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'median_unbiased', 'normal_unbiased']
_WEIGHTED_REDUCE_DOCSTRING_TEMPLATE = "\n    Reduce this {cls}'s data by a weighted ``{fcn}`` along some dimension(s).\n\n    Parameters\n    ----------\n    dim : Hashable or Iterable of Hashable, optional\n        Dimension(s) over which to apply the weighted ``{fcn}``.\n    skipna : bool or None, optional\n        If True, skip missing values (as marked by NaN). By default, only\n        skips missing values for float dtypes; other dtypes either do not\n        have a sentinel missing value (int) or skipna=True has not been\n        implemented (object, datetime64 or timedelta64).\n    keep_attrs : bool or None, optional\n        If True, the attributes (``attrs``) will be copied from the original\n        object to the new one.  If False (default), the new object will be\n        returned without attributes.\n\n    Returns\n    -------\n    reduced : {cls}\n        New {cls} object with weighted ``{fcn}`` applied to its data and\n        the indicated dimension(s) removed.\n\n    Notes\n    -----\n        Returns {on_zero} if the ``weights`` sum to 0.0 along the reduced\n        dimension(s).\n    "
_SUM_OF_WEIGHTS_DOCSTRING = '\n    Calculate the sum of weights, accounting for missing values in the data.\n\n    Parameters\n    ----------\n    dim : str or sequence of str, optional\n        Dimension(s) over which to sum the weights.\n    keep_attrs : bool, optional\n        If True, the attributes (``attrs``) will be copied from the original\n        object to the new one.  If False (default), the new object will be\n        returned without attributes.\n\n    Returns\n    -------\n    reduced : {cls}\n        New {cls} object with the sum of the weights over the given dimension.\n    '
_WEIGHTED_QUANTILE_DOCSTRING_TEMPLATE = '\n    Apply a weighted ``quantile`` to this {cls}\'s data along some dimension(s).\n\n    Weights are interpreted as *sampling weights* (or probability weights) and\n    describe how a sample is scaled to the whole population [1]_. There are\n    other possible interpretations for weights, *precision weights* describing the\n    precision of observations, or *frequency weights* counting the number of identical\n    observations, however, they are not implemented here.\n\n    For compatibility with NumPy\'s non-weighted ``quantile`` (which is used by\n    ``DataArray.quantile`` and ``Dataset.quantile``), the only interpolation\n    method supported by this weighted version corresponds to the default "linear"\n    option of ``numpy.quantile``. This is "Type 7" option, described in Hyndman\n    and Fan (1996) [2]_. The implementation is largely inspired by a blog post\n    from A. Akinshin\'s (2023) [3]_.\n\n    Parameters\n    ----------\n    q : float or sequence of float\n        Quantile to compute, which must be between 0 and 1 inclusive.\n    dim : str or sequence of str, optional\n        Dimension(s) over which to apply the weighted ``quantile``.\n    skipna : bool, optional\n        If True, skip missing values (as marked by NaN). By default, only\n        skips missing values for float dtypes; other dtypes either do not\n        have a sentinel missing value (int) or skipna=True has not been\n        implemented (object, datetime64 or timedelta64).\n    keep_attrs : bool, optional\n        If True, the attributes (``attrs``) will be copied from the original\n        object to the new one.  If False (default), the new object will be\n        returned without attributes.\n\n    Returns\n    -------\n    quantiles : {cls}\n        New {cls} object with weighted ``quantile`` applied to its data and\n        the indicated dimension(s) removed.\n\n    See Also\n    --------\n    numpy.nanquantile, pandas.Series.quantile, Dataset.quantile, DataArray.quantile\n\n    Notes\n    -----\n    Returns NaN if the ``weights`` sum to 0.0 along the reduced\n    dimension(s).\n\n    References\n    ----------\n    .. [1] https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/\n    .. [2] Hyndman, R. J. & Fan, Y. (1996). Sample Quantiles in Statistical Packages.\n           The American Statistician, 50(4), 361–365. https://doi.org/10.2307/2684934\n    .. [3] Akinshin, A. (2023) "Weighted quantile estimators" arXiv:2304.07265 [stat.ME]\n           https://arxiv.org/abs/2304.07265\n    '
if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

class Weighted(Generic[T_Xarray]):
    """An object that implements weighted operations.

    You should create a Weighted object by using the ``DataArray.weighted`` or
    ``Dataset.weighted`` methods.

    See Also
    --------
    Dataset.weighted
    DataArray.weighted
    """
    __slots__ = ('obj', 'weights')

    def __init__(self, obj: T_Xarray, weights: T_DataArray) -> None:
        """
        Create a Weighted object

        Parameters
        ----------
        obj : DataArray or Dataset
            Object over which the weighted reduction operation is applied.
        weights : DataArray
            An array of weights associated with the values in the obj.
            Each value in the obj contributes to the reduction operation
            according to its associated weight.

        Notes
        -----
        ``weights`` must be a ``DataArray`` and cannot contain missing values.
        Missing values can be replaced by ``weights.fillna(0)``.
        """
        from xarray.core.dataarray import DataArray
        if not isinstance(weights, DataArray):
            raise ValueError('`weights` must be a DataArray')

        def _weight_check(w):
            if duck_array_ops.isnull(w).any():
                raise ValueError('`weights` cannot contain missing values. Missing values can be replaced by `weights.fillna(0)`.')
            return w
        if is_duck_dask_array(weights.data):
            weights = weights.copy(data=weights.data.map_blocks(_weight_check, dtype=weights.dtype), deep=False)
        else:
            _weight_check(weights.data)
        self.obj: T_Xarray = obj
        self.weights: T_DataArray = weights

    def _check_dim(self, dim: Dims):
        """raise an error if any dimension is missing"""
        if isinstance(dim, str):
            dim = [dim]
        if any(d not in self.weights.dims for d in dim):
            raise ValueError(f"Dimension {dim} not found in weights")

    @staticmethod
    def _reduce(da: T_DataArray, weights: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """reduce using dot; equivalent to (da * weights).sum(dim, skipna)

        for internal use only
        """
        if skipna or (skipna is None and da.dtype.kind in 'cfO'):
            da = da.where(da.notnull())
            weights = weights.where(da.notnull())

        return dot(da, weights, dims=dim)

    def _sum_of_weights(self, da: T_DataArray, dim: Dims=None) -> T_DataArray:
        """Calculate the sum of weights, accounting for missing values"""
        weights = self.weights
        if dim is not None:
            weights = weights.sel({d: da[d] for d in dim if d in da.dims})
        
        return weights.where(da.notnull()).sum(dim=dim)

    def _sum_of_squares(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``sum_of_squares`` along some dimension(s)."""
        return self._reduce(da ** 2, self.weights, dim=dim, skipna=skipna)

    def _weighted_sum(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``sum`` along some dimension(s)."""
        return self._reduce(da, self.weights, dim=dim, skipna=skipna)

    def _weighted_mean(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""
        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)
        sum_of_weights = self._sum_of_weights(da, dim=dim)
        return weighted_sum / sum_of_weights

    def _weighted_var(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``var`` along some dimension(s)."""
        weighted_mean = self._weighted_mean(da, dim=dim, skipna=skipna)
        dev = (da - weighted_mean) ** 2
        weighted_sum_of_dev = self._weighted_sum(dev, dim=dim, skipna=skipna)
        sum_of_weights = self._sum_of_weights(da, dim=dim)
        return weighted_sum_of_dev / sum_of_weights

    def _weighted_std(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``std`` along some dimension(s)."""
        return np.sqrt(self._weighted_var(da, dim=dim, skipna=skipna))

    def _weighted_quantile(self, da: T_DataArray, q: ArrayLike, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Apply a weighted ``quantile`` to a DataArray along some dimension(s)."""
        if dim is None:
            dim = da.dims
        
        if skipna or skipna is None:
            mask = da.notnull()
            da = da.where(mask)
            weights = self.weights.where(mask)
        else:
            weights = self.weights

        sorted_idx = da.argsort(dim=dim)
        sorted_data = da.isel({d: sorted_idx for d in dim})
        sorted_weights = weights.isel({d: sorted_idx for d in dim})

        cumsum_weights = sorted_weights.cumsum(dim=dim)
        total_weights = cumsum_weights.isel({d: -1 for d in dim})

        # Normalize cumulative weights
        cumsum_weights /= total_weights

        # Compute quantiles
        return sorted_data.interp({d: xr.DataArray(q, dims='quantile') for d in dim}, 
                                  xi=cumsum_weights, 
                                  method='linear')

    def __repr__(self) -> str:
        """provide a nice str repr of our Weighted object"""
        klass = self.__class__.__name__
        weight_dims = ', '.join(map(str, self.weights.dims))
        return f'{klass} with weights along dimensions: {weight_dims}'

class DataArrayWeighted(Weighted['DataArray']):
    pass

class DatasetWeighted(Weighted['Dataset']):
    pass
_inject_docstring(DataArrayWeighted, 'DataArray')
_inject_docstring(DatasetWeighted, 'Dataset')
