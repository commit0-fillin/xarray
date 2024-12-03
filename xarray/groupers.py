"""
This module provides Grouper objects that encapsulate the
"factorization" process - conversion of value we are grouping by
to integer codes (one per group).
"""
from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, cast
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.groupby import T_Group, _DummyGroup
from xarray.core.indexes import safe_cast_to_index
from xarray.core.resample_cftime import CFTimeGrouper
from xarray.core.types import Bins, DatetimeLike, GroupIndices, SideOptions
from xarray.core.variable import Variable
__all__ = ['EncodedGroups', 'Grouper', 'Resampler', 'UniqueGrouper', 'BinGrouper', 'TimeResampler']
RESAMPLE_DIM = '__resample_dim__'

@dataclass
class EncodedGroups:
    """
    Dataclass for storing intermediate values for GroupBy operation.
    Returned by the ``factorize`` method on Grouper objects.

    Attributes
    ----------
    codes : DataArray
        Same shape as the DataArray to group by. Values consist of a unique integer code for each group.
    full_index : pd.Index
        Pandas Index for the group coordinate containing unique group labels.
        This can differ from ``unique_coord`` in the case of resampling and binning,
        where certain groups in the output need not be present in the input.
    group_indices : tuple of int or slice or list of int, optional
        List of indices of array elements belonging to each group. Inferred if not provided.
    unique_coord : Variable, optional
        Unique group values present in dataset. Inferred if not provided
    """
    codes: DataArray
    full_index: pd.Index
    group_indices: GroupIndices | None = field(default=None)
    unique_coord: Variable | _DummyGroup | None = field(default=None)

    def __post_init__(self):
        assert isinstance(self.codes, DataArray)
        if self.codes.name is None:
            raise ValueError('Please set a name on the array you are grouping by.')
        assert isinstance(self.full_index, pd.Index)
        assert isinstance(self.unique_coord, (Variable, _DummyGroup)) or self.unique_coord is None

class Grouper(ABC):
    """Abstract base class for Grouper objects that allow specializing GroupBy instructions."""

    @abstractmethod
    def factorize(self, group: T_Group) -> EncodedGroups:
        """
        Creates intermediates necessary for GroupBy.

        Parameters
        ----------
        group : DataArray
            DataArray we are grouping by.

        Returns
        -------
        EncodedGroups
        """
        if isinstance(group, DataArray):
            data = group.data
        else:
            data = group

        # Use pandas factorize for 1D data
        if group.ndim == 1:
            codes, categories = pd.factorize(data, sort=True)
            if isinstance(categories, pd.MultiIndex):
                raise ValueError("MultiIndex grouping is not supported")
            full_index = pd.Index(categories)
        else:
            # For multi-dimensional data, use numpy unique
            data_flat = data.reshape(-1)
            categories, codes = np.unique(data_flat, return_inverse=True)
            codes = codes.reshape(data.shape)
            full_index = pd.Index(categories)

        group_indices = [np.nonzero(codes == i)[0] for i in range(len(categories))]
        
        unique_coord = Variable(self.name, categories)
        
        codes_da = DataArray(codes, dims=group.dims, coords=group.coords)

        return EncodedGroups(codes=codes_da,
                             full_index=full_index,
                             group_indices=group_indices,
                             unique_coord=unique_coord)

class Resampler(Grouper):
    """
    Abstract base class for Grouper objects that allow specializing resampling-type GroupBy instructions.

    Currently only used for TimeResampler, but could be used for SpaceResampler in the future.
    """
    pass

@dataclass
class UniqueGrouper(Grouper):
    """Grouper object for grouping by a categorical variable."""
    _group_as_index: pd.Index | None = field(default=None, repr=False)

    @property
    def group_as_index(self) -> pd.Index:
        """Caches the group DataArray as a pandas Index."""
        if self._group_as_index is None:
            if isinstance(self.group, DataArray):
                self._group_as_index = self.group.to_index()
            else:
                self._group_as_index = pd.Index(self.group)
        return self._group_as_index

@dataclass
class BinGrouper(Grouper):
    """
    Grouper object for binning numeric data.

    Attributes
    ----------
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error. When `ordered=False`, labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {"raise", "drop"}, default: "raise"
        If bin edges are not unique, raise ValueError or drop non-uniques.
    """
    bins: Bins
    right: bool = True
    labels: Any = None
    precision: int = 3
    include_lowest: bool = False
    duplicates: Literal['raise', 'drop'] = 'raise'

    def __post_init__(self) -> None:
        if duck_array_ops.isnull(self.bins).all():
            raise ValueError('All bin edges are NaN.')

@dataclass(repr=False)
class TimeResampler(Resampler):
    """
    Grouper object specialized to resampling the time coordinate.

    Attributes
    ----------
    freq : str
        Frequency to resample to. See `Pandas frequency
        aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
        for a list of possible values.
    closed : {"left", "right"}, optional
        Side of each interval to treat as closed.
    label : {"left", "right"}, optional
        Side of each interval to use for labeling.
    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'}, pandas.Timestamp, datetime.datetime, numpy.datetime64, or cftime.datetime, default 'start_day'
        The datetime on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.

        If a datetime is not used, these values are also supported:
        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day
    offset : pd.Timedelta, datetime.timedelta, or str, default is None
        An offset timedelta added to the origin.
    """
    freq: str
    closed: SideOptions | None = field(default=None)
    label: SideOptions | None = field(default=None)
    origin: str | DatetimeLike = field(default='start_day')
    offset: pd.Timedelta | datetime.timedelta | str | None = field(default=None)
    index_grouper: CFTimeGrouper | pd.Grouper = field(init=False, repr=False)
    group_as_index: pd.Index = field(init=False, repr=False)

def unique_value_groups(ar, sort: bool=True) -> tuple[np.ndarray | pd.Index, np.ndarray]:
    """Group an array by its unique values.

    Parameters
    ----------
    ar : array-like
        Input array. This will be flattened if it is not already 1-D.
    sort : bool, default: True
        Whether or not to sort unique values.

    Returns
    -------
    values : np.ndarray
        Sorted, unique values as returned by `np.unique`.
    indices : list of lists of int
        Each element provides the integer indices in `ar` with values given by
        the corresponding value in `unique_values`.
    """
    if isinstance(ar, pd.Index):
        values, indices = ar.factorize(sort=sort)
        if isinstance(ar, pd.MultiIndex):
            values = ar[indices.argsort()]
        else:
            values = ar.take(indices.argsort())
    else:
        ar = np.asarray(ar)
        if ar.ndim > 1:
            ar = ar.ravel()
        
        values, inverse = np.unique(ar, return_inverse=True)
        
        if sort:
            perm = values.argsort()
            values = values[perm]
            inverse = np.take(perm, inverse)

        indices = [[] for _ in range(len(values))]
        for n, idx in enumerate(inverse):
            indices[idx].append(n)

    return values, indices
