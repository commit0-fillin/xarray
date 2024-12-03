"""FrequencyInferer analog for cftime.datetime objects"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
_ONE_MICRO = 1
_ONE_MILLI = _ONE_MICRO * 1000
_ONE_SECOND = _ONE_MILLI * 1000
_ONE_MINUTE = 60 * _ONE_SECOND
_ONE_HOUR = 60 * _ONE_MINUTE
_ONE_DAY = 24 * _ONE_HOUR

def infer_freq(index):
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : CFTimeIndex, DataArray, DatetimeIndex, TimedeltaIndex, Series
        If not passed a CFTimeIndex, this simply calls `pandas.infer_freq`.
        If passed a Series or a DataArray will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values or the index is not 1D.
    """
    if isinstance(index, CFTimeIndex):
        return _CFTimeFrequencyInferer(index).get_freq()
    elif isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return pd.infer_freq(index)
    elif isinstance(index, (pd.Series, xr.DataArray)):
        return pd.infer_freq(pd.Index(index.values))
    else:
        raise TypeError("Index must be CFTimeIndex, DatetimeIndex, TimedeltaIndex, Series, or DataArray")

class _CFTimeFrequencyInferer:

    def __init__(self, index):
        self.index = index
        self.values = index.asi8
        if len(index) < 3:
            raise ValueError('Need at least 3 dates to infer frequency')
        self.is_monotonic = self.index.is_monotonic_decreasing or self.index.is_monotonic_increasing
        self._deltas = None
        self._year_deltas = None
        self._month_deltas = None

    def get_freq(self):
        """Find the appropriate frequency string to describe the inferred frequency of self.index

        Adapted from `pandas.tsseries.frequencies._FrequencyInferer.get_freq` for CFTimeIndexes.

        Returns
        -------
        str or None
        """
        if not self.is_monotonic:
            return None

        if len(self.index) < 3:
            return None

        delta_microseconds = self.deltas[0]
        if is_multiple(delta_microseconds, _ONE_DAY):
            return self._infer_daily_rule()

        if is_multiple(delta_microseconds, _ONE_HOUR):
            return self._infer_hourly_rule()

        if is_multiple(delta_microseconds, _ONE_MINUTE):
            return self._infer_minute_rule()

        if is_multiple(delta_microseconds, _ONE_SECOND):
            return self._infer_second_rule()

        if is_multiple(delta_microseconds, _ONE_MILLI):
            return self._infer_millisecond_rule()

        if is_multiple(delta_microseconds, _ONE_MICRO):
            return self._infer_microsecond_rule()

        return None

    @property
    def deltas(self):
        """Sorted unique timedeltas as microseconds."""
        if self._deltas is None:
            deltas = np.diff(self.values)
            self._deltas = _unique_deltas(deltas)
        return self._deltas

    @property
    def year_deltas(self):
        """Sorted unique year deltas."""
        if self._year_deltas is None:
            years = np.array([dt.year for dt in self.index])
            self._year_deltas = _unique_deltas(np.diff(years))
        return self._year_deltas

    @property
    def month_deltas(self):
        """Sorted unique month deltas."""
        if self._month_deltas is None:
            months = np.array([dt.year * 12 + dt.month for dt in self.index])
            self._month_deltas = _unique_deltas(np.diff(months))
        return self._month_deltas

def _unique_deltas(arr):
    """Sorted unique deltas of numpy array"""
    return np.unique(arr)

def _is_multiple(us, mult: int):
    """Whether us is a multiple of mult"""
    return us % mult == 0

def _maybe_add_count(base: str, count: float):
    """If count is greater than 1, add it to the base offset string"""
    if count > 1:
        return f"{count}{base}"
    return base

def month_anchor_check(dates):
    """Return the monthly offset string.

    Return "cs" if all dates are the first days of the month,
    "ce" if all dates are the last day of the month,
    None otherwise.

    Replicated pandas._libs.tslibs.resolution.month_position_check
    but without business offset handling.
    """
    first_day = all(date.day == 1 for date in dates)
    last_day = all(date.day == _days_in_month(date) for date in dates)

    if first_day:
        return "cs"
    elif last_day:
        return "ce"
    else:
        return None
