"""Resampling for CFTimeIndex. Does not support non-integer freq."""
from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import BaseCFTimeOffset, MonthEnd, QuarterEnd, Tick, YearEnd, cftime_range, normalize_date, to_offset
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
if typing.TYPE_CHECKING:
    from xarray.core.types import CFTimeDatetime

class CFTimeGrouper:
    """This is a simple container for the grouping parameters that implements a
    single method, the only one required for resampling in xarray.  It cannot
    be used in a call to groupby like a pandas.Grouper object can."""
    freq: BaseCFTimeOffset
    closed: SideOptions
    label: SideOptions
    loffset: str | datetime.timedelta | BaseCFTimeOffset | None
    origin: str | CFTimeDatetime
    offset: datetime.timedelta | None

    def __init__(self, freq: str | BaseCFTimeOffset, closed: SideOptions | None=None, label: SideOptions | None=None, origin: str | CFTimeDatetime='start_day', offset: str | datetime.timedelta | BaseCFTimeOffset | None=None):
        self.freq = to_offset(freq)
        self.origin = origin
        if isinstance(self.freq, (MonthEnd, QuarterEnd, YearEnd)):
            if closed is None:
                self.closed = 'right'
            else:
                self.closed = closed
            if label is None:
                self.label = 'right'
            else:
                self.label = label
        elif self.origin in ['end', 'end_day']:
            if closed is None:
                self.closed = 'right'
            else:
                self.closed = closed
            if label is None:
                self.label = 'right'
            else:
                self.label = label
        else:
            if closed is None:
                self.closed = 'left'
            else:
                self.closed = closed
            if label is None:
                self.label = 'left'
            else:
                self.label = label
        if offset is not None:
            try:
                self.offset = _convert_offset_to_timedelta(offset)
            except (ValueError, TypeError) as error:
                raise ValueError(f'offset must be a datetime.timedelta object or an offset string that can be converted to a timedelta. Got {type(offset)} instead.') from error
        else:
            self.offset = None

    def first_items(self, index: CFTimeIndex):
        """Meant to reproduce the results of the following

        grouper = pandas.Grouper(...)
        first_items = pd.Series(np.arange(len(index)),
                                index).groupby(grouper).first()

        with index being a CFTimeIndex instead of a DatetimeIndex.
        """
        bins = _get_time_bins(index, self.freq, self.closed, self.label, self.origin, self.offset)
        labels = bins[1:]
        
        grouper = pd.cut(index, bins, labels=labels, include_lowest=True, right=False)
        series = pd.Series(np.arange(len(index)), index=index)
        return series.groupby(grouper).first()

def _get_time_bins(index: CFTimeIndex, freq: BaseCFTimeOffset, closed: SideOptions, label: SideOptions, origin: str | CFTimeDatetime, offset: datetime.timedelta | None):
    """Obtain the bins and their respective labels for resampling operations.

    Parameters
    ----------
    index : CFTimeIndex
        Index object to be resampled (e.g., CFTimeIndex named 'time').
    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency (e.g., 'MS', '2D', 'H', or '3T' with
        coding.cftime_offsets.to_offset() applied to it).
    closed : 'left' or 'right'
        Which side of bin interval is closed.
        The default is 'left' for all frequency offsets except for 'M' and 'A',
        which have a default of 'right'.
    label : 'left' or 'right'
        Which bin edge label to label bucket with.
        The default is 'left' for all frequency offsets except for 'M' and 'A',
        which have a default of 'right'.
    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'
        The datetime on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.

        If a datetime is not used, these values are also supported:
        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day
    offset : datetime.timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    datetime_bins : CFTimeIndex
        Defines the edge of resampling bins by which original index values will
        be grouped into.
    labels : CFTimeIndex
        Define what the user actually sees the bins labeled as.
    """
    if len(index) == 0:
        return CFTimeIndex([], name=index.name), CFTimeIndex([], name=index.name)

    if isinstance(origin, str):
        if origin == 'epoch':
            origin = index[0].replace(year=1970, month=1, day=1)
        elif origin == 'start':
            origin = index[0]
        elif origin == 'start_day':
            origin = index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        elif origin == 'end':
            origin = index[-1]
        elif origin == 'end_day':
            origin = (index[-1] + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Invalid origin: {origin}")

    if offset:
        origin += offset

    start, end = _get_range_edges(index[0], index[-1], freq, closed, origin, offset)
    datetime_bins = cftime_range(start=start, end=end, freq=freq)

    if label == 'right':
        labels = datetime_bins[1:]
    else:
        labels = datetime_bins[:-1]

    datetime_bins, labels = _adjust_bin_edges(datetime_bins, freq, closed, index, labels)

    return datetime_bins, labels

def _adjust_bin_edges(datetime_bins: CFTimeIndex, freq: BaseCFTimeOffset, closed: SideOptions, index: CFTimeIndex, labels: CFTimeIndex) -> tuple[CFTimeIndex, CFTimeIndex]:
    """This is required for determining the bin edges resampling with
    month end, quarter end, and year end frequencies.

    Consider the following example.  Let's say you want to downsample the
    time series with the following coordinates to month end frequency:

    CFTimeIndex([2000-01-01 12:00:00, 2000-01-31 12:00:00,
                 2000-02-01 12:00:00], dtype='object')

    Without this adjustment, _get_time_bins with month-end frequency will
    return the following index for the bin edges (default closed='right' and
    label='right' in this case):

    CFTimeIndex([1999-12-31 00:00:00, 2000-01-31 00:00:00,
                 2000-02-29 00:00:00], dtype='object')

    If 2000-01-31 is used as a bound for a bin, the value on
    2000-01-31T12:00:00 (at noon on January 31st), will not be included in the
    month of January.  To account for this, pandas adds a day minus one worth
    of microseconds to the bin edges generated by cftime range, so that we do
    bin the value at noon on January 31st in the January bin.  This results in
    an index with bin edges like the following:

    CFTimeIndex([1999-12-31 23:59:59, 2000-01-31 23:59:59,
                 2000-02-29 23:59:59], dtype='object')

    The labels are still:

    CFTimeIndex([2000-01-31 00:00:00, 2000-02-29 00:00:00], dtype='object')
    """
    if isinstance(freq, (MonthEnd, QuarterEnd, YearEnd)):
        if closed == 'right':
            datetime_bins = datetime_bins.shift(1, freq='D') - datetime.timedelta(microseconds=1)
        else:
            datetime_bins = datetime_bins - datetime.timedelta(microseconds=1)
    
    return datetime_bins, labels

def _get_range_edges(first: CFTimeDatetime, last: CFTimeDatetime, freq: BaseCFTimeOffset, closed: SideOptions='left', origin: str | CFTimeDatetime='start_day', offset: datetime.timedelta | None=None):
    """Get the correct starting and ending datetimes for the resampled
    CFTimeIndex range.

    Parameters
    ----------
    first : cftime.datetime
        Uncorrected starting datetime object for resampled CFTimeIndex range.
        Usually the min of the original CFTimeIndex.
    last : cftime.datetime
        Uncorrected ending datetime object for resampled CFTimeIndex range.
        Usually the max of the original CFTimeIndex.
    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency. Contains information on offset type (e.g. Day or 'D') and
        offset magnitude (e.g., n = 3).
    closed : 'left' or 'right'
        Which side of bin interval is closed. Defaults to 'left'.
    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'
        The datetime on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.

        If a datetime is not used, these values are also supported:
        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day
    offset : datetime.timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    first : cftime.datetime
        Corrected starting datetime object for resampled CFTimeIndex range.
    last : cftime.datetime
        Corrected ending datetime object for resampled CFTimeIndex range.
    """
    if isinstance(freq, Tick):
        first, last = _adjust_dates_anchored(first, last, freq, closed, origin, offset)
    else:
        if closed == 'right':
            first = freq.rollback(first)
        else:
            first = freq.rollforward(first)
        last = freq.rollforward(last)

    return first, last

def _adjust_dates_anchored(first: CFTimeDatetime, last: CFTimeDatetime, freq: Tick, closed: SideOptions='right', origin: str | CFTimeDatetime='start_day', offset: datetime.timedelta | None=None):
    """First and last offsets should be calculated from the start day to fix
    an error cause by resampling across multiple days when a one day period is
    not a multiple of the frequency.
    See https://github.com/pandas-dev/pandas/issues/8683

    Parameters
    ----------
    first : cftime.datetime
        A datetime object representing the start of a CFTimeIndex range.
    last : cftime.datetime
        A datetime object representing the end of a CFTimeIndex range.
    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency. Contains information on offset type (e.g. Day or 'D') and
        offset magnitude (e.g., n = 3).
    closed : 'left' or 'right'
        Which side of bin interval is closed. Defaults to 'right'.
    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'
        The datetime on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.

        If a datetime is not used, these values are also supported:
        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day
    offset : datetime.timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    fresult : cftime.datetime
        A datetime object representing the start of a date range that has been
        adjusted to fix resampling errors.
    lresult : cftime.datetime
        A datetime object representing the end of a date range that has been
        adjusted to fix resampling errors.
    """
    if isinstance(origin, str):
        if origin == 'epoch':
            origin = first.replace(year=1970, month=1, day=1)
        elif origin == 'start':
            origin = first
        elif origin == 'start_day':
            origin = first.replace(hour=0, minute=0, second=0, microsecond=0)
        elif origin == 'end':
            origin = last
        elif origin == 'end_day':
            origin = (last + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Invalid origin: {origin}")

    if offset:
        origin += offset

    td = exact_cftime_datetime_difference(origin, first)
    foffset = ((td.total_seconds() % freq.as_timedelta().total_seconds()) * -1) % freq.as_timedelta().total_seconds()
    fresult = first + datetime.timedelta(seconds=foffset)

    td = exact_cftime_datetime_difference(origin, last)
    loffset = freq.as_timedelta().total_seconds() - (td.total_seconds() % freq.as_timedelta().total_seconds())
    lresult = last + datetime.timedelta(seconds=loffset)

    if closed == 'right':
        fresult = fresult + freq.as_timedelta()

    return fresult, lresult

def exact_cftime_datetime_difference(a: CFTimeDatetime, b: CFTimeDatetime):
    """Exact computation of b - a

    Assumes:

        a = a_0 + a_m
        b = b_0 + b_m

    Here a_0, and b_0 represent the input dates rounded
    down to the nearest second, and a_m, and b_m represent
    the remaining microseconds associated with date a and
    date b.

    We can then express the value of b - a as:

        b - a = (b_0 + b_m) - (a_0 + a_m) = b_0 - a_0 + b_m - a_m

    By construction, we know that b_0 - a_0 must be a round number
    of seconds.  Therefore we can take the result of b_0 - a_0 using
    ordinary cftime.datetime arithmetic and round to the nearest
    second.  b_m - a_m is the remainder, in microseconds, and we
    can simply add this to the rounded timedelta.

    Parameters
    ----------
    a : cftime.datetime
        Input datetime
    b : cftime.datetime
        Input datetime

    Returns
    -------
    datetime.timedelta
    """
    a_0 = a.replace(microsecond=0)
    b_0 = b.replace(microsecond=0)
    delta_seconds = (b_0 - a_0).total_seconds()
    delta_microseconds = b.microsecond - a.microsecond
    return datetime.timedelta(seconds=delta_seconds, microseconds=delta_microseconds)
