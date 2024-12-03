"""DatetimeIndex analog for cftime.datetime objects"""
from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import _STANDARD_CALENDARS, cftime_to_nptime, infer_calendar_name
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
try:
    import cftime
except ImportError:
    cftime = None
if TYPE_CHECKING:
    from xarray.coding.cftime_offsets import BaseCFTimeOffset
    from xarray.core.types import Self
CFTIME_REPR_LENGTH = 19
ITEMS_IN_REPR_MAX_ELSE_ELLIPSIS = 100
REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END = 10
OUT_OF_BOUNDS_TIMEDELTA_ERRORS: tuple[type[Exception], ...]
try:
    OUT_OF_BOUNDS_TIMEDELTA_ERRORS = (pd.errors.OutOfBoundsTimedelta, OverflowError)
except AttributeError:
    OUT_OF_BOUNDS_TIMEDELTA_ERRORS = (OverflowError,)
_BASIC_PATTERN = build_pattern(date_sep='', time_sep='')
_EXTENDED_PATTERN = build_pattern()
_CFTIME_PATTERN = build_pattern(datetime_sep=' ')
_PATTERNS = [_BASIC_PATTERN, _EXTENDED_PATTERN, _CFTIME_PATTERN]

def _parsed_string_to_bounds(date_type, resolution, parsed):
    """Generalization of
    pandas.tseries.index.DatetimeIndex._parsed_string_to_bounds
    for use with non-standard calendars and cftime.datetime
    objects.
    """
    if resolution == 'year':
        return (
            date_type(*parsed[:1], 1, 1),
            date_type(*parsed[:1], 12, 31, 23, 59, 59, 999999)
        )
    elif resolution == 'month':
        year, month = parsed[:2]
        last_day = date_type(year, month, 1).days_in_month
        return (
            date_type(year, month, 1),
            date_type(year, month, last_day, 23, 59, 59, 999999)
        )
    elif resolution == 'day':
        return (
            date_type(*parsed[:3]),
            date_type(*parsed[:3], 23, 59, 59, 999999)
        )
    elif resolution == 'hour':
        return (
            date_type(*parsed[:4]),
            date_type(*parsed[:4], 59, 59, 999999)
        )
    elif resolution == 'minute':
        return (
            date_type(*parsed[:5]),
            date_type(*parsed[:5], 59, 999999)
        )
    elif resolution == 'second':
        return (
            date_type(*parsed[:6]),
            date_type(*parsed[:6], 999999)
        )
    else:
        return (date_type(*parsed), date_type(*parsed))

def get_date_field(datetimes, field):
    """Adapted from pandas.tslib.get_date_field"""
    if field == 'dayofyear':
        return np.array([getattr(dt, 'dayofyr') for dt in datetimes])
    elif field == 'dayofweek':
        return np.array([getattr(dt, 'dayofwk') for dt in datetimes])
    elif field == 'days_in_month':
        return np.array([getattr(dt, 'daysinmonth') for dt in datetimes])
    else:
        return np.array([getattr(dt, field) for dt in datetimes])

def _field_accessor(name, docstring=None, min_cftime_version='0.0'):
    """Adapted from pandas.tseries.index._field_accessor"""
    def accessor(self):
        if cftime is None:
            raise ImportError("cftime is required for accessing calendar fields")
        if Version(cftime.__version__) < Version(min_cftime_version):
            raise AttributeError(f"{name} requires cftime version {min_cftime_version} or later")
        return get_date_field(self._data, name)

    accessor.__name__ = name
    accessor.__doc__ = docstring
    return property(accessor)

def format_row(times, indent=0, separator=', ', row_end=',\n'):
    """Format a single row from format_times."""
    formatted = separator.join(str(t) for t in times)
    return f"{' ' * indent}{formatted}{row_end}"

def format_times(index, max_width, offset, separator=', ', first_row_offset=0, intermediate_row_end=',\n', last_row_end=''):
    """Format values of cftimeindex as pd.Index."""
    result = []
    current_row = []
    current_width = first_row_offset

    for t in index:
        if current_width + len(str(t)) + len(separator) > max_width:
            result.append(format_row(current_row, offset, separator, intermediate_row_end))
            current_row = []
            current_width = offset

        current_row.append(t)
        current_width += len(str(t)) + len(separator)

    if current_row:
        result.append(format_row(current_row, offset, separator, last_row_end))

    return ''.join(result)

def format_attrs(index, separator=', '):
    """Format attributes of CFTimeIndex for __repr__."""
    attrs = []
    if index.name is not None:
        attrs.append(f"name='{index.name}'")
    attrs.append(f"dtype='{index.dtype}'")
    attrs.append(f"length={len(index)}")
    attrs.append(f"calendar='{index.calendar}'")
    if index.freq is not None:
        attrs.append(f"freq='{index.freq}'")
    return separator.join(attrs)

class CFTimeIndex(pd.Index):
    """Custom Index for working with CF calendars and dates

    All elements of a CFTimeIndex must be cftime.datetime objects.

    Parameters
    ----------
    data : array or CFTimeIndex
        Sequence of cftime.datetime objects to use in index
    name : str, default: None
        Name of the resulting index

    See Also
    --------
    cftime_range
    """
    year = _field_accessor('year', 'The year of the datetime')
    month = _field_accessor('month', 'The month of the datetime')
    day = _field_accessor('day', 'The days of the datetime')
    hour = _field_accessor('hour', 'The hours of the datetime')
    minute = _field_accessor('minute', 'The minutes of the datetime')
    second = _field_accessor('second', 'The seconds of the datetime')
    microsecond = _field_accessor('microsecond', 'The microseconds of the datetime')
    dayofyear = _field_accessor('dayofyr', 'The ordinal day of year of the datetime', '1.0.2.1')
    dayofweek = _field_accessor('dayofwk', 'The day of week of the datetime', '1.0.2.1')
    days_in_month = _field_accessor('daysinmonth', 'The number of days in the month of the datetime', '1.1.0.0')
    date_type = property(get_date_type)

    def __new__(cls, data, name=None, **kwargs):
        assert_all_valid_date_type(data)
        if name is None and hasattr(data, 'name'):
            name = data.name
        result = object.__new__(cls)
        result._data = np.array(data, dtype='O')
        result.name = name
        result._cache = {}
        return result

    def __repr__(self):
        """
        Return a string representation for this object.
        """
        klass_name = type(self).__name__
        display_width = OPTIONS['display_width']
        offset = len(klass_name) + 2
        if len(self) <= ITEMS_IN_REPR_MAX_ELSE_ELLIPSIS:
            datastr = format_times(self.values, display_width, offset=offset, first_row_offset=0)
        else:
            front_str = format_times(self.values[:REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END], display_width, offset=offset, first_row_offset=0, last_row_end=',')
            end_str = format_times(self.values[-REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END:], display_width, offset=offset, first_row_offset=offset)
            datastr = '\n'.join([front_str, f'{' ' * offset}...', end_str])
        attrs_str = format_attrs(self)
        full_repr_str = f'{klass_name}([{datastr}], {attrs_str})'
        if len(full_repr_str) > display_width:
            if len(attrs_str) >= display_width - offset:
                attrs_str = attrs_str.replace(',', f',\n{' ' * (offset - 2)}')
            full_repr_str = f'{klass_name}([{datastr}],\n{' ' * (offset - 1)}{attrs_str})'
        return full_repr_str

    def _partial_date_slice(self, resolution, parsed):
        """Adapted from
        pandas.tseries.index.DatetimeIndex._partial_date_slice

        Note that when using a CFTimeIndex, if a partial-date selection
        returns a single element, it will never be converted to a scalar
        coordinate; this is in slight contrast to the behavior when using
        a DatetimeIndex, which sometimes will return a DataArray with a scalar
        coordinate depending on the resolution of the datetimes used in
        defining the index.  For example:

        >>> from cftime import DatetimeNoLeap
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[DatetimeNoLeap(2001, 1, 1), DatetimeNoLeap(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray (time: 1)> Size: 8B
        array([1])
        Coordinates:
          * time     (time) object 8B 2001-01-01 00:00:00
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[pd.Timestamp(2001, 1, 1), pd.Timestamp(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray ()> Size: 8B
        array(1)
        Coordinates:
            time     datetime64[ns] 8B 2001-01-01
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[pd.Timestamp(2001, 1, 1, 1), pd.Timestamp(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray (time: 1)> Size: 8B
        array([1])
        Coordinates:
          * time     (time) datetime64[ns] 8B 2001-01-01T01:00:00
        """
        pass

    def _get_string_slice(self, key):
        """Adapted from pandas.tseries.index.DatetimeIndex._get_string_slice"""
        pass

    def _get_nearest_indexer(self, target, limit, tolerance):
        """Adapted from pandas.Index._get_nearest_indexer"""
        pass

    def _filter_indexer_tolerance(self, target, indexer, tolerance):
        """Adapted from pandas.Index._filter_indexer_tolerance"""
        pass

    def get_loc(self, key):
        """Adapted from pandas.tseries.index.DatetimeIndex.get_loc"""
        pass

    def _maybe_cast_slice_bound(self, label, side):
        """Adapted from
        pandas.tseries.index.DatetimeIndex._maybe_cast_slice_bound
        """
        pass

    def get_value(self, series, key):
        """Adapted from pandas.tseries.index.DatetimeIndex.get_value"""
        pass

    def __contains__(self, key: Any) -> bool:
        """Adapted from
        pandas.tseries.base.DatetimeIndexOpsMixin.__contains__"""
        try:
            result = self.get_loc(key)
            return is_scalar(result) or isinstance(result, slice) or (isinstance(result, np.ndarray) and result.size > 0)
        except (KeyError, TypeError, ValueError):
            return False

    def contains(self, key: Any) -> bool:
        """Needed for .loc based partial-string indexing"""
        pass

    def shift(self, periods: int | float, freq: str | timedelta | BaseCFTimeOffset | None=None) -> Self:
        """Shift the CFTimeIndex a multiple of the given frequency.

        See the documentation for :py:func:`~xarray.cftime_range` for a
        complete listing of valid frequency strings.

        Parameters
        ----------
        periods : int, float if freq of days or below
            Periods to shift by
        freq : str, datetime.timedelta or BaseCFTimeOffset
            A frequency string or datetime.timedelta object to shift by

        Returns
        -------
        CFTimeIndex

        See Also
        --------
        pandas.DatetimeIndex.shift

        Examples
        --------
        >>> index = xr.cftime_range("2000", periods=1, freq="ME")
        >>> index
        CFTimeIndex([2000-01-31 00:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        >>> index.shift(1, "ME")
        CFTimeIndex([2000-02-29 00:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        >>> index.shift(1.5, "D")
        CFTimeIndex([2000-02-01 12:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        """
        pass

    def __add__(self, other) -> Self:
        if isinstance(other, pd.TimedeltaIndex):
            other = other.to_pytimedelta()
        return type(self)(np.array(self) + other)

    def __radd__(self, other) -> Self:
        if isinstance(other, pd.TimedeltaIndex):
            other = other.to_pytimedelta()
        return type(self)(other + np.array(self))

    def __sub__(self, other):
        if _contains_datetime_timedeltas(other):
            return type(self)(np.array(self) - other)
        if isinstance(other, pd.TimedeltaIndex):
            return type(self)(np.array(self) - other.to_pytimedelta())
        if _contains_cftime_datetimes(np.array(other)):
            try:
                return pd.TimedeltaIndex(np.array(self) - np.array(other))
            except OUT_OF_BOUNDS_TIMEDELTA_ERRORS:
                raise ValueError('The time difference exceeds the range of values that can be expressed at the nanosecond resolution.')
        return NotImplemented

    def __rsub__(self, other):
        try:
            return pd.TimedeltaIndex(other - np.array(self))
        except OUT_OF_BOUNDS_TIMEDELTA_ERRORS:
            raise ValueError('The time difference exceeds the range of values that can be expressed at the nanosecond resolution.')

    def to_datetimeindex(self, unsafe=False):
        """If possible, convert this index to a pandas.DatetimeIndex.

        Parameters
        ----------
        unsafe : bool
            Flag to turn off warning when converting from a CFTimeIndex with
            a non-standard calendar to a DatetimeIndex (default ``False``).

        Returns
        -------
        pandas.DatetimeIndex

        Raises
        ------
        ValueError
            If the CFTimeIndex contains dates that are not possible in the
            standard calendar or outside the nanosecond-precision range.

        Warns
        -----
        RuntimeWarning
            If converting from a non-standard calendar to a DatetimeIndex.

        Warnings
        --------
        Note that for non-standard calendars, this will change the calendar
        type of the index.  In that case the result of this method should be
        used with caution.

        Examples
        --------
        >>> times = xr.cftime_range("2000", periods=2, calendar="gregorian")
        >>> times
        CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00],
                    dtype='object', length=2, calendar='standard', freq=None)
        >>> times.to_datetimeindex()
        DatetimeIndex(['2000-01-01', '2000-01-02'], dtype='datetime64[ns]', freq=None)
        """
        pass

    def strftime(self, date_format):
        """
        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%Y-%m-%d")

        Returns
        -------
        pandas.Index
            Index of formatted strings

        Examples
        --------
        >>> rng = xr.cftime_range(
        ...     start="2000", periods=5, freq="2MS", calendar="noleap"
        ... )
        >>> rng.strftime("%B %d, %Y, %r")
        Index(['January 01, 2000, 12:00:00 AM', 'March 01, 2000, 12:00:00 AM',
               'May 01, 2000, 12:00:00 AM', 'July 01, 2000, 12:00:00 AM',
               'September 01, 2000, 12:00:00 AM'],
              dtype='object')
        """
        pass

    @property
    def asi8(self):
        """Convert to integers with units of microseconds since 1970-01-01."""
        pass

    @property
    def calendar(self):
        """The calendar used by the datetimes in the index."""
        pass

    @property
    def freq(self):
        """The frequency used by the dates in the index."""
        pass

    def _round_via_method(self, freq, method):
        """Round dates using a specified method."""
        pass

    def floor(self, freq):
        """Round dates down to fixed frequency.

        Parameters
        ----------
        freq : str
            The frequency level to round the index to.  Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).  See `frequency
            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for a list of possible values.

        Returns
        -------
        CFTimeIndex
        """
        pass

    def ceil(self, freq):
        """Round dates up to fixed frequency.

        Parameters
        ----------
        freq : str
            The frequency level to round the index to.  Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).  See `frequency
            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for a list of possible values.

        Returns
        -------
        CFTimeIndex
        """
        pass

    def round(self, freq):
        """Round dates to a fixed frequency.

        Parameters
        ----------
        freq : str
            The frequency level to round the index to.  Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).  See `frequency
            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for a list of possible values.

        Returns
        -------
        CFTimeIndex
        """
        pass

def _parse_array_of_cftime_strings(strings, date_type):
    """Create a numpy array from an array of strings.

    For use in generating dates from strings for use with interp.  Assumes the
    array is either 0-dimensional or 1-dimensional.

    Parameters
    ----------
    strings : array of strings
        Strings to convert to dates
    date_type : cftime.datetime type
        Calendar type to use for dates

    Returns
    -------
    np.array
    """
    pass

def _contains_datetime_timedeltas(array):
    """Check if an input array contains datetime.timedelta objects."""
    pass

def _cftimeindex_from_i8(values, date_type, name):
    """Construct a CFTimeIndex from an array of integers.

    Parameters
    ----------
    values : np.array
        Integers representing microseconds since 1970-01-01.
    date_type : cftime.datetime
        Type of date for the index.
    name : str
        Name of the index.

    Returns
    -------
    CFTimeIndex
    """
    pass

def _total_microseconds(delta):
    """Compute the total number of microseconds of a datetime.timedelta.

    Parameters
    ----------
    delta : datetime.timedelta
        Input timedelta.

    Returns
    -------
    int
    """
    pass

def _floor_int(values, unit):
    """Copied from pandas."""
    pass

def _ceil_int(values, unit):
    """Copied from pandas."""
    pass

def _round_to_nearest_half_even(values, unit):
    """Copied from pandas."""
    pass
