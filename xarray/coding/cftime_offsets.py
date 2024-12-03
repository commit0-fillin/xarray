"""Time offset classes for use with cftime.datetime objects"""
from __future__ import annotations
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import _is_standard_calendar, _should_cftime_be_used, convert_time_or_go_back, format_cftime_datetime
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import NoDefault, count_not_none, nanosecond_precision_timestamp, no_default
from xarray.core.utils import emit_user_level_warning
try:
    import cftime
except ImportError:
    cftime = None
if TYPE_CHECKING:
    from xarray.core.types import InclusiveOptions, Self, SideOptions, TypeAlias
DayOption: TypeAlias = Literal['start', 'end']

def get_date_type(calendar, use_cftime=True):
    """Return the cftime date type for a given calendar name."""
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    
    calendar = calendar.lower()
    if calendar in {'standard', 'gregorian'}:
        return cftime.DatetimeGregorian if use_cftime else datetime
    elif calendar == 'proleptic_gregorian':
        return cftime.DatetimeProlepticGregorian
    elif calendar in {'noleap', '365_day'}:
        return cftime.DatetimeNoLeap
    elif calendar in {'all_leap', '366_day'}:
        return cftime.DatetimeAllLeap
    elif calendar == '360_day':
        return cftime.Datetime360Day
    elif calendar == 'julian':
        return cftime.DatetimeJulian
    else:
        raise ValueError(f"Unsupported calendar: {calendar}")

class BaseCFTimeOffset:
    _freq: ClassVar[str | None] = None
    _day_option: ClassVar[DayOption | None] = None
    n: int

    def __init__(self, n: int=1) -> None:
        if not isinstance(n, int):
            raise TypeError(f"The provided multiple 'n' must be an integer. Instead a value of type {type(n)!r} was provided.")
        self.n = n

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseCFTimeOffset):
            return NotImplemented
        return self.n == other.n and self.rule_code() == other.rule_code()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __add__(self, other):
        return self.__apply__(other)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract a cftime.datetime from a time offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            return NotImplemented

    def __mul__(self, other: int) -> Self:
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(n=other * self.n)

    def __neg__(self) -> Self:
        return self * -1

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, BaseCFTimeOffset) and type(self) != type(other):
            raise TypeError('Cannot subtract cftime offsets of differing types')
        return -self + other

    def __apply__(self, other):
        return NotImplemented

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        if self.n == 1:
            return date == self.__apply__(self.__apply__(date) - timedelta(microseconds=1))
        else:
            return False

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}>'

    def __repr__(self):
        return str(self)

class Tick(BaseCFTimeOffset):

    def __mul__(self, other: int | float) -> Tick:
        if not isinstance(other, (int, float)):
            return NotImplemented
        if isinstance(other, float):
            n = other * self.n
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return type(self)(n=other * self.n)

    def as_timedelta(self) -> timedelta:
        """All Tick subclasses must implement an as_timedelta method."""
        raise NotImplementedError("Subclasses must implement this method")

def _get_day_of_month(other, day_option: DayOption) -> int:
    """Find the day in `other`'s month that satisfies a BaseCFTimeOffset's
    onOffset policy, as described by the `day_option` argument.

    Parameters
    ----------
    other : cftime.datetime
    day_option : 'start', 'end'
        'start': returns 1
        'end': returns last day of the month

    Returns
    -------
    day_of_month : int

    """
    if day_option == 'start':
        return 1
    elif day_option == 'end':
        return _days_in_month(other)
    else:
        raise ValueError(f"Invalid day_option: {day_option}")

def _days_in_month(date):
    """The number of days in the month of the given date"""
    if isinstance(date, datetime):
        return (date.replace(day=1) + timedelta(days=32)).replace(day=1).day - 1
    else:
        # For cftime.datetime objects
        year = date.year
        month = date.month
        if month == 12:
            next_month = date.replace(year=year + 1, month=1, day=1)
        else:
            next_month = date.replace(month=month + 1, day=1)
        return (next_month - date.replace(day=1)).days

def _adjust_n_months(other_day, n, reference_day):
    """Adjust the number of times a monthly offset is applied based
    on the day of a given date, and the reference day provided.
    """
    if n > 0 and other_day < reference_day:
        return n - 1
    elif n <= 0 and other_day > reference_day:
        return n + 1
    return n

def _adjust_n_years(other, n, month, reference_day):
    """Adjust the number of times an annual offset is applied based on
    another date, and the reference day provided"""
    if n > 0:
        if other.month < month or (other.month == month and other.day < reference_day):
            return n - 1
    elif n < 0:
        if other.month > month or (other.month == month and other.day > reference_day):
            return n + 1
    return n

def _shift_month(date, months, day_option: DayOption='start'):
    """Shift the date to a month start or end a given number of months away."""
    year = date.year + (date.month + months - 1) // 12
    month = (date.month + months - 1) % 12 + 1
    if day_option == 'start':
        return date.replace(year=year, month=month, day=1)
    elif day_option == 'end':
        return date.replace(year=year, month=month, day=_days_in_month(date.replace(year=year, month=month)))
    else:
        raise ValueError(f"Invalid day_option: {day_option}")

def roll_qtrday(other, n: int, month: int, day_option: DayOption, modby: int=3) -> int:
    """Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : cftime.datetime
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_option : 'start', 'end'
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    _get_day_of_month : Find the day in a month provided an offset.
    """
    months_since = other.month % modby - month % modby
    if n > 0 and months_since < 0:
        n -= 1
    elif n <= 0 and months_since > 0:
        n += 1
    
    reference_day = _get_day_of_month(other, day_option)
    if n > 0 and other.day < reference_day:
        n -= 1
    elif n <= 0 and other.day > reference_day:
        n += 1
    
    return n

class MonthBegin(BaseCFTimeOffset):
    _freq = 'MS'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, 1)
        return _shift_month(other, n, 'start')

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == 1

class MonthEnd(BaseCFTimeOffset):
    _freq = 'ME'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_month(other, n, 'end')

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date)
_MONTH_ABBREVIATIONS = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

class QuarterOffset(BaseCFTimeOffset):
    """Quarter representation copied off of pandas/tseries/offsets.py"""
    _default_month: ClassVar[int]
    month: int

    def __init__(self, n: int=1, month: int | None=None) -> None:
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        months_since = other.month % 3 - self.month % 3
        qtrs = roll_qtrday(other, self.n, self.month, day_option=self._day_option, modby=3)
        months = qtrs * 3 - months_since
        return _shift_month(other, months, self._day_option)

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        if self._day_option == 'start':
            return (date.month % 3 == self.month % 3) and date.day == 1
        elif self._day_option == 'end':
            return (date.month % 3 == self.month % 3) and date.day == _days_in_month(date)
        else:
            return False

    def __sub__(self, other: Self) -> Self:
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        if type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class QuarterBegin(QuarterOffset):
    _default_month = 3
    _freq = 'QS'
    _day_option = 'start'

    def rollforward(self, date):
        """Roll date forward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        months_to_next = 3 - ((date.month - self.month) % 3)
        return _shift_month(date, months_to_next, 'start')

    def rollback(self, date):
        """Roll date backward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        months_to_prev = ((date.month - self.month) % 3)
        return _shift_month(date, -months_to_prev, 'start')

class QuarterEnd(QuarterOffset):
    _default_month = 3
    _freq = 'QE'
    _day_option = 'end'

    def rollforward(self, date):
        """Roll date forward to nearest end of quarter"""
        if self.onOffset(date):
            return date
        months_to_next = 3 - ((date.month - self.month) % 3)
        return _shift_month(date, months_to_next, 'end')

    def rollback(self, date):
        """Roll date backward to nearest end of quarter"""
        if self.onOffset(date):
            return date
        months_to_prev = ((date.month - self.month) % 3) or 3
        return _shift_month(date, -months_to_prev, 'end')

class YearOffset(BaseCFTimeOffset):
    _default_month: ClassVar[int]
    month: int

    def __init__(self, n: int=1, month: int | None=None) -> None:
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        reference_day = _get_day_of_month(other, self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_month(other, months, self._day_option)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def __str__(self) -> str:
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class YearBegin(YearOffset):
    _freq = 'YS'
    _day_option = 'start'
    _default_month = 1

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.month == self.month and date.day == 1

    def rollforward(self, date):
        """Roll date forward to nearest start of year"""
        if self.onOffset(date):
            return date
        if date.month < self.month or (date.month == self.month and date.day > 1):
            return date.replace(year=date.year + 1, month=self.month, day=1)
        return date.replace(month=self.month, day=1)

    def rollback(self, date):
        """Roll date backward to nearest start of year"""
        if self.onOffset(date):
            return date
        if date.month > self.month or (date.month == self.month and date.day > 1):
            return date.replace(month=self.month, day=1)
        return date.replace(year=date.year - 1, month=self.month, day=1)

class YearEnd(YearOffset):
    _freq = 'YE'
    _day_option = 'end'
    _default_month = 12

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.month == self.month and date.day == _days_in_month(date)

    def rollforward(self, date):
        """Roll date forward to nearest end of year"""
        if self.onOffset(date):
            return date
        if date.month > self.month or (date.month == self.month and date.day == _days_in_month(date)):
            return date.replace(year=date.year + 1, month=self.month, day=_days_in_month(date.replace(year=date.year + 1, month=self.month)))
        return date.replace(month=self.month, day=_days_in_month(date.replace(month=self.month)))

    def rollback(self, date):
        """Roll date backward to nearest end of year"""
        if self.onOffset(date):
            return date
        if date.month < self.month or (date.month == self.month and date.day < _days_in_month(date)):
            return date.replace(year=date.year - 1, month=self.month, day=_days_in_month(date.replace(year=date.year - 1, month=self.month)))
        return date.replace(month=self.month, day=_days_in_month(date.replace(month=self.month)))

class Day(Tick):
    _freq = 'D'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Hour(Tick):
    _freq = 'h'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Minute(Tick):
    _freq = 'min'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Second(Tick):
    _freq = 's'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Millisecond(Tick):
    _freq = 'ms'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Microsecond(Tick):
    _freq = 'us'

    def __apply__(self, other):
        return other + self.as_timedelta()
_FREQUENCIES: Mapping[str, type[BaseCFTimeOffset]] = {'A': YearEnd, 'AS': YearBegin, 'Y': YearEnd, 'YE': YearEnd, 'YS': YearBegin, 'Q': partial(QuarterEnd, month=12), 'QE': partial(QuarterEnd, month=12), 'QS': partial(QuarterBegin, month=1), 'M': MonthEnd, 'ME': MonthEnd, 'MS': MonthBegin, 'D': Day, 'H': Hour, 'h': Hour, 'T': Minute, 'min': Minute, 'S': Second, 's': Second, 'L': Millisecond, 'ms': Millisecond, 'U': Microsecond, 'us': Microsecond, **_generate_anchored_offsets('AS', YearBegin), **_generate_anchored_offsets('A', YearEnd), **_generate_anchored_offsets('YS', YearBegin), **_generate_anchored_offsets('Y', YearEnd), **_generate_anchored_offsets('YE', YearEnd), **_generate_anchored_offsets('QS', QuarterBegin), **_generate_anchored_offsets('Q', QuarterEnd), **_generate_anchored_offsets('QE', QuarterEnd)}
_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = f'^((?P<multiple>[+-]?\\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$'
CFTIME_TICKS = (Day, Hour, Minute, Second)
_DEPRECATED_FREQUENICES: dict[str, str] = {'A': 'YE', 'Y': 'YE', 'AS': 'YS', 'Q': 'QE', 'M': 'ME', 'H': 'h', 'T': 'min', 'S': 's', 'L': 'ms', 'U': 'us', **_generate_anchored_deprecated_frequencies('A', 'YE'), **_generate_anchored_deprecated_frequencies('Y', 'YE'), **_generate_anchored_deprecated_frequencies('AS', 'YS'), **_generate_anchored_deprecated_frequencies('Q', 'QE')}
_DEPRECATION_MESSAGE = '{deprecated_freq!r} is deprecated and will be removed in a future version. Please use {recommended_freq!r} instead of {deprecated_freq!r}.'

def to_offset(freq: BaseCFTimeOffset | str, warn: bool=True) -> BaseCFTimeOffset:
    """Convert a frequency string to the appropriate subclass of
    BaseCFTimeOffset."""
    if isinstance(freq, BaseCFTimeOffset):
        return freq

    if not isinstance(freq, str):
        raise TypeError(f"freq must be a string or BaseCFTimeOffset, not {type(freq)}")

    pattern = re.compile(_PATTERN)
    m = pattern.match(freq)

    if m is None:
        raise ValueError(f"Invalid frequency: {freq}")

    groups = m.groupdict()

    n = int(groups['multiple'] or 1)
    offset_name = groups['freq']

    if offset_name in _DEPRECATED_FREQUENICES and warn:
        recommended_freq = _DEPRECATED_FREQUENICES[offset_name]
        emit_user_level_warning(_DEPRECATION_MESSAGE.format(deprecated_freq=offset_name, recommended_freq=recommended_freq), FutureWarning)

    if offset_name in _FREQUENCIES:
        offset = _FREQUENCIES[offset_name]
        if isinstance(offset, partial):
            return offset(n=n)
        else:
            return offset(n=n)
    else:
        raise ValueError(f"Invalid frequency: {freq}")

def normalize_date(date):
    """Round datetime down to midnight."""
    if isinstance(date, datetime):
        return date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif cftime and isinstance(date, cftime.datetime):
        return date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise TypeError(f"Unsupported date type: {type(date)}")

def _maybe_normalize_date(date, normalize):
    """Round datetime down to midnight if normalize is True."""
    return normalize_date(date) if normalize else date

def _generate_linear_range(start, end, periods):
    """Generate an equally-spaced sequence of cftime.datetime objects between
    and including two dates (whose length equals the number of periods)."""
    if periods == 1:
        return [start]
    
    delta = (end - start) / (periods - 1)
    
    dates = [start + i * delta for i in range(periods)]
    dates[-1] = end  # Ensure the last date is exactly the end date
    
    return dates

def _generate_range(start, end, periods, offset):
    """Generate a regular range of cftime.datetime objects with a
    given time offset.

    Adapted from pandas.tseries.offsets.generate_range (now at
    pandas.core.arrays.datetimes._generate_range).

    Parameters
    ----------
    start : cftime.datetime, or None
        Start of range
    end : cftime.datetime, or None
        End of range
    periods : int, or None
        Number of elements in the sequence
    offset : BaseCFTimeOffset
        An offset class designed for working with cftime.datetime objects

    Returns
    -------
    A generator object
    """
    if start is None and end is None:
        raise ValueError("At least one of start and end must be specified")
    
    if start is not None and end is not None and periods is not None:
        raise ValueError("Either specify periods or start and end, but not all three")
    
    if periods is not None and periods < 1:
        raise ValueError("Periods must be a positive integer")
    
    if start is None:
        start = end - (periods - 1) * offset
    elif end is None:
        end = start + (periods - 1) * offset
    
    current = start
    if offset.n >= 0:
        while current <= end:
            yield current
            current = offset.__apply__(current)
    else:
        while current >= end:
            yield current
            current = offset.__apply__(current)

def _translate_closed_to_inclusive(closed):
    """Follows code added in pandas #43504."""
    if closed is None:
        return None
    if closed == "left":
        return "both"
    if closed == "right":
        return "both"
    raise ValueError(f"Closed must be None, 'left' or 'right', got {closed}")

def _infer_inclusive(closed: NoDefault | SideOptions, inclusive: InclusiveOptions | None) -> InclusiveOptions:
    """Follows code added in pandas #43504."""
    if inclusive is not None:
        return inclusive
    if closed is no_default:
        return "both"
    return _translate_closed_to_inclusive(closed)

def cftime_range(start=None, end=None, periods=None, freq=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard') -> CFTimeIndex:
    """Return a fixed frequency CFTimeIndex."""
    calendar = calendar.lower()
    date_type = get_date_type(calendar)

    if start is not None:
        start = _parse_iso8601_with_reso(start, date_type)
    if end is not None:
        end = _parse_iso8601_with_reso(end, date_type)

    if start is not None and end is not None and start > end:
        raise ValueError("Start date must be before end date.")

    if freq is None:
        freq = 'D'

    offset = to_offset(freq)

    if normalize:
        start = normalize_date(start) if start is not None else None
        end = normalize_date(end) if end is not None else None

    if start is None and end is None:
        if periods is None:
            raise ValueError("Must specify start, end, or periods")
        raise ValueError("Must specify start or end when periods is specified")

    inclusive = _infer_inclusive(closed, inclusive)

    if inclusive is None:
        inclusive = "both"

    if start is None and periods is None:
        periods = 1
    elif end is None and periods is None:
        periods = 1
    elif periods is None:
        periods = len(list(_generate_range(start, end, periods, offset)))

    if inclusive == "left":
        if end is not None:
            end = end - offset
    elif inclusive == "right":
        if start is not None:
            start = start + offset
    elif inclusive == "neither":
        if start is not None:
            start = start + offset
        if end is not None:
            end = end - offset

    dates = list(_generate_range(start, end, periods, offset))

    return CFTimeIndex(dates, name=name)

def date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard', use_cftime=None):
    """Return a fixed frequency datetime index."""
    calendar = calendar.lower()

    if use_cftime is None:
        use_cftime = not _is_standard_calendar(calendar)

    if tz is not None:
        use_cftime = False

    if use_cftime:
        return cftime_range(
            start=start,
            end=end,
            periods=periods,
            freq=freq,
            normalize=normalize,
            name=name,
            closed=closed,
            inclusive=inclusive,
            calendar=calendar,
        )

    if not _is_standard_calendar(calendar):
        raise ValueError(
            f"Calendar '{calendar}' is not supported by pandas. "
            "Set use_cftime=True to use cftime.datetime objects instead."
        )

    return pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=_infer_inclusive(closed, inclusive),
    )

def date_range_like(source, calendar, use_cftime=None):
    """Generate a datetime array with the same frequency, start and end as
    another one, but in a different calendar."""
    if isinstance(source, pd.DatetimeIndex):
        start = source[0].to_pydatetime()
        end = source[-1].to_pydatetime()
        periods = len(source)
        freq = source.freq
    elif isinstance(source, CFTimeIndex):
        start = source[0]
        end = source[-1]
        periods = len(source)
        freq = source.freq
    elif isinstance(source, xr.DataArray):
        if not _contains_datetime_like_objects(source):
            raise ValueError("Source must contain datetime-like values")
        start = source.values[0]
        end = source.values[-1]
        periods = len(source)
        freq = source.attrs.get('freq', infer_freq(source))
    else:
        raise TypeError("Source must be a DataArray, CFTimeIndex, or pd.DatetimeIndex")

    if use_cftime is None:
        use_cftime = not _is_standard_calendar(calendar)

    if use_cftime:
        return cftime_range(start=start, end=end, periods=periods, freq=freq, calendar=calendar)
    else:
        if not _is_standard_calendar(calendar):
            raise ValueError(f"Calendar '{calendar}' is not supported by pandas. Set use_cftime=True to use cftime.datetime objects instead.")
        return pd.date_range(start=start, end=end, periods=periods, freq=freq)
