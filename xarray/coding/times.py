from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Literal, Union, cast
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import SerializationWarning, VariableCoder, lazy_elemwise_func, pop_to, safe_setitem, unpack_for_decoding, unpack_for_encoding
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray, ravel, reshape
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
try:
    import cftime
except ImportError:
    cftime = None
from xarray.core.types import CFCalendar, NPDatetimeUnitOptions, T_DuckArray
T_Name = Union[Hashable, None]
_STANDARD_CALENDARS = {'standard', 'gregorian', 'proleptic_gregorian'}
_NS_PER_TIME_DELTA = {'ns': 1, 'us': int(1000.0), 'ms': int(1000000.0), 's': int(1000000000.0), 'm': int(1000000000.0) * 60, 'h': int(1000000000.0) * 60 * 60, 'D': int(1000000000.0) * 60 * 60 * 24}
_US_PER_TIME_DELTA = {'microseconds': 1, 'milliseconds': 1000, 'seconds': 1000000, 'minutes': 60 * 1000000, 'hours': 60 * 60 * 1000000, 'days': 24 * 60 * 60 * 1000000}
_NETCDF_TIME_UNITS_CFTIME = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']
_NETCDF_TIME_UNITS_NUMPY = _NETCDF_TIME_UNITS_CFTIME + ['nanoseconds']
TIME_UNITS = frozenset(['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds'])

def decode_cf_datetime(num_dates, units: str, calendar: str | None=None, use_cftime: bool | None=None) -> np.ndarray:
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than cftime.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Note that time unit in `units` must not be smaller than microseconds and
    not larger than days.

    See Also
    --------
    cftime.num2date
    """
    num_dates = np.asarray(num_dates)
    units_split = units.split()
    if len(units_split) != 3 or units_split[1] != "since":
        raise ValueError(f"Invalid units: {units}")
    
    unit, _, reference_date = units_split
    
    if unit not in TIME_UNITS:
        raise ValueError(f"Invalid time unit: {unit}")
    
    if calendar is None or calendar.lower() in _STANDARD_CALENDARS:
        try:
            reference = pd.Timestamp(reference_date)
            if unit == 'days':
                delta = pd.Timedelta(days=1)
            elif unit == 'hours':
                delta = pd.Timedelta(hours=1)
            elif unit == 'minutes':
                delta = pd.Timedelta(minutes=1)
            elif unit == 'seconds':
                delta = pd.Timedelta(seconds=1)
            elif unit == 'milliseconds':
                delta = pd.Timedelta(milliseconds=1)
            elif unit == 'microseconds':
                delta = pd.Timedelta(microseconds=1)
            
            dates = reference + num_dates * delta
            return dates.values
        except (OutOfBoundsDatetime, OverflowError):
            use_cftime = True
    
    if use_cftime or cftime is None:
        import cftime
        return cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
    else:
        raise ValueError("Unable to decode dates without cftime for non-standard calendars")

def decode_cf_timedelta(num_timedeltas, units: str) -> np.ndarray:
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = np.asarray(num_timedeltas)
    if units not in TIME_UNITS:
        raise ValueError(f"Invalid time unit: {units}")
    
    if units == 'nanoseconds':
        return num_timedeltas.astype('timedelta64[ns]')
    elif units == 'microseconds':
        return (num_timedeltas * 1000).astype('timedelta64[ns]')
    elif units == 'milliseconds':
        return (num_timedeltas * 1_000_000).astype('timedelta64[ns]')
    elif units == 'seconds':
        return (num_timedeltas * 1_000_000_000).astype('timedelta64[ns]')
    elif units == 'minutes':
        return (num_timedeltas * 60 * 1_000_000_000).astype('timedelta64[ns]')
    elif units == 'hours':
        return (num_timedeltas * 3600 * 1_000_000_000).astype('timedelta64[ns]')
    elif units == 'days':
        return (num_timedeltas * 86400 * 1_000_000_000).astype('timedelta64[ns]')

def infer_calendar_name(dates) -> CFCalendar:
    """Given an array of datetimes, infer the CF calendar name"""
    if isinstance(dates, (pd.Timestamp, np.datetime64)):
        return 'proleptic_gregorian'
    elif cftime and isinstance(dates, cftime.datetime):
        return dates.calendar
    
    sample = dates[0] if isinstance(dates, (list, np.ndarray)) else dates
    if isinstance(sample, (pd.Timestamp, np.datetime64)):
        return 'proleptic_gregorian'
    elif cftime and isinstance(sample, cftime.datetime):
        return sample.calendar
    else:
        raise ValueError("Unable to infer calendar name from input dates")

def infer_datetime_units(dates) -> str:
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    dates = pd.to_datetime(dates)
    reference_date = dates[0]
    
    if len(dates) == 1:
        return f"seconds since {reference_date.isoformat()}"
    
    deltas = np.diff(dates)
    unique_deltas = np.unique(deltas)
    
    for unit in ['days', 'hours', 'minutes', 'seconds']:
        if np.all(unique_deltas % np.timedelta64(1, unit[0].upper()) == np.timedelta64(0)):
            return f"{unit} since {reference_date.isoformat()}"
    
    return f"seconds since {reference_date.isoformat()}"

def format_cftime_datetime(date) -> str:
    """Converts a cftime.datetime object to a string with the format:
    YYYY-MM-DD HH:MM:SS.UUUUUU
    """
    if not cftime or not isinstance(date, cftime.datetime):
        raise ValueError("Input must be a cftime.datetime object")
    
    return f"{date.year:04d}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}.{date.microsecond:06d}"

def infer_timedelta_units(deltas) -> str:
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    deltas = pd.to_timedelta(deltas)
    unique_deltas = np.unique(deltas)
    
    for unit in ['days', 'hours', 'minutes', 'seconds']:
        if np.all(unique_deltas % pd.Timedelta(1, unit) == pd.Timedelta(0)):
            return unit
    
    return 'seconds'

def cftime_to_nptime(times, raise_on_invalid: bool=True) -> np.ndarray:
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size

    If raise_on_invalid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.NaT."""
    if not cftime:
        raise ImportError("cftime is required for this function")
    
    times = np.asarray(times)
    result = np.empty(times.shape, dtype='datetime64[ns]')
    
    for i, t in np.ndenumerate(times):
        try:
            result[i] = np.datetime64(pd.Timestamp(t.isoformat()).to_numpy())
        except (ValueError, OutOfBoundsDatetime):
            if raise_on_invalid:
                raise ValueError(f"Invalid date: {t}")
            result[i] = np.datetime64('NaT')
    
    return result

def convert_times(times, date_type, raise_on_invalid: bool=True) -> np.ndarray:
    """Given an array of datetimes, return the same dates in another cftime or numpy date type.

    Useful to convert between calendars in numpy and cftime or between cftime calendars.

    If raise_on_valid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.nan for cftime types and np.NaT for np.datetime64.
    """
    times = np.asarray(times)
    result = np.empty(times.shape, dtype=object)
    
    for i, t in np.ndenumerate(times):
        try:
            if isinstance(date_type, str) and date_type == 'datetime64':
                result[i] = np.datetime64(pd.Timestamp(t).to_numpy())
            elif cftime and issubclass(date_type, cftime.datetime):
                result[i] = date_type(*t.timetuple()[:6])
            else:
                raise ValueError(f"Unsupported date_type: {date_type}")
        except (ValueError, OutOfBoundsDatetime):
            if raise_on_invalid:
                raise ValueError(f"Invalid date: {t}")
            result[i] = np.datetime64('NaT') if date_type == 'datetime64' else np.nan
    
    return result

def convert_time_or_go_back(date, date_type):
    """Convert a single date to a new date_type (cftime.datetime or pd.Timestamp).

    If the new date is invalid, it goes back a day and tries again. If it is still
    invalid, goes back a second day.

    This is meant to convert end-of-month dates into a new calendar.
    """
    for days_back in range(3):
        try:
            if issubclass(date_type, pd.Timestamp):
                return pd.Timestamp(date.year, date.month, date.day - days_back)
            elif cftime and issubclass(date_type, cftime.datetime):
                return date_type(date.year, date.month, date.day - days_back)
            else:
                raise ValueError(f"Unsupported date_type: {date_type}")
        except (ValueError, OutOfBoundsDatetime):
            if days_back == 2:
                raise ValueError(f"Unable to convert date: {date}")

def _should_cftime_be_used(source, target_calendar: str, use_cftime: bool | None) -> bool:
    """Return whether conversion of the source to the target calendar should
    result in a cftime-backed array.

    Source is a 1D datetime array, target_cal a string (calendar name) and
    use_cftime is a boolean or None. If use_cftime is None, this returns True
    if the source's range and target calendar are convertible to np.datetime64 objects.
    """
    if use_cftime is not None:
        return use_cftime
    
    if target_calendar.lower() not in _STANDARD_CALENDARS:
        return True
    
    try:
        min_date = np.datetime64(source.min())
        max_date = np.datetime64(source.max())
        return False
    except (ValueError, TypeError):
        return True

def _encode_datetime_with_cftime(dates, units: str, calendar: str) -> np.ndarray:
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    if not cftime:
        raise ImportError("cftime is required for this function")
    
    dates = np.asarray(dates)
    if dates.ndim == 0:
        dates = np.array([dates])
    
    calendar = calendar.lower()
    date_type = cftime.date2num
    
    if calendar == 'standard' or calendar == 'gregorian':
        date_type = cftime.DatetimeGregorian
    elif calendar == 'proleptic_gregorian':
        date_type = cftime.DatetimeProlepticGregorian
    
    encoded = []
    for date in dates.flat:
        if isinstance(date, (cftime.datetime, pd.Timestamp, np.datetime64)):
            date = date_type(date.year, date.month, date.day, 
                             date.hour, date.minute, date.second, 
                             date.microsecond)
        encoded.append(cftime.date2num(date, units=units, calendar=calendar))
    
    return np.array(encoded).reshape(dates.shape)

def encode_cf_datetime(dates: T_DuckArray, units: str | None=None, calendar: str | None=None, dtype: np.dtype | None=None) -> tuple[T_DuckArray, str, str]:
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF compliant time variable.

    Unlike `date2num`, this function can handle datetime64 arrays.

    See Also
    --------
    cftime.date2num
    """
    dates = np.asarray(dates)
    
    if calendar is None:
        calendar = infer_calendar_name(dates)
    
    if units is None:
        units = infer_datetime_units(dates)
    
    if dtype is None:
        dtype = np.dtype('float64')
    
    if is_np_datetime_like(dates):
        num = encode_np_datetime(dates, units)
    else:
        num = _encode_datetime_with_cftime(dates, units, calendar)
    
    num = np.array(num, dtype=dtype)
    
    return num, units, calendar

class CFDatetimeCoder(VariableCoder):

    def __init__(self, use_cftime: bool | None=None) -> None:
        self.use_cftime = use_cftime
        super().__init__()

class CFTimedeltaCoder(VariableCoder):
    pass
