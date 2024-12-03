from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like, is_np_timedelta_like
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import CFCalendar

def _season_from_months(months):
    """Compute season (DJF, MAM, JJA, SON) from month ordinal"""
    seasons = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
               7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}
    return np.vectorize(seasons.get)(months)

def _access_through_cftimeindex(values, name):
    """Coerce an array of datetime-like values to a CFTimeIndex
    and access requested datetime component
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    index = CFTimeIndex(values)
    return getattr(index, name)

def _access_through_series(values, name):
    """Coerce an array of datetime-like values to a pandas Series and
    access requested datetime component
    """
    import pandas as pd
    series = pd.Series(values)
    return getattr(series.dt, name)

def _get_date_field(values, name, dtype):
    """Indirectly access pandas' libts.get_date_field by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : str
        Name of datetime field to access
    dtype : dtype-like
        dtype for output date field values

    Returns
    -------
    datetime_fields : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if is_duck_dask_array(values):
        import dask.array as da
        return da.map_overlap(_access_through_series, values, name=name, dtype=dtype)
    else:
        return _access_through_series(values, name).astype(dtype)

def _round_through_series_or_index(values, name, freq):
    """Coerce an array of datetime-like values to a pandas Series or xarray
    CFTimeIndex and apply requested rounding
    """
    if contains_cftime_datetimes(values):
        from xarray.coding.cftimeindex import CFTimeIndex
        index = CFTimeIndex(values)
        return getattr(index, name)(freq=freq)
    else:
        import pandas as pd
        series = pd.Series(values)
        return getattr(series.dt, name)(freq=freq)

def _round_field(values, name, freq):
    """Indirectly access rounding functions by wrapping data
    as a Series or CFTimeIndex

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : {"ceil", "floor", "round"}
        Name of rounding function
    freq : str
        a freq string indicating the rounding resolution

    Returns
    -------
    rounded timestamps : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if is_duck_dask_array(values):
        import dask.array as da
        return da.map_overlap(_round_through_series_or_index, values, name=name, freq=freq)
    else:
        return _round_through_series_or_index(values, name, freq)

def _strftime_through_cftimeindex(values, date_format: str):
    """Coerce an array of cftime-like values to a CFTimeIndex
    and access requested datetime component
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    index = CFTimeIndex(values)
    return index.strftime(date_format)

def _strftime_through_series(values, date_format: str):
    """Coerce an array of datetime-like values to a pandas Series and
    apply string formatting
    """
    import pandas as pd
    series = pd.Series(values)
    return series.dt.strftime(date_format)

class TimeAccessor(Generic[T_DataArray]):
    __slots__ = ('_obj',)

    def __init__(self, obj: T_DataArray) -> None:
        self._obj = obj

    def floor(self, freq: str) -> T_DataArray:
        """
        Round timestamps downward to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        floor-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._obj._replace(_data=_round_field(self._obj.data, "floor", freq))

    def ceil(self, freq: str) -> T_DataArray:
        """
        Round timestamps upward to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        ceil-ed timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._obj._replace(_data=_round_field(self._obj.data, "ceil", freq))

    def round(self, freq: str) -> T_DataArray:
        """
        Round timestamps to specified frequency resolution.

        Parameters
        ----------
        freq : str
            a freq string indicating the rounding resolution e.g. "D" for daily resolution

        Returns
        -------
        rounded timestamps : same type as values
            Array-like of datetime fields accessed for each element in values
        """
        return self._obj._replace(_data=_round_field(self._obj.data, "round", freq))

class DatetimeAccessor(TimeAccessor[T_DataArray]):
    """Access datetime fields for DataArrays with datetime-like dtypes.

    Fields can be accessed through the `.dt` attribute
    for applicable DataArrays.

    Examples
    ---------
    >>> dates = pd.date_range(start="2000/01/01", freq="D", periods=10)
    >>> ts = xr.DataArray(dates, dims=("time"))
    >>> ts
    <xarray.DataArray (time: 10)> Size: 80B
    array(['2000-01-01T00:00:00.000000000', '2000-01-02T00:00:00.000000000',
           '2000-01-03T00:00:00.000000000', '2000-01-04T00:00:00.000000000',
           '2000-01-05T00:00:00.000000000', '2000-01-06T00:00:00.000000000',
           '2000-01-07T00:00:00.000000000', '2000-01-08T00:00:00.000000000',
           '2000-01-09T00:00:00.000000000', '2000-01-10T00:00:00.000000000'],
          dtype='datetime64[ns]')
    Coordinates:
      * time     (time) datetime64[ns] 80B 2000-01-01 2000-01-02 ... 2000-01-10
    >>> ts.dt  # doctest: +ELLIPSIS
    <xarray.core.accessor_dt.DatetimeAccessor object at 0x...>
    >>> ts.dt.dayofyear
    <xarray.DataArray 'dayofyear' (time: 10)> Size: 80B
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    Coordinates:
      * time     (time) datetime64[ns] 80B 2000-01-01 2000-01-02 ... 2000-01-10
    >>> ts.dt.quarter
    <xarray.DataArray 'quarter' (time: 10)> Size: 80B
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Coordinates:
      * time     (time) datetime64[ns] 80B 2000-01-01 2000-01-02 ... 2000-01-10

    """

    def strftime(self, date_format: str) -> T_DataArray:
        """
        Return an array of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__

        Parameters
        ----------
        date_format : str
            date format string (e.g. "%Y-%m-%d")

        Returns
        -------
        formatted strings : same type as values
            Array-like of strings formatted for each element in values

        Examples
        --------
        >>> import datetime
        >>> rng = xr.Dataset({"time": datetime.datetime(2000, 1, 1)})
        >>> rng["time"].dt.strftime("%B %d, %Y, %r")
        <xarray.DataArray 'strftime' ()> Size: 8B
        array('January 01, 2000, 12:00:00 AM', dtype=object)
        """
        pass

    def isocalendar(self) -> Dataset:
        """Dataset containing ISO year, week number, and weekday.

        Notes
        -----
        The iso year and weekday differ from the nominal year and weekday.
        """
        pass

    @property
    def year(self) -> T_DataArray:
        """The year of the datetime"""
        pass

    @property
    def month(self) -> T_DataArray:
        """The month as January=1, December=12"""
        pass

    @property
    def day(self) -> T_DataArray:
        """The days of the datetime"""
        pass

    @property
    def hour(self) -> T_DataArray:
        """The hours of the datetime"""
        pass

    @property
    def minute(self) -> T_DataArray:
        """The minutes of the datetime"""
        pass

    @property
    def second(self) -> T_DataArray:
        """The seconds of the datetime"""
        pass

    @property
    def microsecond(self) -> T_DataArray:
        """The microseconds of the datetime"""
        pass

    @property
    def nanosecond(self) -> T_DataArray:
        """The nanoseconds of the datetime"""
        pass

    @property
    def weekofyear(self) -> DataArray:
        """The week ordinal of the year"""
        pass
    week = weekofyear

    @property
    def dayofweek(self) -> T_DataArray:
        """The day of the week with Monday=0, Sunday=6"""
        pass
    weekday = dayofweek

    @property
    def dayofyear(self) -> T_DataArray:
        """The ordinal day of the year"""
        pass

    @property
    def quarter(self) -> T_DataArray:
        """The quarter of the date"""
        pass

    @property
    def days_in_month(self) -> T_DataArray:
        """The number of days in the month"""
        pass
    daysinmonth = days_in_month

    @property
    def season(self) -> T_DataArray:
        """Season of the year"""
        pass

    @property
    def time(self) -> T_DataArray:
        """Timestamps corresponding to datetimes"""
        pass

    @property
    def date(self) -> T_DataArray:
        """Date corresponding to datetimes"""
        pass

    @property
    def is_month_start(self) -> T_DataArray:
        """Indicate whether the date is the first day of the month"""
        pass

    @property
    def is_month_end(self) -> T_DataArray:
        """Indicate whether the date is the last day of the month"""
        pass

    @property
    def is_quarter_start(self) -> T_DataArray:
        """Indicate whether the date is the first day of a quarter"""
        pass

    @property
    def is_quarter_end(self) -> T_DataArray:
        """Indicate whether the date is the last day of a quarter"""
        pass

    @property
    def is_year_start(self) -> T_DataArray:
        """Indicate whether the date is the first day of a year"""
        pass

    @property
    def is_year_end(self) -> T_DataArray:
        """Indicate whether the date is the last day of the year"""
        pass

    @property
    def is_leap_year(self) -> T_DataArray:
        """Indicate if the date belongs to a leap year"""
        pass

    @property
    def calendar(self) -> CFCalendar:
        """The name of the calendar of the dates.

        Only relevant for arrays of :py:class:`cftime.datetime` objects,
        returns "proleptic_gregorian" for arrays of :py:class:`numpy.datetime64` values.
        """
        pass

class TimedeltaAccessor(TimeAccessor[T_DataArray]):
    """Access Timedelta fields for DataArrays with Timedelta-like dtypes.

    Fields can be accessed through the `.dt` attribute for applicable DataArrays.

    Examples
    --------
    >>> dates = pd.timedelta_range(start="1 day", freq="6h", periods=20)
    >>> ts = xr.DataArray(dates, dims=("time"))
    >>> ts
    <xarray.DataArray (time: 20)> Size: 160B
    array([ 86400000000000, 108000000000000, 129600000000000, 151200000000000,
           172800000000000, 194400000000000, 216000000000000, 237600000000000,
           259200000000000, 280800000000000, 302400000000000, 324000000000000,
           345600000000000, 367200000000000, 388800000000000, 410400000000000,
           432000000000000, 453600000000000, 475200000000000, 496800000000000],
          dtype='timedelta64[ns]')
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt  # doctest: +ELLIPSIS
    <xarray.core.accessor_dt.TimedeltaAccessor object at 0x...>
    >>> ts.dt.days
    <xarray.DataArray 'days' (time: 20)> Size: 160B
    array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.microseconds
    <xarray.DataArray 'microseconds' (time: 20)> Size: 160B
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.seconds
    <xarray.DataArray 'seconds' (time: 20)> Size: 160B
    array([    0, 21600, 43200, 64800,     0, 21600, 43200, 64800,     0,
           21600, 43200, 64800,     0, 21600, 43200, 64800,     0, 21600,
           43200, 64800])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.total_seconds()
    <xarray.DataArray 'total_seconds' (time: 20)> Size: 160B
    array([ 86400., 108000., 129600., 151200., 172800., 194400., 216000.,
           237600., 259200., 280800., 302400., 324000., 345600., 367200.,
           388800., 410400., 432000., 453600., 475200., 496800.])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    """

    @property
    def days(self) -> T_DataArray:
        """Number of days for each element"""
        pass

    @property
    def seconds(self) -> T_DataArray:
        """Number of seconds (>= 0 and less than 1 day) for each element"""
        pass

    @property
    def microseconds(self) -> T_DataArray:
        """Number of microseconds (>= 0 and less than 1 second) for each element"""
        pass

    @property
    def nanoseconds(self) -> T_DataArray:
        """Number of nanoseconds (>= 0 and less than 1 microsecond) for each element"""
        pass

    def total_seconds(self) -> T_DataArray:
        """Total duration of each element expressed in seconds."""
        pass

class CombinedDatetimelikeAccessor(DatetimeAccessor[T_DataArray], TimedeltaAccessor[T_DataArray]):

    def __new__(cls, obj: T_DataArray) -> CombinedDatetimelikeAccessor:
        if not _contains_datetime_like_objects(obj.variable):
            raise AttributeError("'.dt' accessor only available for DataArray with datetime64 timedelta64 dtype or for arrays containing cftime datetime objects.")
        if is_np_timedelta_like(obj.dtype):
            return TimedeltaAccessor(obj)
        else:
            return DatetimeAccessor(obj)
