from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from xarray.core.dataarray import DataArray
_cpython_optimized_encoders = ('utf-8', 'utf8', 'latin-1', 'latin1', 'iso-8859-1', 'mbcs', 'ascii')
_cpython_optimized_decoders = _cpython_optimized_encoders + ('utf-16', 'utf-32')

def _contains_obj_type(*, pat: Any, checker: Any) -> bool:
    """Determine if the object fits some rule or is array of objects that do so."""
    pass

def _contains_str_like(pat: Any) -> bool:
    """Determine if the object is a str-like or array of str-like."""
    pass

def _contains_compiled_re(pat: Any) -> bool:
    """Determine if the object is a compiled re or array of compiled re."""
    pass

def _contains_callable(pat: Any) -> bool:
    """Determine if the object is a callable or array of callables."""
    pass

class StringAccessor(Generic[T_DataArray]):
    """Vectorized string functions for string-like arrays.

    Similar to pandas, fields can be accessed through the `.str` attribute
    for applicable DataArrays.

        >>> da = xr.DataArray(["some", "text", "in", "an", "array"])
        >>> da.str.len()
        <xarray.DataArray (dim_0: 5)> Size: 40B
        array([4, 4, 2, 2, 5])
        Dimensions without coordinates: dim_0

    It also implements ``+``, ``*``, and ``%``, which operate as elementwise
    versions of the corresponding ``str`` methods. These will automatically
    broadcast for array-like inputs.

        >>> da1 = xr.DataArray(["first", "second", "third"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
        >>> da1.str + da2
        <xarray.DataArray (X: 3, Y: 3)> Size: 252B
        array([['first1', 'first2', 'first3'],
               ['second1', 'second2', 'second3'],
               ['third1', 'third2', 'third3']], dtype='<U7')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["a", "b", "c", "d"], dims=["X"])
        >>> reps = xr.DataArray([3, 4], dims=["Y"])
        >>> da1.str * reps
        <xarray.DataArray (X: 4, Y: 2)> Size: 128B
        array([['aaa', 'aaaa'],
               ['bbb', 'bbbb'],
               ['ccc', 'cccc'],
               ['ddd', 'dddd']], dtype='<U4')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["%s_%s", "%s-%s", "%s|%s"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2], dims=["Y"])
        >>> da3 = xr.DataArray([0.1, 0.2], dims=["Z"])
        >>> da1.str % (da2, da3)
        <xarray.DataArray (X: 3, Y: 2, Z: 2)> Size: 240B
        array([[['1_0.1', '1_0.2'],
                ['2_0.1', '2_0.2']],
        <BLANKLINE>
               [['1-0.1', '1-0.2'],
                ['2-0.1', '2-0.2']],
        <BLANKLINE>
               [['1|0.1', '1|0.2'],
                ['2|0.1', '2|0.2']]], dtype='<U5')
        Dimensions without coordinates: X, Y, Z

    .. note::
        When using ``%`` formatting with a dict, the values are always used as a
        single value, they are not applied elementwise.

            >>> da1 = xr.DataArray(["%(a)s"], dims=["X"])
            >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
            >>> da1 % {"a": da2}
            <xarray.DataArray (X: 1)> Size: 8B
            array(['<xarray.DataArray (Y: 3)> Size: 24B\\narray([1, 2, 3])\\nDimensions without coordinates: Y'],
                  dtype=object)
            Dimensions without coordinates: X
    """
    __slots__ = ('_obj',)

    def __init__(self, obj: T_DataArray) -> None:
        self._obj = obj

    def _stringify(self, invar: Any) -> str | bytes | Any:
        """
        Convert a string-like to the correct string/bytes type.

        This is mostly here to tell mypy a pattern is a str/bytes not a re.Pattern.
        """
        pass

    def len(self) -> T_DataArray:
        """
        Compute the length of each string in the array.

        Returns
        -------
        lengths array : array of int
        """
        pass

    def __getitem__(self, key: int | slice) -> T_DataArray:
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def __add__(self, other: Any) -> T_DataArray:
        return self.cat(other, sep='')

    def __mul__(self, num: int | Any) -> T_DataArray:
        return self.repeat(num)

    def __mod__(self, other: Any) -> T_DataArray:
        if isinstance(other, dict):
            other = {key: self._stringify(val) for key, val in other.items()}
            return self._apply(func=lambda x: x % other)
        elif isinstance(other, tuple):
            other = tuple((self._stringify(x) for x in other))
            return self._apply(func=lambda x, *y: x % y, func_args=other)
        else:
            return self._apply(func=lambda x, y: x % y, func_args=(other,))

    def get(self, i: int | Any, default: str | bytes='') -> T_DataArray:
        """
        Extract character number `i` from each string in the array.

        If `i` is array-like, they are broadcast against the array and
        applied elementwise.

        Parameters
        ----------
        i : int or array-like of int
            Position of element to extract.
            If array-like, it is broadcast.
        default : str or bytes, default: ""
            Value for out-of-range index.

        Returns
        -------
        items : array of object
        """
        pass

    def slice(self, start: int | Any | None=None, stop: int | Any | None=None, step: int | Any | None=None) -> T_DataArray:
        """
        Slice substrings from each string in the array.

        If `start`, `stop`, or 'step` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        start : int or array-like of int, optional
            Start position for slice operation.
            If array-like, it is broadcast.
        stop : int or array-like of int, optional
            Stop position for slice operation.
            If array-like, it is broadcast.
        step : int or array-like of int, optional
            Step size for slice operation.
            If array-like, it is broadcast.

        Returns
        -------
        sliced strings : same type as values
        """
        pass

    def slice_replace(self, start: int | Any | None=None, stop: int | Any | None=None, repl: str | bytes | Any='') -> T_DataArray:
        """
        Replace a positional slice of a string with another value.

        If `start`, `stop`, or 'repl` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        start : int or array-like of int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string. If array-like, it is broadcast.
        stop : int or array-like of int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string. If array-like, it is broadcast.
        repl : str or array-like of str, default: ""
            String for replacement. If not specified, the sliced region
            is replaced with an empty string. If array-like, it is broadcast.

        Returns
        -------
        replaced : same type as values
        """
        pass

    def cat(self, *others, sep: str | bytes | Any='') -> T_DataArray:
        """
        Concatenate strings elementwise in the DataArray with other strings.

        The other strings can either be string scalars or other array-like.
        Dimensions are automatically broadcast together.

        An optional separator `sep` can also be specified. If `sep` is
        array-like, it is broadcast against the array and applied elementwise.

        Parameters
        ----------
        *others : str or array-like of str
            Strings or array-like of strings to concatenate elementwise with
            the current DataArray.
        sep : str or array-like of str, default: "".
            Separator to use between strings.
            It is broadcast in the same way as the other input strings.
            If array-like, its dimensions will be placed at the end of the output array dimensions.

        Returns
        -------
        concatenated : same type as values

        Examples
        --------
        Create a string array

        >>> myarray = xr.DataArray(
        ...     ["11111", "4"],
        ...     dims=["X"],
        ... )

        Create some arrays to concatenate with it

        >>> values_1 = xr.DataArray(
        ...     ["a", "bb", "cccc"],
        ...     dims=["Y"],
        ... )
        >>> values_2 = np.array(3.4)
        >>> values_3 = ""
        >>> values_4 = np.array("test", dtype=np.str_)

        Determine the separator to use

        >>> seps = xr.DataArray(
        ...     [" ", ", "],
        ...     dims=["ZZ"],
        ... )

        Concatenate the arrays using the separator

        >>> myarray.str.cat(values_1, values_2, values_3, values_4, sep=seps)
        <xarray.DataArray (X: 2, Y: 3, ZZ: 2)> Size: 1kB
        array([[['11111 a 3.4  test', '11111, a, 3.4, , test'],
                ['11111 bb 3.4  test', '11111, bb, 3.4, , test'],
                ['11111 cccc 3.4  test', '11111, cccc, 3.4, , test']],
        <BLANKLINE>
               [['4 a 3.4  test', '4, a, 3.4, , test'],
                ['4 bb 3.4  test', '4, bb, 3.4, , test'],
                ['4 cccc 3.4  test', '4, cccc, 3.4, , test']]], dtype='<U24')
        Dimensions without coordinates: X, Y, ZZ

        See Also
        --------
        pandas.Series.str.cat
        str.join
        """
        pass

    def join(self, dim: Hashable=None, sep: str | bytes | Any='') -> T_DataArray:
        """
        Concatenate strings in a DataArray along a particular dimension.

        An optional separator `sep` can also be specified. If `sep` is
        array-like, it is broadcast against the array and applied elementwise.

        Parameters
        ----------
        dim : hashable, optional
            Dimension along which the strings should be concatenated.
            Only one dimension is allowed at a time.
            Optional for 0D or 1D DataArrays, required for multidimensional DataArrays.
        sep : str or array-like, default: "".
            Separator to use between strings.
            It is broadcast in the same way as the other input strings.
            If array-like, its dimensions will be placed at the end of the output array dimensions.

        Returns
        -------
        joined : same type as values

        Examples
        --------
        Create an array

        >>> values = xr.DataArray(
        ...     [["a", "bab", "abc"], ["abcd", "", "abcdef"]],
        ...     dims=["X", "Y"],
        ... )

        Determine the separator

        >>> seps = xr.DataArray(
        ...     ["-", "_"],
        ...     dims=["ZZ"],
        ... )

        Join the strings along a given dimension

        >>> values.str.join(dim="Y", sep=seps)
        <xarray.DataArray (X: 2, ZZ: 2)> Size: 192B
        array([['a-bab-abc', 'a_bab_abc'],
               ['abcd--abcdef', 'abcd__abcdef']], dtype='<U12')
        Dimensions without coordinates: X, ZZ

        See Also
        --------
        pandas.Series.str.join
        str.join
        """
        pass

    def format(self, *args: Any, **kwargs: Any) -> T_DataArray:
        """
        Perform python string formatting on each element of the DataArray.

        This is equivalent to calling `str.format` on every element of the
        DataArray. The replacement values can either be a string-like
        scalar or array-like of string-like values. If array-like,
        the values will be broadcast and applied elementwiseto the input
        DataArray.

        .. note::
            Array-like values provided as `*args` will have their
            dimensions added even if those arguments are not used in any
            string formatting.

        .. warning::
            Array-like arguments are only applied elementwise for `*args`.
            For `**kwargs`, values are used as-is.

        Parameters
        ----------
        *args : str or bytes or array-like of str or bytes
            Values for positional formatting.
            If array-like, the values are broadcast and applied elementwise.
            The dimensions will be placed at the end of the output array dimensions
            in the order they are provided.
        **kwargs : str or bytes or array-like of str or bytes
            Values for keyword-based formatting.
            These are **not** broadcast or applied elementwise.

        Returns
        -------
        formatted : same type as values

        Examples
        --------
        Create an array to format.

        >>> values = xr.DataArray(
        ...     ["{} is {adj0}", "{} and {} are {adj1}"],
        ...     dims=["X"],
        ... )

        Set the values to fill.

        >>> noun0 = xr.DataArray(
        ...     ["spam", "egg"],
        ...     dims=["Y"],
        ... )
        >>> noun1 = xr.DataArray(
        ...     ["lancelot", "arthur"],
        ...     dims=["ZZ"],
        ... )
        >>> adj0 = "unexpected"
        >>> adj1 = "like a duck"

        Insert the values into the array

        >>> values.str.format(noun0, noun1, adj0=adj0, adj1=adj1)
        <xarray.DataArray (X: 2, Y: 2, ZZ: 2)> Size: 1kB
        array([[['spam is unexpected', 'spam is unexpected'],
                ['egg is unexpected', 'egg is unexpected']],
        <BLANKLINE>
               [['spam and lancelot are like a duck',
                 'spam and arthur are like a duck'],
                ['egg and lancelot are like a duck',
                 'egg and arthur are like a duck']]], dtype='<U33')
        Dimensions without coordinates: X, Y, ZZ

        See Also
        --------
        str.format
        """
        pass

    def capitalize(self) -> T_DataArray:
        """
        Convert strings in the array to be capitalized.

        Returns
        -------
        capitalized : same type as values

        Examples
        --------
        >>> da = xr.DataArray(
        ...     ["temperature", "PRESSURE", "PreCipiTation", "daily rainfall"], dims="x"
        ... )
        >>> da
        <xarray.DataArray (x: 4)> Size: 224B
        array(['temperature', 'PRESSURE', 'PreCipiTation', 'daily rainfall'],
              dtype='<U14')
        Dimensions without coordinates: x
        >>> capitalized = da.str.capitalize()
        >>> capitalized
        <xarray.DataArray (x: 4)> Size: 224B
        array(['Temperature', 'Pressure', 'Precipitation', 'Daily rainfall'],
              dtype='<U14')
        Dimensions without coordinates: x
        """
        pass

    def lower(self) -> T_DataArray:
        """
        Convert strings in the array to lowercase.

        Returns
        -------
        lowered : same type as values

        Examples
        --------
        >>> da = xr.DataArray(["Temperature", "PRESSURE"], dims="x")
        >>> da
        <xarray.DataArray (x: 2)> Size: 88B
        array(['Temperature', 'PRESSURE'], dtype='<U11')
        Dimensions without coordinates: x
        >>> lowered = da.str.lower()
        >>> lowered
        <xarray.DataArray (x: 2)> Size: 88B
        array(['temperature', 'pressure'], dtype='<U11')
        Dimensions without coordinates: x
        """
        pass

    def swapcase(self) -> T_DataArray:
        """
        Convert strings in the array to be swapcased.

        Returns
        -------
        swapcased : same type as values

        Examples
        --------
        >>> import xarray as xr
        >>> da = xr.DataArray(["temperature", "PRESSURE", "HuMiDiTy"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 132B
        array(['temperature', 'PRESSURE', 'HuMiDiTy'], dtype='<U11')
        Dimensions without coordinates: x
        >>> swapcased = da.str.swapcase()
        >>> swapcased
        <xarray.DataArray (x: 3)> Size: 132B
        array(['TEMPERATURE', 'pressure', 'hUmIdItY'], dtype='<U11')
        Dimensions without coordinates: x
        """
        pass

    def title(self) -> T_DataArray:
        """
        Convert strings in the array to titlecase.

        Returns
        -------
        titled : same type as values

        Examples
        --------
        >>> da = xr.DataArray(["temperature", "PRESSURE", "HuMiDiTy"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 132B
        array(['temperature', 'PRESSURE', 'HuMiDiTy'], dtype='<U11')
        Dimensions without coordinates: x
        >>> titled = da.str.title()
        >>> titled
        <xarray.DataArray (x: 3)> Size: 132B
        array(['Temperature', 'Pressure', 'Humidity'], dtype='<U11')
        Dimensions without coordinates: x
        """
        pass

    def upper(self) -> T_DataArray:
        """
        Convert strings in the array to uppercase.

        Returns
        -------
        uppered : same type as values

        Examples
        --------
        >>> da = xr.DataArray(["temperature", "HuMiDiTy"], dims="x")
        >>> da
        <xarray.DataArray (x: 2)> Size: 88B
        array(['temperature', 'HuMiDiTy'], dtype='<U11')
        Dimensions without coordinates: x
        >>> uppered = da.str.upper()
        >>> uppered
        <xarray.DataArray (x: 2)> Size: 88B
        array(['TEMPERATURE', 'HUMIDITY'], dtype='<U11')
        Dimensions without coordinates: x
        """
        pass

    def casefold(self) -> T_DataArray:
        """
        Convert strings in the array to be casefolded.

        Casefolding is similar to converting to lowercase,
        but removes all case distinctions.
        This is important in some languages that have more complicated
        cases and case conversions. For example,
        the 'ß' character in German is case-folded to 'ss', whereas it is lowercased
        to 'ß'.

        Returns
        -------
        casefolded : same type as values

        Examples
        --------
        >>> da = xr.DataArray(["TEMPERATURE", "HuMiDiTy"], dims="x")
        >>> da
        <xarray.DataArray (x: 2)> Size: 88B
        array(['TEMPERATURE', 'HuMiDiTy'], dtype='<U11')
        Dimensions without coordinates: x
        >>> casefolded = da.str.casefold()
        >>> casefolded
        <xarray.DataArray (x: 2)> Size: 88B
        array(['temperature', 'humidity'], dtype='<U11')
        Dimensions without coordinates: x

        >>> da = xr.DataArray(["ß", "İ"], dims="x")
        >>> da
        <xarray.DataArray (x: 2)> Size: 8B
        array(['ß', 'İ'], dtype='<U1')
        Dimensions without coordinates: x
        >>> casefolded = da.str.casefold()
        >>> casefolded
        <xarray.DataArray (x: 2)> Size: 16B
        array(['ss', 'i̇'], dtype='<U2')
        Dimensions without coordinates: x
        """
        pass

    def normalize(self, form: str) -> T_DataArray:
        """
        Return the Unicode normal form for the strings in the datarray.

        For more information on the forms, see the documentation for
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {"NFC", "NFKC", "NFD", "NFKD"}
            Unicode form.

        Returns
        -------
        normalized : same type as values

        """
        pass

    def isalnum(self) -> T_DataArray:
        """
        Check whether all characters in each string are alphanumeric.

        Returns
        -------
        isalnum : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["H2O", "NaCl-"], dims="x")
        >>> da
        <xarray.DataArray (x: 2)> Size: 40B
        array(['H2O', 'NaCl-'], dtype='<U5')
        Dimensions without coordinates: x
        >>> isalnum = da.str.isalnum()
        >>> isalnum
        <xarray.DataArray (x: 2)> Size: 2B
        array([ True, False])
        Dimensions without coordinates: x
        """
        pass

    def isalpha(self) -> T_DataArray:
        """
        Check whether all characters in each string are alphabetic.

        Returns
        -------
        isalpha : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["Mn", "H2O", "NaCl-"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 60B
        array(['Mn', 'H2O', 'NaCl-'], dtype='<U5')
        Dimensions without coordinates: x
        >>> isalpha = da.str.isalpha()
        >>> isalpha
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def isdecimal(self) -> T_DataArray:
        """
        Check whether all characters in each string are decimal.

        Returns
        -------
        isdecimal : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["2.3", "123", "0"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 36B
        array(['2.3', '123', '0'], dtype='<U3')
        Dimensions without coordinates: x
        >>> isdecimal = da.str.isdecimal()
        >>> isdecimal
        <xarray.DataArray (x: 3)> Size: 3B
        array([False,  True,  True])
        Dimensions without coordinates: x
        """
        pass

    def isdigit(self) -> T_DataArray:
        """
        Check whether all characters in each string are digits.

        Returns
        -------
        isdigit : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["123", "1.2", "0", "CO2", "NaCl"], dims="x")
        >>> da
        <xarray.DataArray (x: 5)> Size: 80B
        array(['123', '1.2', '0', 'CO2', 'NaCl'], dtype='<U4')
        Dimensions without coordinates: x
        >>> isdigit = da.str.isdigit()
        >>> isdigit
        <xarray.DataArray (x: 5)> Size: 5B
        array([ True, False,  True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def islower(self) -> T_DataArray:
        """
        Check whether all characters in each string are lowercase.

        Returns
        -------
        islower : array of bool
            Array of boolean values with the same shape as the original array indicating whether all characters of each
            element of the string array are lowercase (True) or not (False).

        Examples
        --------
        >>> da = xr.DataArray(["temperature", "HUMIDITY", "pREciPiTaTioN"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 156B
        array(['temperature', 'HUMIDITY', 'pREciPiTaTioN'], dtype='<U13')
        Dimensions without coordinates: x
        >>> islower = da.str.islower()
        >>> islower
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def isnumeric(self) -> T_DataArray:
        """
        Check whether all characters in each string are numeric.

        Returns
        -------
        isnumeric : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["123", "2.3", "H2O", "NaCl-", "Mn"], dims="x")
        >>> da
        <xarray.DataArray (x: 5)> Size: 100B
        array(['123', '2.3', 'H2O', 'NaCl-', 'Mn'], dtype='<U5')
        Dimensions without coordinates: x
        >>> isnumeric = da.str.isnumeric()
        >>> isnumeric
        <xarray.DataArray (x: 5)> Size: 5B
        array([ True, False, False, False, False])
        Dimensions without coordinates: x
        """
        pass

    def isspace(self) -> T_DataArray:
        """
        Check whether all characters in each string are spaces.

        Returns
        -------
        isspace : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["", " ", "\\t", "\\n"], dims="x")
        >>> da
        <xarray.DataArray (x: 4)> Size: 16B
        array(['', ' ', '\\t', '\\n'], dtype='<U1')
        Dimensions without coordinates: x
        >>> isspace = da.str.isspace()
        >>> isspace
        <xarray.DataArray (x: 4)> Size: 4B
        array([False,  True,  True,  True])
        Dimensions without coordinates: x
        """
        pass

    def istitle(self) -> T_DataArray:
        """
        Check whether all characters in each string are titlecase.

        Returns
        -------
        istitle : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [
        ...         "The Evolution Of Species",
        ...         "The Theory of relativity",
        ...         "the quantum mechanics of atoms",
        ...     ],
        ...     dims="title",
        ... )
        >>> da
        <xarray.DataArray (title: 3)> Size: 360B
        array(['The Evolution Of Species', 'The Theory of relativity',
               'the quantum mechanics of atoms'], dtype='<U30')
        Dimensions without coordinates: title
        >>> istitle = da.str.istitle()
        >>> istitle
        <xarray.DataArray (title: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: title
        """
        pass

    def isupper(self) -> T_DataArray:
        """
        Check whether all characters in each string are uppercase.

        Returns
        -------
        isupper : array of bool
            Array of boolean values with the same shape as the original array.

        Examples
        --------
        >>> da = xr.DataArray(["TEMPERATURE", "humidity", "PreCIpiTAtioN"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 156B
        array(['TEMPERATURE', 'humidity', 'PreCIpiTAtioN'], dtype='<U13')
        Dimensions without coordinates: x
        >>> isupper = da.str.isupper()
        >>> isupper
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def count(self, pat: str | bytes | Pattern | Any, flags: int=0, case: bool | None=None) -> T_DataArray:
        """
        Count occurrences of pattern in each string of the array.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~xarray.DataArray`.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.

        Returns
        -------
        counts : array of int

        Examples
        --------
        >>> da = xr.DataArray(["jjklmn", "opjjqrs", "t-JJ99vwx"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 108B
        array(['jjklmn', 'opjjqrs', 't-JJ99vwx'], dtype='<U9')
        Dimensions without coordinates: x

        Using a string:
        >>> da.str.count("jj")
        <xarray.DataArray (x: 3)> Size: 24B
        array([1, 1, 0])
        Dimensions without coordinates: x

        Enable case-insensitive matching by setting case to false:
        >>> counts = da.str.count("jj", case=False)
        >>> counts
        <xarray.DataArray (x: 3)> Size: 24B
        array([1, 1, 1])
        Dimensions without coordinates: x

        Using regex:
        >>> pat = "JJ[0-9]{2}[a-z]{3}"
        >>> counts = da.str.count(pat)
        >>> counts
        <xarray.DataArray (x: 3)> Size: 24B
        array([0, 0, 1])
        Dimensions without coordinates: x

        Using an array of strings (the pattern will be broadcast against the array):

        >>> pat = xr.DataArray(["jj", "JJ"], dims="y")
        >>> counts = da.str.count(pat)
        >>> counts
        <xarray.DataArray (x: 3, y: 2)> Size: 48B
        array([[1, 0],
               [1, 0],
               [0, 1]])
        Dimensions without coordinates: x, y
        """
        pass

    def startswith(self, pat: str | bytes | Any) -> T_DataArray:
        """
        Test if the start of each string in the array matches a pattern.

        The pattern `pat` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.
            If array-like, it is broadcast.

        Returns
        -------
        startswith : array of bool
            An array of booleans indicating whether the given pattern matches
            the start of each string element.

        Examples
        --------
        >>> da = xr.DataArray(["$100", "£23", "100"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 48B
        array(['$100', '£23', '100'], dtype='<U4')
        Dimensions without coordinates: x
        >>> startswith = da.str.startswith("$")
        >>> startswith
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def endswith(self, pat: str | bytes | Any) -> T_DataArray:
        """
        Test if the end of each string in the array matches a pattern.

        The pattern `pat` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.
            If array-like, it is broadcast.

        Returns
        -------
        endswith : array of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.

        Examples
        --------
        >>> da = xr.DataArray(["10C", "10c", "100F"], dims="x")
        >>> da
        <xarray.DataArray (x: 3)> Size: 48B
        array(['10C', '10c', '100F'], dtype='<U4')
        Dimensions without coordinates: x
        >>> endswith = da.str.endswith("C")
        >>> endswith
        <xarray.DataArray (x: 3)> Size: 3B
        array([ True, False, False])
        Dimensions without coordinates: x
        """
        pass

    def pad(self, width: int | Any, side: str='left', fillchar: str | bytes | Any=' ') -> T_DataArray:
        """
        Pad strings in the array up to width.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with character defined in ``fillchar``.
            If array-like, it is broadcast.
        side : {"left", "right", "both"}, default: "left"
            Side from which to fill resulting string.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
            Array with a minimum number of char in each element.

        Examples
        --------
        Pad strings in the array with a single string on the left side.

        Define the string in the array.

        >>> da = xr.DataArray(["PAR184", "TKO65", "NBO9139", "NZ39"], dims="x")
        >>> da
        <xarray.DataArray (x: 4)> Size: 112B
        array(['PAR184', 'TKO65', 'NBO9139', 'NZ39'], dtype='<U7')
        Dimensions without coordinates: x

        Pad the strings

        >>> filled = da.str.pad(8, side="left", fillchar="0")
        >>> filled
        <xarray.DataArray (x: 4)> Size: 128B
        array(['00PAR184', '000TKO65', '0NBO9139', '0000NZ39'], dtype='<U8')
        Dimensions without coordinates: x

        Pad strings on the right side

        >>> filled = da.str.pad(8, side="right", fillchar="0")
        >>> filled
        <xarray.DataArray (x: 4)> Size: 128B
        array(['PAR18400', 'TKO65000', 'NBO91390', 'NZ390000'], dtype='<U8')
        Dimensions without coordinates: x

        Pad strings on both sides

        >>> filled = da.str.pad(8, side="both", fillchar="0")
        >>> filled
        <xarray.DataArray (x: 4)> Size: 128B
        array(['0PAR1840', '0TKO6500', 'NBO91390', '00NZ3900'], dtype='<U8')
        Dimensions without coordinates: x

        Using an array-like width

        >>> width = xr.DataArray([8, 10], dims="y")
        >>> filled = da.str.pad(width, side="left", fillchar="0")
        >>> filled
        <xarray.DataArray (x: 4, y: 2)> Size: 320B
        array([['00PAR184', '0000PAR184'],
               ['000TKO65', '00000TKO65'],
               ['0NBO9139', '000NBO9139'],
               ['0000NZ39', '000000NZ39']], dtype='<U10')
        Dimensions without coordinates: x, y

        Using an array-like value for fillchar

        >>> fillchar = xr.DataArray(["0", "-"], dims="y")
        >>> filled = da.str.pad(8, side="left", fillchar=fillchar)
        >>> filled
        <xarray.DataArray (x: 4, y: 2)> Size: 256B
        array([['00PAR184', '--PAR184'],
               ['000TKO65', '---TKO65'],
               ['0NBO9139', '-NBO9139'],
               ['0000NZ39', '----NZ39']], dtype='<U8')
        Dimensions without coordinates: x, y
        """
        pass

    def _padder(self, *, func: Callable, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        """
        Wrapper function to handle padding operations
        """
        pass

    def center(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        """
        Pad left and right side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        pass

    def ljust(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        """
        Pad right side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        pass

    def rjust(self, width: int | Any, fillchar: str | bytes | Any=' ') -> T_DataArray:
        """
        Pad left side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        pass

    def zfill(self, width: int | Any) -> T_DataArray:
        """
        Pad each string in the array by prepending '0' characters.

        Strings in the array are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the array with length greater or equal to `width` are unchanged.

        If `width` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters. If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        pass

    def contains(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0, regex: bool=True) -> T_DataArray:
        """
        Test if pattern or regex is contained within each string of the array.

        Return boolean array based on whether a given pattern or regex is
        contained within a string of the array.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            Character sequence, a string containing a regular expression,
            or a compiled regular expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        regex : bool, default: True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.
            Cannot be set to `False` if `pat` is a compiled regex.

        Returns
        -------
        contains : array of bool
            An array of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the array.
        """
        pass

    def match(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0) -> T_DataArray:
        """
        Determine if each string in the array matches a regular expression.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or
            a compiled regular expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        matched : array of bool
        """
        pass

    def strip(self, to_strip: str | bytes | Any=None, side: str='both') -> T_DataArray:
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from left and/or right sides.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.
        side : {"left", "right", "both"}, default: "both"
            Side from which to strip.

        Returns
        -------
        stripped : same type as values
        """
        pass

    def lstrip(self, to_strip: str | bytes | Any=None) -> T_DataArray:
        """
        Remove leading characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the left side.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.

        Returns
        -------
        stripped : same type as values
        """
        pass

    def rstrip(self, to_strip: str | bytes | Any=None) -> T_DataArray:
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the right side.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.

        Returns
        -------
        stripped : same type as values
        """
        pass

    def wrap(self, width: int | Any, **kwargs) -> T_DataArray:
        """
        Wrap long strings in the array in paragraphs with length less than `width`.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        If `width` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Maximum line-width.
            If array-like, it is broadcast.
        **kwargs
            keyword arguments passed into :class:`textwrap.TextWrapper`.

        Returns
        -------
        wrapped : same type as values
        """
        pass

    def translate(self, table: Mapping[Any, str | bytes | int | None]) -> T_DataArray:
        """
        Map characters of each string through the given mapping table.

        Parameters
        ----------
        table : dict-like from and to str or bytes or int
            A a mapping of Unicode ordinals to Unicode ordinals, strings, int
            or None. Unmapped characters are left untouched. Characters mapped
            to None are deleted. :meth:`str.maketrans` is a helper function for
            making translation tables.

        Returns
        -------
        translated : same type as values
        """
        pass

    def repeat(self, repeats: int | Any) -> T_DataArray:
        """
        Repeat each string in the array.

        If `repeats` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        repeats : int or array-like of int
            Number of repetitions.
            If array-like, it is broadcast.

        Returns
        -------
        repeated : same type as values
            Array of repeated string objects.
        """
        pass

    def find(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None, side: str='left') -> T_DataArray:
        """
        Return lowest or highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int
        """
        pass

    def rfind(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None) -> T_DataArray:
        """
        Return highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.

        Returns
        -------
        found : array of int
        """
        pass

    def index(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None, side: str='left') -> T_DataArray:
        """
        Return lowest or highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.find`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int

        Raises
        ------
        ValueError
            substring is not found
        """
        pass

    def rindex(self, sub: str | bytes | Any, start: int | Any=0, end: int | Any=None) -> T_DataArray:
        """
        Return highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.rfind`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.

        Returns
        -------
        found : array of int

        Raises
        ------
        ValueError
            substring is not found
        """
        pass

    def replace(self, pat: str | bytes | Pattern | Any, repl: str | bytes | Callable | Any, n: int | Any=-1, case: bool | None=None, flags: int=0, regex: bool=True) -> T_DataArray:
        """
        Replace occurrences of pattern/regex in the array with some string.

        If `pat`, `repl`, or 'n` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            String can be a character sequence or regular expression.
            If array-like, it is broadcast.
        repl : str or callable or array-like of str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            See :func:`re.sub`.
            If array-like, it is broadcast.
        n : int or array of int, default: -1
            Number of replacements to make from start. Use ``-1`` to replace all.
            If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        regex : bool, default: True
            If True, assumes the passed-in pattern is a regular expression.
            If False, treats the pattern as a literal string.
            Cannot be set to False if `pat` is a compiled regex or `repl` is
            a callable.

        Returns
        -------
        replaced : same type as values
            A copy of the object with all matching occurrences of `pat`
            replaced by `repl`.
        """
        pass

    def extract(self, pat: str | bytes | Pattern | Any, dim: Hashable, case: bool | None=None, flags: int=0) -> T_DataArray:
        """
        Extract the first match of capture groups in the regex pat as a new
        dimension in a DataArray.

        For each string in the DataArray, extract groups from the first match
        of regular expression pat.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        dim : hashable or None
            Name of the new dimension to store the captured strings in.
            If None, the pattern must have only one capture group and the
            resulting DataArray will have the same size as the original.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : same type as values or object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `dim` is None and there is more than one capture group.
        ValueError
            `case` is set when `pat` is a compiled regular expression.
        KeyError
            The given dimension is already present in the DataArray.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.extract(r"(\\w+)_Xy_(\\d*)", dim="match")
        <xarray.DataArray (X: 2, Y: 3, match: 2)> Size: 288B
        array([[['a', '0'],
                ['bab', '110'],
                ['abc', '01']],
        <BLANKLINE>
               [['abcd', ''],
                ['', ''],
                ['abcdef', '101']]], dtype='<U6')
        Dimensions without coordinates: X, Y, match

        See Also
        --------
        DataArray.str.extractall
        DataArray.str.findall
        re.compile
        re.search
        pandas.Series.str.extract
        """
        pass

    def extractall(self, pat: str | bytes | Pattern | Any, group_dim: Hashable, match_dim: Hashable, case: bool | None=None, flags: int=0) -> T_DataArray:
        """
        Extract all matches of capture groups in the regex pat as new
        dimensions in a DataArray.

        For each string in the DataArray, extract groups from all matches
        of regular expression pat.
        Equivalent to applying re.findall() to all the elements in the DataArray
        and splitting the results across dimensions.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        group_dim : hashable
            Name of the new dimensions corresponding to the capture groups.
            This dimension is added to the new DataArray first.
        match_dim : hashable
            Name of the new dimensions corresponding to the matches for each group.
            This dimension is added to the new DataArray second.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : same type as values or object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `case` is set when `pat` is a compiled regular expression.
        KeyError
            Either of the given dimensions is already present in the DataArray.
        KeyError
            The given dimensions names are the same.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.extractall(
        ...     r"(\\w+)_Xy_(\\d*)", group_dim="group", match_dim="match"
        ... )
        <xarray.DataArray (X: 2, Y: 3, group: 3, match: 2)> Size: 1kB
        array([[[['a', '0'],
                 ['', ''],
                 ['', '']],
        <BLANKLINE>
                [['bab', '110'],
                 ['baab', '1100'],
                 ['', '']],
        <BLANKLINE>
                [['abc', '01'],
                 ['cbc', '2210'],
                 ['', '']]],
        <BLANKLINE>
        <BLANKLINE>
               [[['abcd', ''],
                 ['dcd', '33210'],
                 ['dccd', '332210']],
        <BLANKLINE>
                [['', ''],
                 ['', ''],
                 ['', '']],
        <BLANKLINE>
                [['abcdef', '101'],
                 ['fef', '5543210'],
                 ['', '']]]], dtype='<U7')
        Dimensions without coordinates: X, Y, group, match

        See Also
        --------
        DataArray.str.extract
        DataArray.str.findall
        re.compile
        re.findall
        pandas.Series.str.extractall
        """
        pass

    def findall(self, pat: str | bytes | Pattern | Any, case: bool | None=None, flags: int=0) -> T_DataArray:
        """
        Find all occurrences of pattern or regular expression in the DataArray.

        Equivalent to applying re.findall() to all the elements in the DataArray.
        Results in an object array of lists.
        If there is only one capture group, the lists will be a sequence of matches.
        If there are multiple capture groups, the lists will be a sequence of lists,
        each of which contains a sequence of matches.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `case` is set when `pat` is a compiled regular expression.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.findall(r"(\\w+)_Xy_(\\d*)")
        <xarray.DataArray (X: 2, Y: 3)> Size: 48B
        array([[list([('a', '0')]), list([('bab', '110'), ('baab', '1100')]),
                list([('abc', '01'), ('cbc', '2210')])],
               [list([('abcd', ''), ('dcd', '33210'), ('dccd', '332210')]),
                list([]), list([('abcdef', '101'), ('fef', '5543210')])]],
              dtype=object)
        Dimensions without coordinates: X, Y

        See Also
        --------
        DataArray.str.extract
        DataArray.str.extractall
        re.compile
        re.findall
        pandas.Series.str.findall
        """
        pass

    def _partitioner(self, *, func: Callable, dim: Hashable | None, sep: str | bytes | Any | None) -> T_DataArray:
        """
        Implements logic for `partition` and `rpartition`.
        """
        pass

    def partition(self, dim: Hashable | None, sep: str | bytes | Any=' ') -> T_DataArray:
        """
        Split the strings in the DataArray at the first occurrence of separator `sep`.

        This method splits the string at the first occurrence of `sep`,
        and returns 3 elements containing the part before the separator,
        the separator itself, and the part after the separator.
        If the separator is not found, return 3 elements containing the string itself,
        followed by two empty strings.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the 3 elements in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str or bytes or array-like, default: " "
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        partitioned : same type as values or object array

        See Also
        --------
        DataArray.str.rpartition
        str.partition
        pandas.Series.str.partition
        """
        pass

    def rpartition(self, dim: Hashable | None, sep: str | bytes | Any=' ') -> T_DataArray:
        """
        Split the strings in the DataArray at the last occurrence of separator `sep`.

        This method splits the string at the last occurrence of `sep`,
        and returns 3 elements containing the part before the separator,
        the separator itself, and the part after the separator.
        If the separator is not found, return 3 elements containing two empty strings,
        followed by the string itself.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the 3 elements in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str or bytes or array-like, default: " "
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        rpartitioned : same type as values or object array

        See Also
        --------
        DataArray.str.partition
        str.rpartition
        pandas.Series.str.rpartition
        """
        pass

    def _splitter(self, *, func: Callable, pre: bool, dim: Hashable, sep: str | bytes | Any | None, maxsplit: int) -> DataArray:
        """
        Implements logic for `split` and `rsplit`.
        """
        pass

    def split(self, dim: Hashable | None, sep: str | bytes | Any=None, maxsplit: int=-1) -> DataArray:
        """
        Split strings in a DataArray around the given separator/delimiter `sep`.

        Splits the string in the DataArray from the beginning,
        at the specified delimiter string.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the results in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str, default: None
            String to split on. If ``None`` (the default), split on any whitespace.
            If array-like, it is broadcast.
        maxsplit : int, default: -1
            Limit number of splits in output, starting from the beginning.
            If -1 (the default), return all splits.

        Returns
        -------
        splitted : same type as values or object array

        Examples
        --------
        Create a string DataArray

        >>> values = xr.DataArray(
        ...     [
        ...         ["abc def", "spam\\t\\teggs\\tswallow", "red_blue"],
        ...         ["test0\\ntest1\\ntest2\\n\\ntest3", "", "abra  ka\\nda\\tbra"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Split once and put the results in a new dimension

        >>> values.str.split(dim="splitted", maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3, splitted: 2)> Size: 864B
        array([[['abc', 'def'],
                ['spam', 'eggs\\tswallow'],
                ['red_blue', '']],
        <BLANKLINE>
               [['test0', 'test1\\ntest2\\n\\ntest3'],
                ['', ''],
                ['abra', 'ka\\nda\\tbra']]], dtype='<U18')
        Dimensions without coordinates: X, Y, splitted

        Split as many times as needed and put the results in a new dimension

        >>> values.str.split(dim="splitted")
        <xarray.DataArray (X: 2, Y: 3, splitted: 4)> Size: 768B
        array([[['abc', 'def', '', ''],
                ['spam', 'eggs', 'swallow', ''],
                ['red_blue', '', '', '']],
        <BLANKLINE>
               [['test0', 'test1', 'test2', 'test3'],
                ['', '', '', ''],
                ['abra', 'ka', 'da', 'bra']]], dtype='<U8')
        Dimensions without coordinates: X, Y, splitted

        Split once and put the results in lists

        >>> values.str.split(dim=None, maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3)> Size: 48B
        array([[list(['abc', 'def']), list(['spam', 'eggs\\tswallow']),
                list(['red_blue'])],
               [list(['test0', 'test1\\ntest2\\n\\ntest3']), list([]),
                list(['abra', 'ka\\nda\\tbra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split as many times as needed and put the results in a list

        >>> values.str.split(dim=None)
        <xarray.DataArray (X: 2, Y: 3)> Size: 48B
        array([[list(['abc', 'def']), list(['spam', 'eggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0', 'test1', 'test2', 'test3']), list([]),
                list(['abra', 'ka', 'da', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split only on spaces

        >>> values.str.split(dim="splitted", sep=" ")
        <xarray.DataArray (X: 2, Y: 3, splitted: 3)> Size: 2kB
        array([[['abc', 'def', ''],
                ['spam\\t\\teggs\\tswallow', '', ''],
                ['red_blue', '', '']],
        <BLANKLINE>
               [['test0\\ntest1\\ntest2\\n\\ntest3', '', ''],
                ['', '', ''],
                ['abra', '', 'ka\\nda\\tbra']]], dtype='<U24')
        Dimensions without coordinates: X, Y, splitted

        See Also
        --------
        DataArray.str.rsplit
        str.split
        pandas.Series.str.split
        """
        pass

    def rsplit(self, dim: Hashable | None, sep: str | bytes | Any=None, maxsplit: int | Any=-1) -> DataArray:
        """
        Split strings in a DataArray around the given separator/delimiter `sep`.

        Splits the string in the DataArray from the end,
        at the specified delimiter string.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the results in.
            If `None`, place the results as list elements in an object DataArray
        sep : str, default: None
            String to split on. If ``None`` (the default), split on any whitespace.
            If array-like, it is broadcast.
        maxsplit : int, default: -1
            Limit number of splits in output, starting from the end.
            If -1 (the default), return all splits.
            The final number of split values may be less than this if there are no
            DataArray elements with that many values.

        Returns
        -------
        rsplitted : same type as values or object array

        Examples
        --------
        Create a string DataArray

        >>> values = xr.DataArray(
        ...     [
        ...         ["abc def", "spam\\t\\teggs\\tswallow", "red_blue"],
        ...         ["test0\\ntest1\\ntest2\\n\\ntest3", "", "abra  ka\\nda\\tbra"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Split once and put the results in a new dimension

        >>> values.str.rsplit(dim="splitted", maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3, splitted: 2)> Size: 816B
        array([[['abc', 'def'],
                ['spam\\t\\teggs', 'swallow'],
                ['', 'red_blue']],
        <BLANKLINE>
               [['test0\\ntest1\\ntest2', 'test3'],
                ['', ''],
                ['abra  ka\\nda', 'bra']]], dtype='<U17')
        Dimensions without coordinates: X, Y, splitted

        Split as many times as needed and put the results in a new dimension

        >>> values.str.rsplit(dim="splitted")
        <xarray.DataArray (X: 2, Y: 3, splitted: 4)> Size: 768B
        array([[['', '', 'abc', 'def'],
                ['', 'spam', 'eggs', 'swallow'],
                ['', '', '', 'red_blue']],
        <BLANKLINE>
               [['test0', 'test1', 'test2', 'test3'],
                ['', '', '', ''],
                ['abra', 'ka', 'da', 'bra']]], dtype='<U8')
        Dimensions without coordinates: X, Y, splitted

        Split once and put the results in lists

        >>> values.str.rsplit(dim=None, maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3)> Size: 48B
        array([[list(['abc', 'def']), list(['spam\\t\\teggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0\\ntest1\\ntest2', 'test3']), list([]),
                list(['abra  ka\\nda', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split as many times as needed and put the results in a list

        >>> values.str.rsplit(dim=None)
        <xarray.DataArray (X: 2, Y: 3)> Size: 48B
        array([[list(['abc', 'def']), list(['spam', 'eggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0', 'test1', 'test2', 'test3']), list([]),
                list(['abra', 'ka', 'da', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split only on spaces

        >>> values.str.rsplit(dim="splitted", sep=" ")
        <xarray.DataArray (X: 2, Y: 3, splitted: 3)> Size: 2kB
        array([[['', 'abc', 'def'],
                ['', '', 'spam\\t\\teggs\\tswallow'],
                ['', '', 'red_blue']],
        <BLANKLINE>
               [['', '', 'test0\\ntest1\\ntest2\\n\\ntest3'],
                ['', '', ''],
                ['abra', '', 'ka\\nda\\tbra']]], dtype='<U24')
        Dimensions without coordinates: X, Y, splitted

        See Also
        --------
        DataArray.str.split
        str.rsplit
        pandas.Series.str.rsplit
        """
        pass

    def get_dummies(self, dim: Hashable, sep: str | bytes | Any='|') -> DataArray:
        """
        Return DataArray of dummy/indicator variables.

        Each string in the DataArray is split at `sep`.
        A new dimension is created with coordinates for each unique result,
        and the corresponding element of that dimension is `True` if
        that result is present and `False` if not.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable
            Name for the dimension to place the results in.
        sep : str, default: "|".
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        dummies : array of bool

        Examples
        --------
        Create a string array

        >>> values = xr.DataArray(
        ...     [
        ...         ["a|ab~abc|abc", "ab", "a||abc|abcd"],
        ...         ["abcd|ab|a", "abc|ab~abc", "|a"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract dummy values

        >>> values.str.get_dummies(dim="dummies")
        <xarray.DataArray (X: 2, Y: 3, dummies: 5)> Size: 30B
        array([[[ True, False,  True, False,  True],
                [False,  True, False, False, False],
                [ True, False,  True,  True, False]],
        <BLANKLINE>
               [[ True,  True, False,  True, False],
                [False, False,  True, False,  True],
                [ True, False, False, False, False]]])
        Coordinates:
          * dummies  (dummies) <U6 120B 'a' 'ab' 'abc' 'abcd' 'ab~abc'
        Dimensions without coordinates: X, Y

        See Also
        --------
        pandas.Series.str.get_dummies
        """
        pass

    def decode(self, encoding: str, errors: str='strict') -> T_DataArray:
        """
        Decode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
            The encoding to use.
            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
            section for a list of encodings handlers.
        errors : str, default: "strict"
            The handler for encoding errors.
            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_
            for a list of error handlers.

        Returns
        -------
        decoded : same type as values
        """
        pass

    def encode(self, encoding: str, errors: str='strict') -> T_DataArray:
        """
        Encode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
            The encoding to use.
            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
            section for a list of encodings handlers.
        errors : str, default: "strict"
            The handler for encoding errors.
            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_
            for a list of error handlers.

        Returns
        -------
        encoded : same type as values
        """
        pass