from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import IndexSelResult, PandasIndexingAdapter, PandasMultiIndexingAdapter
from xarray.core.utils import Frozen, emit_user_level_warning, get_valid_numpy_dtype, is_dict_like, is_scalar
if TYPE_CHECKING:
    from xarray.core.types import ErrorOptions, JoinOptions, Self
    from xarray.core.variable import Variable
IndexVars = dict[Any, 'Variable']

class Index:
    """
    Base class inherited by all xarray-compatible indexes.

    Do not use this class directly for creating index objects. Xarray indexes
    are created exclusively from subclasses of ``Index``, mostly via Xarray's
    public API like ``Dataset.set_xindex``.

    Every subclass must at least implement :py:meth:`Index.from_variables`. The
    (re)implementation of the other methods of this base class is optional but
    mostly required in order to support operations relying on indexes such as
    label-based selection or alignment.

    The ``Index`` API closely follows the :py:meth:`Dataset` and
    :py:meth:`DataArray` API, e.g., for an index to support ``.sel()`` it needs
    to implement :py:meth:`Index.sel`, to support ``.stack()`` and
    ``.unstack()`` it needs to implement :py:meth:`Index.stack` and
    :py:meth:`Index.unstack`, etc.

    When a method is not (re)implemented, depending on the case the
    corresponding operation on a :py:meth:`Dataset` or :py:meth:`DataArray`
    either will raise a ``NotImplementedError`` or will simply drop/pass/copy
    the index from/to the result.

    Do not use this class directly for creating index objects.
    """

    @classmethod
    def from_variables(cls, variables: Mapping[Any, Variable], *, options: Mapping[str, Any]) -> Self:
        """Create a new index object from one or more coordinate variables.

        This factory method must be implemented in all subclasses of Index.

        The coordinate variables may be passed here in an arbitrary number and
        order and each with arbitrary dimensions. It is the responsibility of
        the index to check the consistency and validity of these coordinates.

        Parameters
        ----------
        variables : dict-like
            Mapping of :py:class:`Variable` objects holding the coordinate labels
            to index.

        Returns
        -------
        index : Index
            A new Index object.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    @classmethod
    def concat(cls, indexes: Sequence[Self], dim: Hashable, positions: Iterable[Iterable[int]] | None=None) -> Self:
        """Create a new index by concatenating one or more indexes of the same
        type.

        Implementation is optional but required in order to support
        ``concat``. Otherwise it will raise an error if the index needs to be
        updated during the operation.

        Parameters
        ----------
        indexes : sequence of Index objects
            Indexes objects to concatenate together. All objects must be of the
            same type.
        dim : Hashable
            Name of the dimension to concatenate along.
        positions : None or list of integer arrays, optional
            List of integer arrays which specifies the integer positions to which
            to assign each dataset along the concatenated dimension. If not
            supplied, objects are concatenated in the provided order.

        Returns
        -------
        index : Index
            A new Index object.
        """
        raise NotImplementedError("Concatenation is not supported for this index type")

    @classmethod
    def stack(cls, variables: Mapping[Any, Variable], dim: Hashable) -> Self:
        """Create a new index by stacking coordinate variables into a single new
        dimension.

        Implementation is optional but required in order to support ``stack``.
        Otherwise it will raise an error when trying to pass the Index subclass
        as argument to :py:meth:`Dataset.stack`.

        Parameters
        ----------
        variables : dict-like
            Mapping of :py:class:`Variable` objects to stack together.
        dim : Hashable
            Name of the new, stacked dimension.

        Returns
        -------
        index
            A new Index object.
        """
        raise NotImplementedError("Stacking is not supported for this index type")

    def unstack(self) -> tuple[dict[Hashable, Index], pd.MultiIndex]:
        """Unstack a (multi-)index into multiple (single) indexes.

        Implementation is optional but required in order to support unstacking
        the coordinates from which this index has been built.

        Returns
        -------
        indexes : tuple
            A 2-length tuple where the 1st item is a dictionary of unstacked
            Index objects and the 2nd item is a :py:class:`pandas.MultiIndex`
            object used to unstack unindexed coordinate variables or data
            variables.
        """
        raise NotImplementedError("Unstacking is not supported for this index type")

    def create_variables(self, variables: Mapping[Any, Variable] | None=None) -> IndexVars:
        """Maybe create new coordinate variables from this index.

        This method is useful if the index data can be reused as coordinate
        variable data. It is often the case when the underlying index structure
        has an array-like interface, like :py:class:`pandas.Index` objects.

        The variables given as argument (if any) are either returned as-is
        (default behavior) or can be used to copy their metadata (attributes and
        encoding) into the new returned coordinate variables.

        Note: the input variables may or may not have been filtered for this
        index.

        Parameters
        ----------
        variables : dict-like, optional
            Mapping of :py:class:`Variable` objects.

        Returns
        -------
        index_variables : dict-like
            Dictionary of :py:class:`Variable` or :py:class:`IndexVariable`
            objects.
        """
        if variables is None:
            return {}
        return dict(variables)

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a
        ``TypeError`` if this is not supported.

        This method is used by all xarray operations that still rely on
        pandas.Index objects.

        By default it raises a ``TypeError``, unless it is re-implemented in
        subclasses of Index.
        """
        raise TypeError(f"Cannot convert {type(self).__name__} to pandas.Index")

    def isel(self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]) -> Self | None:
        """Maybe returns a new index from the current index itself indexed by
        positional indexers.

        This method should be re-implemented in subclasses of Index if the
        wrapped index structure supports indexing operations. For example,
        indexing a ``pandas.Index`` is pretty straightforward as it behaves very
        much like an array. By contrast, it may be harder doing so for a
        structure like a kd-tree that differs much from a simple array.

        If not re-implemented in subclasses of Index, this method returns
        ``None``, i.e., calling :py:meth:`Dataset.isel` will either drop the
        index in the resulting dataset or pass it unchanged if its corresponding
        coordinate(s) are not indexed.

        Parameters
        ----------
        indexers : dict
            A dictionary of positional indexers as passed from
            :py:meth:`Dataset.isel` and where the entries have been filtered
            for the current index.

        Returns
        -------
        maybe_index : Index
            A new Index object or ``None``.
        """
        return None

    def sel(self, labels: dict[Any, Any]) -> IndexSelResult:
        """Query the index with arbitrary coordinate label indexers.

        Implementation is optional but required in order to support label-based
        selection. Otherwise it will raise an error when trying to call
        :py:meth:`Dataset.sel` with labels for this index coordinates.

        Coordinate label indexers can be of many kinds, e.g., scalar, list,
        tuple, array-like, slice, :py:class:`Variable`, :py:class:`DataArray`, etc.
        It is the responsibility of the index to handle those indexers properly.

        Parameters
        ----------
        labels : dict
            A dictionary of coordinate label indexers passed from
            :py:meth:`Dataset.sel` and where the entries have been filtered
            for the current index.

        Returns
        -------
        sel_results : :py:class:`IndexSelResult`
            An index query result object that contains dimension positional indexers.
            It may also contain new indexes, coordinate variables, etc.
        """
        raise NotImplementedError("Label-based selection is not supported for this index type")

    def join(self, other: Self, how: JoinOptions='inner') -> Self:
        """Return a new index from the combination of this index with another
        index of the same type.

        Implementation is optional but required in order to support alignment.

        Parameters
        ----------
        other : Index
            The other Index object to combine with this index.
        join : str, optional
            Method for joining the two indexes (see :py:func:`~xarray.align`).

        Returns
        -------
        joined : Index
            A new Index object.
        """
        raise NotImplementedError("Joining is not supported for this index type")

    def reindex_like(self, other: Self) -> dict[Hashable, Any]:
        """Query the index with another index of the same type.

        Implementation is optional but required in order to support alignment.

        Parameters
        ----------
        other : Index
            The other Index object used to query this index.

        Returns
        -------
        dim_positional_indexers : dict
            A dictionary where keys are dimension names and values are positional
            indexers.
        """
        raise NotImplementedError("Reindexing is not supported for this index type")

    def equals(self, other: Self) -> bool:
        """Compare this index with another index of the same type.

        Implementation is optional but required in order to support alignment.

        Parameters
        ----------
        other : Index
            The other Index object to compare with this object.

        Returns
        -------
        is_equal : bool
            ``True`` if the indexes are equal, ``False`` otherwise.
        """
        return self is other

    def roll(self, shifts: Mapping[Any, int]) -> Self | None:
        """Roll this index by an offset along one or more dimensions.

        This method can be re-implemented in subclasses of Index, e.g., when the
        index can be itself indexed.

        If not re-implemented, this method returns ``None``, i.e., calling
        :py:meth:`Dataset.roll` will either drop the index in the resulting
        dataset or pass it unchanged if its corresponding coordinate(s) are not
        rolled.

        Parameters
        ----------
        shifts : mapping of hashable to int, optional
            A dict with keys matching dimensions and values given
            by integers to rotate each of the given dimensions, as passed
            :py:meth:`Dataset.roll`.

        Returns
        -------
        rolled : Index
            A new index with rolled data.
        """
        return None

    def rename(self, name_dict: Mapping[Any, Hashable], dims_dict: Mapping[Any, Hashable]) -> Self:
        """Maybe update the index with new coordinate and dimension names.

        This method should be re-implemented in subclasses of Index if it has
        attributes that depend on coordinate or dimension names.

        By default (if not re-implemented), it returns the index itself.

        Warning: the input names are not filtered for this method, they may
        correspond to any variable or dimension of a Dataset or a DataArray.

        Parameters
        ----------
        name_dict : dict-like
            Mapping of current variable or coordinate names to the desired names,
            as passed from :py:meth:`Dataset.rename_vars`.
        dims_dict : dict-like
            Mapping of current dimension names to the desired names, as passed
            from :py:meth:`Dataset.rename_dims`.

        Returns
        -------
        renamed : Index
            Index with renamed attributes.
        """
        return self

    def copy(self, deep: bool=True) -> Self:
        """Return a (deep) copy of this index.

        Implementation in subclasses of Index is optional. The base class
        implements the default (deep) copy semantics.

        Parameters
        ----------
        deep : bool, optional
            If true (default), a copy of the internal structures
            (e.g., wrapped index) is returned with the new object.

        Returns
        -------
        index : Index
            A new Index object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def __copy__(self) -> Self:
        return self.copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None=None) -> Index:
        return self._copy(deep=True, memo=memo)

    def __getitem__(self, indexer: Any) -> Self:
        raise NotImplementedError()

def safe_cast_to_index(array: Any) -> pd.Index:
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    pass

def _asarray_tuplesafe(values):
    """
    Convert values into a numpy array of at most 1-dimension, while preserving
    tuples.

    Adapted from pandas.core.common._asarray_tuplesafe
    """
    pass

def get_indexer_nd(index: pd.Index, labels, method=None, tolerance=None) -> np.ndarray:
    """Wrapper around :meth:`pandas.Index.get_indexer` supporting n-dimensional
    labels
    """
    pass
T_PandasIndex = TypeVar('T_PandasIndex', bound='PandasIndex')

class PandasIndex(Index):
    """Wrap a pandas.Index as an xarray compatible index."""
    index: pd.Index
    dim: Hashable
    coord_dtype: Any
    __slots__ = ('index', 'dim', 'coord_dtype')

    def __init__(self, array: Any, dim: Hashable, coord_dtype: Any=None, *, fastpath: bool=False):
        if fastpath:
            index = array
        else:
            index = safe_cast_to_index(array)
        if index.name is None:
            index = index.copy()
            index.name = dim
        self.index = index
        self.dim = dim
        if coord_dtype is None:
            coord_dtype = get_valid_numpy_dtype(index)
        self.coord_dtype = coord_dtype

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])

    def __repr__(self):
        return f'PandasIndex({repr(self.index)})'

def _check_dim_compat(variables: Mapping[Any, Variable], all_dims: str='equal'):
    """Check that all multi-index variable candidates are 1-dimensional and
    either share the same (single) dimension or each have a different dimension.

    """
    pass
T_PDIndex = TypeVar('T_PDIndex', bound=pd.Index)

def remove_unused_levels_categories(index: T_PDIndex) -> T_PDIndex:
    """
    Remove unused levels from MultiIndex and unused categories from CategoricalIndex
    """
    pass

class PandasMultiIndex(PandasIndex):
    """Wrap a pandas.MultiIndex as an xarray compatible index."""
    index: pd.MultiIndex
    dim: Hashable
    coord_dtype: Any
    level_coords_dtype: dict[str, Any]
    __slots__ = ('index', 'dim', 'coord_dtype', 'level_coords_dtype')

    def __init__(self, array: Any, dim: Hashable, level_coords_dtype: Any=None):
        super().__init__(array, dim)
        names = []
        for i, idx in enumerate(self.index.levels):
            name = idx.name or f'{dim}_level_{i}'
            if name == dim:
                raise ValueError(f'conflicting multi-index level name {name!r} with dimension {dim!r}')
            names.append(name)
        self.index.names = names
        if level_coords_dtype is None:
            level_coords_dtype = {idx.name: get_valid_numpy_dtype(idx) for idx in self.index.levels}
        self.level_coords_dtype = level_coords_dtype

    @classmethod
    def stack(cls, variables: Mapping[Any, Variable], dim: Hashable) -> PandasMultiIndex:
        """Create a new Pandas MultiIndex from the product of 1-d variables (levels) along a
        new dimension.

        Level variables must have a dimension distinct from each other.

        Keeps levels the same (doesn't refactorize them) so that it gives back the original
        labels after a stack/unstack roundtrip.

        """
        pass

    @classmethod
    def from_variables_maybe_expand(cls, dim: Hashable, current_variables: Mapping[Any, Variable], variables: Mapping[Any, Variable]) -> tuple[PandasMultiIndex, IndexVars]:
        """Create a new multi-index maybe by expanding an existing one with
        new variables as index levels.

        The index and its corresponding coordinates may be created along a new dimension.
        """
        pass

    def keep_levels(self, level_variables: Mapping[Any, Variable]) -> PandasMultiIndex | PandasIndex:
        """Keep only the provided levels and return a new multi-index with its
        corresponding coordinates.

        """
        pass

    def reorder_levels(self, level_variables: Mapping[Any, Variable]) -> PandasMultiIndex:
        """Re-arrange index levels using input order and return a new multi-index with
        its corresponding coordinates.

        """
        pass

def create_default_index_implicit(dim_variable: Variable, all_variables: Mapping | Iterable[Hashable] | None=None) -> tuple[PandasIndex, IndexVars]:
    """Create a default index from a dimension variable.

    Create a PandasMultiIndex if the given variable wraps a pandas.MultiIndex,
    otherwise create a PandasIndex (note that this will become obsolete once we
    depreciate implicitly passing a pandas.MultiIndex as a coordinate).

    """
    pass
T_PandasOrXarrayIndex = TypeVar('T_PandasOrXarrayIndex', Index, pd.Index)

class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
    """Immutable proxy for Dataset or DataArray indexes.

    It is a mapping where keys are coordinate names and values are either pandas
    or xarray indexes.

    It also contains the indexed coordinate variables and provides some utility
    methods.

    """
    _index_type: type[Index] | type[pd.Index]
    _indexes: dict[Any, T_PandasOrXarrayIndex]
    _variables: dict[Any, Variable]
    __slots__ = ('_index_type', '_indexes', '_variables', '_dims', '__coord_name_id', '__id_index', '__id_coord_names')

    def __init__(self, indexes: Mapping[Any, T_PandasOrXarrayIndex] | None=None, variables: Mapping[Any, Variable] | None=None, index_type: type[Index] | type[pd.Index]=Index):
        """Constructor not for public consumption.

        Parameters
        ----------
        indexes : dict
            Indexes held by this object.
        variables : dict
            Indexed coordinate variables in this object. Entries must
            match those of `indexes`.
        index_type : type
            The type of all indexes, i.e., either :py:class:`xarray.indexes.Index`
            or :py:class:`pandas.Index`.

        """
        if indexes is None:
            indexes = {}
        if variables is None:
            variables = {}
        unmatched_keys = set(indexes) ^ set(variables)
        if unmatched_keys:
            raise ValueError(f'unmatched keys found in indexes and variables: {unmatched_keys}')
        if any((not isinstance(idx, index_type) for idx in indexes.values())):
            index_type_str = f'{index_type.__module__}.{index_type.__name__}'
            raise TypeError(f'values of indexes must all be instances of {index_type_str}')
        self._index_type = index_type
        self._indexes = dict(**indexes)
        self._variables = dict(**variables)
        self._dims: Mapping[Hashable, int] | None = None
        self.__coord_name_id: dict[Any, int] | None = None
        self.__id_index: dict[int, T_PandasOrXarrayIndex] | None = None
        self.__id_coord_names: dict[int, tuple[Hashable, ...]] | None = None

    def get_unique(self) -> list[T_PandasOrXarrayIndex]:
        """Return a list of unique indexes, preserving order."""
        pass

    def is_multi(self, key: Hashable) -> bool:
        """Return True if ``key`` maps to a multi-coordinate index,
        False otherwise.
        """
        pass

    def get_all_coords(self, key: Hashable, errors: ErrorOptions='raise') -> dict[Hashable, Variable]:
        """Return all coordinates having the same index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        coords : dict
            A dictionary of all coordinate variables having the same index.

        """
        pass

    def get_all_dims(self, key: Hashable, errors: ErrorOptions='raise') -> Mapping[Hashable, int]:
        """Return all dimensions shared by an index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        dims : dict
            A dictionary of all dimensions shared by an index.

        """
        pass

    def group_by_index(self) -> list[tuple[T_PandasOrXarrayIndex, dict[Hashable, Variable]]]:
        """Returns a list of unique indexes and their corresponding coordinates."""
        pass

    def to_pandas_indexes(self) -> Indexes[pd.Index]:
        """Returns an immutable proxy for Dataset or DataArrary pandas indexes.

        Raises an error if this proxy contains indexes that cannot be coerced to
        pandas.Index objects.

        """
        pass

    def copy_indexes(self, deep: bool=True, memo: dict[int, T_PandasOrXarrayIndex] | None=None) -> tuple[dict[Hashable, T_PandasOrXarrayIndex], dict[Hashable, Variable]]:
        """Return a new dictionary with copies of indexes, preserving
        unique indexes.

        Parameters
        ----------
        deep : bool, default: True
            Whether the indexes are deep or shallow copied onto the new object.
        memo : dict if object id to copied objects or None, optional
            To prevent infinite recursion deepcopy stores all copied elements
            in this dict.

        """
        pass

    def __iter__(self) -> Iterator[T_PandasOrXarrayIndex]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __contains__(self, key) -> bool:
        return key in self._indexes

    def __getitem__(self, key) -> T_PandasOrXarrayIndex:
        return self._indexes[key]

    def __repr__(self):
        indexes = formatting._get_indexes_dict(self)
        return formatting.indexes_repr(indexes)

def default_indexes(coords: Mapping[Any, Variable], dims: Iterable) -> dict[Hashable, Index]:
    """Default indexes for a Dataset/DataArray.

    Parameters
    ----------
    coords : Mapping[Any, xarray.Variable]
        Coordinate variables from which to draw default indexes.
    dims : iterable
        Iterable of dimension names.

    Returns
    -------
    Mapping from indexing keys (levels/dimension names) to indexes used for
    indexing along that dimension.
    """
    pass

def indexes_equal(index: Index, other_index: Index, variable: Variable, other_variable: Variable, cache: dict[tuple[int, int], bool | None] | None=None) -> bool:
    """Check if two indexes are equal, possibly with cached results.

    If the two indexes are not of the same type or they do not implement
    equality, fallback to coordinate labels equality check.

    """
    pass

def indexes_all_equal(elements: Sequence[tuple[Index, dict[Hashable, Variable]]]) -> bool:
    """Check if indexes are all equal.

    If they are not of the same type or they do not implement this check, check
    if their coordinate variables are all equal instead.

    """
    pass

def filter_indexes_from_coords(indexes: Mapping[Any, Index], filtered_coord_names: set) -> dict[Hashable, Index]:
    """Filter index items given a (sub)set of coordinate names.

    Drop all multi-coordinate related index items for any key missing in the set
    of coordinate names.

    """
    pass

def assert_no_index_corrupted(indexes: Indexes[Index], coord_names: set[Hashable], action: str='remove coordinate(s)') -> None:
    """Assert removing coordinates or indexes will not corrupt indexes."""
    pass
