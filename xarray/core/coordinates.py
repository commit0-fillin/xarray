from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generic, cast
import numpy as np
import pandas as pd
from xarray.core import formatting
from xarray.core.alignment import Aligner
from xarray.core.indexes import Index, Indexes, PandasIndex, PandasMultiIndex, assert_no_index_corrupted, create_default_index_implicit
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import DataVars, Self, T_DataArray, T_Xarray
from xarray.core.utils import Frozen, ReprObject, either_dict_or_kwargs, emit_user_level_warning
from xarray.core.variable import Variable, as_variable, calculate_dimensions
if TYPE_CHECKING:
    from xarray.core.common import DataWithCoords
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
_THIS_ARRAY = ReprObject('<this-array>')

class AbstractCoordinates(Mapping[Hashable, 'T_DataArray']):
    _data: DataWithCoords
    __slots__ = ('_data',)

    def __getitem__(self, key: Hashable) -> T_DataArray:
        raise NotImplementedError()

    @property
    def indexes(self) -> Indexes[pd.Index]:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this Coordinates object has indexes that cannot
        be coerced to pandas.Index objects.

        See Also
        --------
        Coordinates.xindexes
        """
        return Indexes({k: v.to_pandas_index() for k, v in self.xindexes.items()})

    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of :py:class:`~xarray.indexes.Index` objects
        used for label based indexing.
        """
        return self._data.xindexes

    def __iter__(self) -> Iterator[Hashable]:
        for k in self.variables:
            if k in self._names:
                yield k

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._names

    def __repr__(self) -> str:
        return formatting.coords_repr(self)

    def to_index(self, ordered_dims: Sequence[Hashable] | None=None) -> pd.Index:
        """Convert all index coordinates into a :py:class:`pandas.Index`.

        Parameters
        ----------
        ordered_dims : sequence of hashable, optional
            Possibly reordered version of this object's dimensions indicating
            the order in which dimensions should appear on the result.

        Returns
        -------
        pandas.Index
            Index subclass corresponding to the outer-product of all dimension
            coordinates. This will be a MultiIndex if this object is has more
            than more dimension.
        """
        if ordered_dims is None:
            ordered_dims = list(self.dims)
        indexes = [self[dim].to_index() for dim in ordered_dims]
        if len(indexes) == 1:
            return indexes[0]
        return pd.MultiIndex.from_product(indexes, names=ordered_dims)

class Coordinates(AbstractCoordinates):
    """Dictionary like container for Xarray coordinates (variables + indexes).

    This collection is a mapping of coordinate names to
    :py:class:`~xarray.DataArray` objects.

    It can be passed directly to the :py:class:`~xarray.Dataset` and
    :py:class:`~xarray.DataArray` constructors via their `coords` argument. This
    will add both the coordinates variables and their index.

    Coordinates are either:

    - returned via the :py:attr:`Dataset.coords` and :py:attr:`DataArray.coords`
      properties
    - built from Pandas or other index objects
      (e.g., :py:meth:`Coordinates.from_pandas_multiindex`)
    - built directly from coordinate data and Xarray ``Index`` objects (beware that
      no consistency check is done on those inputs)

    Parameters
    ----------
    coords: dict-like, optional
        Mapping where keys are coordinate names and values are objects that
        can be converted into a :py:class:`~xarray.Variable` object
        (see :py:func:`~xarray.as_variable`). If another
        :py:class:`~xarray.Coordinates` object is passed, its indexes
        will be added to the new created object.
    indexes: dict-like, optional
        Mapping where keys are coordinate names and values are
        :py:class:`~xarray.indexes.Index` objects. If None (default),
        pandas indexes will be created for each dimension coordinate.
        Passing an empty dictionary will skip this default behavior.

    Examples
    --------
    Create a dimension coordinate with a default (pandas) index:

    >>> xr.Coordinates({"x": [1, 2]})
    Coordinates:
      * x        (x) int64 16B 1 2

    Create a dimension coordinate with no index:

    >>> xr.Coordinates(coords={"x": [1, 2]}, indexes={})
    Coordinates:
        x        (x) int64 16B 1 2

    Create a new Coordinates object from existing dataset coordinates
    (indexes are passed):

    >>> ds = xr.Dataset(coords={"x": [1, 2]})
    >>> xr.Coordinates(ds.coords)
    Coordinates:
      * x        (x) int64 16B 1 2

    Create indexed coordinates from a ``pandas.MultiIndex`` object:

    >>> midx = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
    >>> xr.Coordinates.from_pandas_multiindex(midx, "x")
    Coordinates:
      * x          (x) object 32B MultiIndex
      * x_level_0  (x) object 32B 'a' 'a' 'b' 'b'
      * x_level_1  (x) int64 32B 0 1 0 1

    Create a new Dataset object by passing a Coordinates object:

    >>> midx_coords = xr.Coordinates.from_pandas_multiindex(midx, "x")
    >>> xr.Dataset(coords=midx_coords)
    <xarray.Dataset> Size: 96B
    Dimensions:    (x: 4)
    Coordinates:
      * x          (x) object 32B MultiIndex
      * x_level_0  (x) object 32B 'a' 'a' 'b' 'b'
      * x_level_1  (x) int64 32B 0 1 0 1
    Data variables:
        *empty*

    """
    _data: DataWithCoords
    __slots__ = ('_data',)

    def __init__(self, coords: Mapping[Any, Any] | None=None, indexes: Mapping[Any, Index] | None=None) -> None:
        from xarray.core.dataset import Dataset
        if coords is None:
            coords = {}
        variables: dict[Hashable, Variable] = {}
        default_indexes: dict[Hashable, PandasIndex] = {}
        coords_obj_indexes: dict[Hashable, Index] = {}
        if isinstance(coords, Coordinates):
            if indexes is not None:
                raise ValueError('passing both a ``Coordinates`` object and a mapping of indexes to ``Coordinates.__init__`` is not allowed (this constructor does not support merging them)')
            variables = {k: v.copy() for k, v in coords.variables.items()}
            coords_obj_indexes = dict(coords.xindexes)
        else:
            for name, data in coords.items():
                var = as_variable(data, name=name, auto_convert=False)
                if var.dims == (name,) and indexes is None:
                    index, index_vars = create_default_index_implicit(var, list(coords))
                    default_indexes.update({k: index for k in index_vars})
                    variables.update(index_vars)
                else:
                    variables[name] = var
        if indexes is None:
            indexes = {}
        else:
            indexes = dict(indexes)
        indexes.update(default_indexes)
        indexes.update(coords_obj_indexes)
        no_coord_index = set(indexes) - set(variables)
        if no_coord_index:
            raise ValueError(f'no coordinate variables found for these indexes: {no_coord_index}')
        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                raise TypeError(f"'{k}' is not an `xarray.indexes.Index` object")
        for k, v in variables.items():
            if k not in indexes:
                variables[k] = v.to_base_variable()
        self._data = Dataset._construct_direct(coord_names=set(variables), variables=variables, indexes=indexes)

    @classmethod
    def from_pandas_multiindex(cls, midx: pd.MultiIndex, dim: str) -> Self:
        """Wrap a pandas multi-index as Xarray coordinates (dimension + levels).

        The returned coordinates can be directly assigned to a
        :py:class:`~xarray.Dataset` or :py:class:`~xarray.DataArray` via the
        ``coords`` argument of their constructor.

        Parameters
        ----------
        midx : :py:class:`pandas.MultiIndex`
            Pandas multi-index object.
        dim : str
            Dimension name.

        Returns
        -------
        coords : Coordinates
            A collection of Xarray indexed coordinates created from the multi-index.

        """
        pass

    @property
    def dims(self) -> Frozen[Hashable, int] | tuple[Hashable, ...]:
        """Mapping from dimension names to lengths or tuple of dimension names."""
        pass

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths."""
        pass

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly.

        See Also
        --------
        Dataset.dtypes
        """
        pass

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Coordinates contents as dict of Variable objects.

        This dictionary is frozen to prevent mutation.
        """
        pass

    def to_dataset(self) -> Dataset:
        """Convert these coordinates into a new Dataset."""
        pass

    def __getitem__(self, key: Hashable) -> DataArray:
        return self._data[key]

    def __delitem__(self, key: Hashable) -> None:
        del self._data.coords[key]

    def equals(self, other: Self) -> bool:
        """Two Coordinates objects are equal if they have matching variables,
        all of which are equal.

        See Also
        --------
        Coordinates.identical
        """
        pass

    def identical(self, other: Self) -> bool:
        """Like equals, but also checks all variable attributes.

        See Also
        --------
        Coordinates.equals
        """
        pass

    def _merge_raw(self, other, reflexive):
        """For use with binary arithmetic."""
        pass

    @contextmanager
    def _merge_inplace(self, other):
        """For use with in-place binary arithmetic."""
        pass

    def merge(self, other: Mapping[Any, Any] | None) -> Dataset:
        """Merge two sets of coordinates to create a new Dataset

        The method implements the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised. You must align your data before passing it to this method.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : dict-like, optional
            A :py:class:`Coordinates` object or any mapping that can be turned
            into coordinates.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        pass

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.update({key: value})

    def update(self, other: Mapping[Any, Any]) -> None:
        """Update this Coordinates variables with other coordinate variables."""
        pass

    def assign(self, coords: Mapping | None=None, **coords_kwargs: Any) -> Self:
        """Assign new coordinates (and indexes) to a Coordinates object, returning
        a new object with all the original coordinates in addition to the new ones.

        Parameters
        ----------
        coords : mapping of dim to coord, optional
            A mapping whose keys are the names of the coordinates and values are the
            coordinates to assign. The mapping will generally be a dict or
            :class:`Coordinates`.

            * If a value is a standard data value — for example, a ``DataArray``,
              scalar, or array — the data is simply assigned as a coordinate.

            * A coordinate can also be defined and attached to an existing dimension
              using a tuple with the first element the dimension name and the second
              element the values for this new coordinate.

        **coords_kwargs
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        new_coords : Coordinates
            A new Coordinates object with the new coordinates (and indexes)
            in addition to all the existing coordinates.

        Examples
        --------
        >>> coords = xr.Coordinates()
        >>> coords
        Coordinates:
            *empty*

        >>> coords.assign(x=[1, 2])
        Coordinates:
          * x        (x) int64 16B 1 2

        >>> midx = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
        >>> coords.assign(xr.Coordinates.from_pandas_multiindex(midx, "y"))
        Coordinates:
          * y          (y) object 32B MultiIndex
          * y_level_0  (y) object 32B 'a' 'a' 'b' 'b'
          * y_level_1  (y) int64 32B 0 1 0 1

        """
        pass

    def _reindex_callback(self, aligner: Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
        """Callback called from ``Aligner`` to create a new reindexed Coordinate."""
        pass

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        pass

    def copy(self, deep: bool=False, memo: dict[int, Any] | None=None) -> Self:
        """Return a copy of this Coordinates object."""
        pass

class DatasetCoordinates(Coordinates):
    """Dictionary like container for Dataset coordinates (variables + indexes).

    This collection can be passed directly to the :py:class:`~xarray.Dataset`
    and :py:class:`~xarray.DataArray` constructors via their `coords` argument.
    This will add both the coordinates variables and their index.
    """
    _data: Dataset
    __slots__ = ('_data',)

    def __init__(self, dataset: Dataset):
        self._data = dataset

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        Dataset.dtypes
        """
        pass

    def __getitem__(self, key: Hashable) -> DataArray:
        if key in self._data.data_vars:
            raise KeyError(key)
        return self._data[key]

    def to_dataset(self) -> Dataset:
        """Convert these coordinates into a new Dataset"""
        pass

    def __delitem__(self, key: Hashable) -> None:
        if key in self:
            del self._data[key]
        else:
            raise KeyError(f'{key!r} is not in coordinate variables {tuple(self.keys())}')

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        pass

class DataArrayCoordinates(Coordinates, Generic[T_DataArray]):
    """Dictionary like container for DataArray coordinates (variables + indexes).

    This collection can be passed directly to the :py:class:`~xarray.Dataset`
    and :py:class:`~xarray.DataArray` constructors via their `coords` argument.
    This will add both the coordinates variables and their index.
    """
    _data: T_DataArray
    __slots__ = ('_data',)

    def __init__(self, dataarray: T_DataArray) -> None:
        self._data = dataarray

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        DataArray.dtype
        """
        pass

    def __getitem__(self, key: Hashable) -> T_DataArray:
        return self._data._getitem_coord(key)

    def __delitem__(self, key: Hashable) -> None:
        if key not in self:
            raise KeyError(f'{key!r} is not in coordinate variables {tuple(self.keys())}')
        assert_no_index_corrupted(self._data.xindexes, {key})
        del self._data._coords[key]
        if self._data._indexes is not None and key in self._data._indexes:
            del self._data._indexes[key]

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        pass

def drop_indexed_coords(coords_to_drop: set[Hashable], coords: Coordinates) -> Coordinates:
    """Drop indexed coordinates associated with coordinates in coords_to_drop.

    This will raise an error in case it corrupts any passed index and its
    coordinate variables.

    """
    pass

def assert_coordinate_consistent(obj: T_Xarray, coords: Mapping[Any, Variable]) -> None:
    """Make sure the dimension coordinate of obj is consistent with coords.

    obj: DataArray or Dataset
    coords: Dict-like of variables
    """
    pass

def create_coords_with_default_indexes(coords: Mapping[Any, Any], data_vars: DataVars | None=None) -> Coordinates:
    """Returns a Coordinates object from a mapping of coordinates (arbitrary objects).

    Create default (pandas) indexes for each of the input dimension coordinates.
    Extract coordinates from each input DataArray.

    """
    pass
