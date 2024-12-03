from __future__ import annotations
import itertools
import textwrap
from collections import ChainMap
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NoReturn, Union, overload
from xarray.core import utils
from xarray.core.alignment import align
from xarray.core.common import TreeAttrAccessMixin
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.datatree_mapping import TreeIsomorphismError, check_isomorphic, map_over_subtree
from xarray.core.datatree_ops import DataTreeArithmeticMixin, MappedDatasetMethodsMixin, MappedDataWithCoords
from xarray.core.datatree_render import RenderDataTree
from xarray.core.formatting import datatree_repr, dims_and_coords_repr
from xarray.core.formatting_html import datatree_repr as datatree_repr_html
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import Default, Frozen, HybridMappingProxy, _default, either_dict_or_kwargs, maybe_wrap_array
from xarray.core.variable import Variable
try:
    from xarray.core.variable import calculate_dimensions
except ImportError:
    from xarray.core.dataset import calculate_dimensions
if TYPE_CHECKING:
    import pandas as pd
    from xarray.core.datatree_io import T_DataTreeNetcdfEngine, T_DataTreeNetcdfTypes
    from xarray.core.merge import CoercibleMapping, CoercibleValue
    from xarray.core.types import ErrorOptions, NetcdfWriteModes, ZarrWriteModes
T_Path = Union[str, NodePath]

class DatasetView(Dataset):
    """
    An immutable Dataset-like view onto the data in a single DataTree node.

    In-place operations modifying this object should raise an AttributeError.
    This requires overriding all inherited constructors.

    Operations returning a new result will return a new xarray.Dataset object.
    This includes all API on Dataset, which will be inherited.
    """
    __slots__ = ('_attrs', '_cache', '_coord_names', '_dims', '_encoding', '_close', '_indexes', '_variables')

    def __init__(self, data_vars: Mapping[Any, Any] | None=None, coords: Mapping[Any, Any] | None=None, attrs: Mapping[Any, Any] | None=None):
        raise AttributeError('DatasetView objects are not to be initialized directly')

    @classmethod
    def _constructor(cls, variables: dict[Any, Variable], coord_names: set[Hashable], dims: dict[Any, int], attrs: dict | None, indexes: dict[Any, Index], encoding: dict | None, close: Callable[[], None] | None) -> DatasetView:
        """Private constructor, from Dataset attributes."""
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._attrs = attrs
        obj._indexes = indexes
        obj._encoding = encoding
        obj._close = close
        return obj

    def __setitem__(self, key, val) -> None:
        raise AttributeError('Mutation of the DatasetView is not allowed, please use `.__setitem__` on the wrapping DataTree node, or use `dt.to_dataset()` if you want a mutable dataset. If calling this from within `map_over_subtree`,use `.copy()` first to get a mutable version of the input dataset.')

    @overload
    def __getitem__(self, key: Mapping) -> Dataset:
        ...

    @overload
    def __getitem__(self, key: Hashable) -> DataArray:
        ...

    @overload
    def __getitem__(self, key: Any) -> Dataset:
        ...

    def __getitem__(self, key) -> DataArray | Dataset:
        return Dataset.__getitem__(self, key)

    @classmethod
    def _construct_direct(cls, variables: dict[Any, Variable], coord_names: set[Hashable], dims: dict[Any, int] | None=None, attrs: dict | None=None, indexes: dict[Any, Index] | None=None, encoding: dict | None=None, close: Callable[[], None] | None=None) -> Dataset:
        """
        Overriding this method (along with ._replace) and modifying it to return a Dataset object
        should hopefully ensure that the return type of any method on this object is a Dataset.
        """
        return Dataset._construct_direct(variables, coord_names, dims, attrs, indexes, encoding, close)

    def _replace(self, variables: dict[Hashable, Variable] | None=None, coord_names: set[Hashable] | None=None, dims: dict[Any, int] | None=None, attrs: dict[Hashable, Any] | None | Default=_default, indexes: dict[Hashable, Index] | None=None, encoding: dict | None | Default=_default, inplace: bool=False) -> Dataset:
        """
        Overriding this method (along with ._construct_direct) and modifying it to return a Dataset object
        should hopefully ensure that the return type of any method on this object is a Dataset.
        """
        return Dataset._replace(self, variables, coord_names, dims, attrs, indexes, encoding, inplace)

    def map(self, func: Callable, keep_attrs: bool | None=None, args: Iterable[Any]=(), **kwargs: Any) -> Dataset:
        """Apply a function to each data variable in this dataset

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, *args, **kwargs)`
            to transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool | None, optional
            If True, both the dataset's and variables' attributes (`attrs`) will be
            copied from the original objects to the new ones. If False, the new dataset
            and variables will be returned without copying the attributes.
        args : iterable, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying ``func`` to each data variable.

        Examples
        --------
        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset({"foo": da, "bar": ("x", [-1, 2])})
        >>> ds
        <xarray.Dataset> Size: 64B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 48B 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2
        >>> ds.map(np.fabs)
        <xarray.Dataset> Size: 64B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 48B 1.764 0.4002 0.9787 2.241 1.868 0.9773
            bar      (x) float64 16B 1.0 2.0
        """
        variables = {k: func(v, *args, **kwargs) for k, v in self.data_vars.items()}
        if keep_attrs is None:
            keep_attrs = XR_OPTS["keep_attrs"]
        return type(self)._construct_direct(
            variables,
            self._coord_names,
            self._dims,
            self._attrs if keep_attrs else None,
            self._indexes,
            self._encoding,
            self._close,
        )

class DataTree(NamedNode, MappedDatasetMethodsMixin, MappedDataWithCoords, DataTreeArithmeticMixin, TreeAttrAccessMixin, Generic[Tree], Mapping):
    """
    A tree-like hierarchical collection of xarray objects.

    Attempts to present an API like that of xarray.Dataset, but methods are wrapped to also update all the tree's child nodes.
    """
    _name: str | None
    _parent: DataTree | None
    _children: dict[str, DataTree]
    _cache: dict[str, Any]
    _data_variables: dict[Hashable, Variable]
    _node_coord_variables: dict[Hashable, Variable]
    _node_dims: dict[Hashable, int]
    _node_indexes: dict[Hashable, Index]
    _attrs: dict[Hashable, Any] | None
    _encoding: dict[Hashable, Any] | None
    _close: Callable[[], None] | None
    __slots__ = ('_name', '_parent', '_children', '_cache', '_data_variables', '_node_coord_variables', '_node_dims', '_node_indexes', '_attrs', '_encoding', '_close')

    def __init__(self, data: Dataset | DataArray | None=None, parent: DataTree | None=None, children: Mapping[str, DataTree] | None=None, name: str | None=None):
        """
        Create a single node of a DataTree.

        The node may optionally contain data in the form of data and coordinate
        variables, stored in the same way as data is stored in an
        xarray.Dataset.

        Parameters
        ----------
        data : Dataset, DataArray, or None, optional
            Data to store under the .ds attribute of this node. DataArrays will
            be promoted to Datasets. Default is None.
        parent : DataTree, optional
            Parent node to this node. Default is None.
        children : Mapping[str, DataTree], optional
            Any child nodes of this node. Default is None.
        name : str, optional
            Name for this node of the tree. Default is None.

        Returns
        -------
        DataTree

        See Also
        --------
        DataTree.from_dict
        """
        if children is None:
            children = {}
        super().__init__(name=name)
        self._set_node_data(_coerce_to_dataset(data))
        self.parent = parent
        self.children = children

    @property
    def parent(self: DataTree) -> DataTree | None:
        """Parent of this node."""
        return self._parent

    @property
    def ds(self) -> DatasetView:
        """
        An immutable Dataset-like view onto the data in this node.

        Includes inherited coordinates and indexes from parent nodes.

        For a mutable Dataset containing the same data as in this node, use
        `.to_dataset()` instead.

        See Also
        --------
        DataTree.to_dataset
        """
        return DatasetView._constructor(
            self._data_variables,
            set(self._node_coord_variables),
            self._node_dims,
            self._attrs,
            self._node_indexes,
            self._encoding,
            self._close
        )

    def to_dataset(self, inherited: bool=True) -> Dataset:
        """
        Return the data in this node as a new xarray.Dataset object.

        Parameters
        ----------
        inherited : bool, optional
            If False, only include coordinates and indexes defined at the level
            of this DataTree node, excluding inherited coordinates.

        See Also
        --------
        DataTree.ds
        """
        if inherited:
            coords = self.coords
            indexes = self.indexes
        else:
            coords = {k: v for k, v in self._node_coord_variables.items()}
            indexes = self._node_indexes

        return Dataset(
            data_vars=self._data_variables,
            coords=coords,
            attrs=self._attrs,
            indexes=indexes
        )

    @property
    def has_data(self) -> bool:
        """Whether or not there are any variables in this node."""
        pass

    @property
    def has_attrs(self) -> bool:
        """Whether or not there are any metadata attributes in this node."""
        pass

    @property
    def is_empty(self) -> bool:
        """False if node contains any data or attrs. Does not look at children."""
        pass

    @property
    def is_hollow(self) -> bool:
        """True if only leaf nodes contain data."""
        pass

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to node contents as dict of Variable objects.

        This dictionary is frozen to prevent mutation that could violate
        Dataset invariants. It contains all variable objects constituting this
        DataTree node, including both data variables and coordinates.
        """
        pass

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """Dictionary of global attributes on this node object."""
        pass

    @property
    def encoding(self) -> dict:
        """Dictionary of global encoding attributes on this node object."""
        pass

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `DataTree.sizes`, `Dataset.sizes`, and `DataArray.sizes` for consistently named
        properties.
        """
        pass

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `DataTree.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        pass

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        pass

    @property
    def _item_sources(self) -> Iterable[Mapping[Any, Any]]:
        """Places to look-up items for key-completion"""
        pass

    def _ipython_key_completions_(self) -> list[str]:
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        pass

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is either an array stored in the datatree or a child node, or neither.
        """
        return key in self.variables or key in self.children

    def __bool__(self) -> bool:
        return bool(self._data_variables) or bool(self._children)

    def __iter__(self) -> Iterator[Hashable]:
        return itertools.chain(self._data_variables, self._children)

    def __array__(self, dtype=None, copy=None):
        raise TypeError('cannot directly convert a DataTree into a numpy array. Instead, create an xarray.DataArray first, either with indexing on the DataTree or by invoking the `to_array()` method.')

    def __repr__(self) -> str:
        return datatree_repr(self)

    def __str__(self) -> str:
        return datatree_repr(self)

    def _repr_html_(self):
        """Make html representation of datatree object"""
        pass

    def copy(self: DataTree, deep: bool=False) -> DataTree:
        """
        Returns a copy of this subtree.

        Copies this node and all child nodes.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new datatree is the same as in
        the original datatree.

        Parameters
        ----------
        deep : bool, default: False
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.

        Returns
        -------
        object : DataTree
            New object with dimensions, attributes, coordinates, name, encoding,
            and data of this node and all child nodes copied from original.

        See Also
        --------
        xarray.Dataset.copy
        pandas.DataFrame.copy
        """
        pass

    def _copy_subtree(self: DataTree, deep: bool=False, memo: dict[int, Any] | None=None) -> DataTree:
        """Copy entire subtree"""
        pass

    def _copy_node(self: DataTree, deep: bool=False) -> DataTree:
        """Copy just one node of a tree"""
        pass

    def __copy__(self: DataTree) -> DataTree:
        return self._copy_subtree(deep=False)

    def __deepcopy__(self: DataTree, memo: dict[int, Any] | None=None) -> DataTree:
        return self._copy_subtree(deep=True, memo=memo)

    def get(self: DataTree, key: str, default: DataTree | DataArray | None=None) -> DataTree | DataArray | None:
        """
        Access child nodes, variables, or coordinates stored in this node.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node. Must lie in this immediate node (not elsewhere in the tree).
        default : DataTree | DataArray | None, optional
            A value to return if the specified key does not exist. Default return value is None.
        """
        pass

    def __getitem__(self: DataTree, key: str) -> DataTree | DataArray:
        """
        Access child nodes, variables, or coordinates stored anywhere in this tree.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node, or unix-like path to variable / child within another node.

        Returns
        -------
        DataTree | DataArray
        """
        if utils.is_dict_like(key):
            raise NotImplementedError('Should this index over whole tree?')
        elif isinstance(key, str):
            path = NodePath(key)
            return self._get_item(path)
        elif utils.is_list_like(key):
            raise NotImplementedError('Selecting via tags is deprecated, and selecting multiple items should be implemented via .subset')
        else:
            raise ValueError(f'Invalid format for key: {key}')

    def _set(self, key: str, val: DataTree | CoercibleValue) -> None:
        """
        Set the child node or variable with the specified key to value.

        Counterpart to the public .get method, and also only works on the immediate node, not other nodes in the tree.
        """
        pass

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Add either a child node or an array to the tree, at any position.

        Data can be added anywhere, and new nodes will be created to cross the path to the new location if necessary.

        If there is already a node at the given location, then if value is a Node class or Dataset it will overwrite the
        data already present at that node, and if value is a single array, it will be merged with it.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError
        elif isinstance(key, str):
            path = NodePath(key)
            return self._set_item(path, value, new_nodes_along_path=True)
        else:
            raise ValueError('Invalid format for key')

    def update(self, other: Dataset | Mapping[Hashable, DataArray | Variable] | Mapping[str, DataTree | DataArray | Variable]) -> None:
        """
        Update this node's children and / or variables.

        Just like `dict.update` this is an in-place operation.
        """
        pass

    def assign(self, items: Mapping[Any, Any] | None=None, **items_kwargs: Any) -> DataTree:
        """
        Assign new data variables or child nodes to a DataTree, returning a new object
        with all the original items in addition to the new ones.

        Parameters
        ----------
        items : mapping of hashable to Any
            Mapping from variable or child node names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataTree, DataArray,
            scalar, or array), they are simply assigned.
        **items_kwargs
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

        Returns
        -------
        dt : DataTree
            A new DataTree with the new variables or children in addition to all the
            existing items.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well-defined.
        Assigning multiple items within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        xarray.Dataset.assign
        pandas.DataFrame.assign
        """
        pass

    def drop_nodes(self: DataTree, names: str | Iterable[str], *, errors: ErrorOptions='raise') -> DataTree:
        """
        Drop child nodes from this node.

        Parameters
        ----------
        names : str or iterable of str
            Name(s) of nodes to drop.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a KeyError if any of the node names
            passed are not present as children of this node. If 'ignore',
            any given names that are present are dropped and no error is raised.

        Returns
        -------
        dropped : DataTree
            A copy of the node with the specified children dropped.
        """
        pass

    @classmethod
    def from_dict(cls, d: MutableMapping[str, Dataset | DataArray | DataTree | None], name: str | None=None) -> DataTree:
        """
        Create a datatree from a dictionary of data objects, organised by paths into the tree.

        Parameters
        ----------
        d : dict-like
            A mapping from path names to xarray.Dataset, xarray.DataArray, or DataTree objects.

            Path names are to be given as unix-like path. If path names containing more than one part are given, new
            tree nodes will be constructed as necessary.

            To assign data to the root node of the tree use "/" as the path.
        name : Hashable | None, optional
            Name for the root node of the tree. Default is None.

        Returns
        -------
        DataTree

        Notes
        -----
        If your dictionary is nested you will need to flatten it before using this method.
        """
        pass

    def to_dict(self) -> dict[str, Dataset]:
        """
        Create a dictionary mapping of absolute node paths to the data contained in those nodes.

        Returns
        -------
        dict[str, Dataset]
        """
        pass

    def __len__(self) -> int:
        return len(self.children) + len(self.data_vars)

    @property
    def indexes(self) -> Indexes[pd.Index]:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this DataTree node has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        DataTree.xindexes
        """
        pass

    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        pass

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        pass

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        pass

    def isomorphic(self, other: DataTree, from_root: bool=False, strict_names: bool=False) -> bool:
        """
        Two DataTrees are considered isomorphic if every node has the same number of children.

        Nothing about the data in each node is checked.

        Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
        such as ``tree1 + tree2``.

        By default this method does not check any part of the tree above the given node.
        Therefore this method can be used as default to check that two subtrees are isomorphic.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is False
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.
        strict_names : bool, optional, default is False
            Whether or not to also check that every node in the tree has the same name as its counterpart in the other
            tree.

        See Also
        --------
        DataTree.equals
        DataTree.identical
        """
        pass

    def equals(self, other: DataTree, from_root: bool=True) -> bool:
        """
        Two DataTrees are equal if they have isomorphic node structures, with matching node names,
        and if they have matching variables and coordinates, all of which are equal.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.equals
        DataTree.isomorphic
        DataTree.identical
        """
        pass

    def identical(self, other: DataTree, from_root=True) -> bool:
        """
        Like equals, but will also check all dataset attributes and the attributes on
        all variables and coordinates.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.identical
        DataTree.isomorphic
        DataTree.equals
        """
        pass

    def filter(self: DataTree, filterfunc: Callable[[DataTree], bool]) -> DataTree:
        """
        Filter nodes according to a specified condition.

        Returns a new tree containing only the nodes in the original tree for which `fitlerfunc(node)` is True.
        Will also contain empty nodes at intermediate positions if required to support leaves.

        Parameters
        ----------
        filterfunc: function
            A function which accepts only one DataTree - the node on which filterfunc will be called.

        Returns
        -------
        DataTree

        See Also
        --------
        match
        pipe
        map_over_subtree
        """
        pass

    def match(self, pattern: str) -> DataTree:
        """
        Return nodes with paths matching pattern.

        Uses unix glob-like syntax for pattern-matching.

        Parameters
        ----------
        pattern: str
            A pattern to match each node path against.

        Returns
        -------
        DataTree

        See Also
        --------
        filter
        pipe
        map_over_subtree

        Examples
        --------
        >>> dt = DataTree.from_dict(
        ...     {
        ...         "/a/A": None,
        ...         "/a/B": None,
        ...         "/b/A": None,
        ...         "/b/B": None,
        ...     }
        ... )
        >>> dt.match("*/B")
        <xarray.DataTree>
        Group: /
        ├── Group: /a
        │   └── Group: /a/B
        └── Group: /b
            └── Group: /b/B
        """
        pass

    def map_over_subtree(self, func: Callable, *args: Iterable[Any], **kwargs: Any) -> DataTree | tuple[DataTree]:
        """
        Apply a function to every dataset in this subtree, returning a new tree which stores the results.

        The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
        descendant nodes. The returned tree will have the same structure as the original subtree.

        func needs to return a Dataset in order to rebuild the subtree.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        subtrees : DataTree, tuple of DataTrees
            One or more subtrees containing results from applying ``func`` to the data at each node.
        """
        pass

    def map_over_subtree_inplace(self, func: Callable, *args: Iterable[Any], **kwargs: Any) -> None:
        """
        Apply a function to every dataset in this subtree, updating data in place.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets,
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.
        """
        pass

    def pipe(self, func: Callable | tuple[Callable, str], *args: Any, **kwargs: Any) -> Any:
        """Apply ``func(self, *args, **kwargs)``

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : callable
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        *args
            positional arguments passed into ``func``.
        **kwargs
            a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : Any
            the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        .. code:: python

            f(g(h(dt), arg1=a), arg2=b, arg3=c)

        You can write

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe((f, "arg2"), arg1=a, arg3=c))

        """
        pass

    def render(self):
        """Print tree structure, including any data stored at each node."""
        pass

    def merge(self, datatree: DataTree) -> DataTree:
        """Merge all the leaves of a second DataTree into this one."""
        pass

    def merge_child_nodes(self, *paths, new_path: T_Path) -> DataTree:
        """Merge a set of child nodes into a single new node."""
        pass

    @property
    def groups(self):
        """Return all netCDF4 groups in the tree, given as a tuple of path-like strings."""
        pass

    def to_netcdf(self, filepath, mode: NetcdfWriteModes='w', encoding=None, unlimited_dims=None, format: T_DataTreeNetcdfTypes | None=None, engine: T_DataTreeNetcdfEngine | None=None, group: str | None=None, compute: bool=True, **kwargs):
        """
        Write datatree contents to a netCDF file.

        Parameters
        ----------
        filepath : str or Path
            Path to which to save this datatree.
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten. Only appies to the root group.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}, ...}``. See ``xarray.Dataset.to_netcdf`` for available
            options.
        unlimited_dims : dict, optional
            Mapping of unlimited dimensions per group that that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding["unlimited_dims"]``.
        format : {"NETCDF4", }, optional
            File format for the resulting netCDF file:

            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API features.
        engine : {"netcdf4", "h5netcdf"}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for "netcdf4" if writing to a file on disk.
        group : str, optional
            Path to the netCDF4 group in the given file to open as the root group
            of the ``DataTree``. Currently, specifying a group is not supported.
        compute : bool, default: True
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
            Currently, ``compute=False`` is not supported.
        kwargs :
            Addional keyword arguments to be passed to ``xarray.Dataset.to_netcdf``
        """
        pass

    def to_zarr(self, store, mode: ZarrWriteModes='w-', encoding=None, consolidated: bool=True, group: str | None=None, compute: Literal[True]=True, **kwargs):
        """
        Write datatree contents to a Zarr store.

        Parameters
        ----------
        store : MutableMapping, str or Path, optional
            Store or path to directory in file system
        mode : {{"w", "w-", "a", "r+", None}, default: "w-"
            Persistence mode: “w” means create (overwrite if exists); “w-” means create (fail if exists);
            “a” means override existing variables (create if does not exist); “r+” means modify existing
            array values only (raise an error if any metadata or shapes would change). The default mode
            is “w-”.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        consolidated : bool
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing metadata for all groups.
        group : str, optional
            Group path. (a.k.a. `path` in zarr terminology.)
        compute : bool, default: True
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later. Metadata
            is always updated eagerly. Currently, ``compute=False`` is not
            supported.
        kwargs :
            Additional keyword arguments to be passed to ``xarray.Dataset.to_zarr``
        """
        pass
