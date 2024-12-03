from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Generic, TypeVar
from xarray.core.utils import Frozen, is_dict_like
if TYPE_CHECKING:
    from xarray.core.types import T_DataArray

class InvalidTreeError(Exception):
    """Raised when user attempts to create an invalid tree in some way."""

class NotFoundInTreeError(ValueError):
    """Raised when operation can't be completed because one node is not part of the expected tree."""

class NodePath(PurePosixPath):
    """Represents a path from one node to another within a tree."""

    def __init__(self, *pathsegments):
        if sys.version_info >= (3, 12):
            super().__init__(*pathsegments)
        else:
            super().__new__(PurePosixPath, *pathsegments)
        if self.drive:
            raise ValueError('NodePaths cannot have drives')
        if self.root not in ['/', '']:
            raise ValueError('Root of NodePath can only be either "/" or "", with "" meaning the path is relative.')
Tree = TypeVar('Tree', bound='TreeNode')

class TreeNode(Generic[Tree]):
    """
    Base class representing a node of a tree, with methods for traversing and altering the tree.

    This class stores no data, it has only parents and children attributes, and various methods.

    Stores child nodes in an dict, ensuring that equality checks between trees
    and order of child nodes is preserved (since python 3.7).

    Nodes themselves are intrinsically unnamed (do not possess a ._name attribute), but if the node has a parent you can
    find the key it is stored under via the .name property.

    The .parent attribute is read-only: to replace the parent using public API you must set this node as the child of a
    new parent using `new_parent.children[name] = child_node`, or to instead detach from the current parent use
    `child_node.orphan()`.

    This class is intended to be subclassed by DataTree, which will overwrite some of the inherited behaviour,
    in particular to make names an inherent attribute, and allow setting parents directly. The intention is to mirror
    the class structure of xarray.Variable & xarray.DataArray, where Variable is unnamed but DataArray is (optionally)
    named.

    Also allows access to any other node in the tree via unix-like paths, including upwards referencing via '../'.

    (This class is heavily inspired by the anytree library's NodeMixin class.)

    """
    _parent: Tree | None
    _children: dict[str, Tree]

    def __init__(self, children: Mapping[str, Tree] | None=None):
        """Create a parentless node."""
        self._parent = None
        self._children = {}
        if children is not None:
            self.children = children

    @property
    def parent(self) -> Tree | None:
        """Parent of this node."""
        return self._parent

    def _check_loop(self, new_parent: Tree | None) -> None:
        """Checks that assignment of this new parent will not create a cycle."""
        if new_parent is not None:
            if new_parent is self:
                raise InvalidTreeError("Cannot set an object as its own parent.")
            parent = new_parent
            while parent is not None:
                if parent is self:
                    raise InvalidTreeError("Assignment would create a cycle in the tree.")
                parent = parent.parent

    def orphan(self) -> None:
        """Detach this node from its parent."""
        if self._parent is not None:
            self._pre_detach(self._parent)
            for name, child in self._parent._children.items():
                if child is self:
                    del self._parent._children[name]
                    break
            self._parent._post_detach_children({name: self})
            old_parent = self._parent
            self._parent = None
            self._post_detach(old_parent)

    @property
    def children(self: Tree) -> Mapping[str, Tree]:
        """Child nodes of this node, stored under a mapping via their names."""
        return Frozen(self._children)

    @staticmethod
    def _check_children(children: Mapping[str, Tree]) -> None:
        """Check children for correct types and for any duplicates."""
        for name, child in children.items():
            if not isinstance(name, str):
                raise TypeError(f"Child name must be a string, not {type(name)}")
            if not isinstance(child, TreeNode):
                raise TypeError(f"Child must be a TreeNode, not {type(child)}")
        if len(set(children.values())) != len(children):
            raise ValueError("Duplicate child nodes are not allowed")

    def __repr__(self) -> str:
        return f'TreeNode(children={dict(self._children)})'

    def _pre_detach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call before detaching `children`."""
        for child in children.values():
            child._pre_detach(self)

    def _post_detach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call after detaching `children`."""
        for child in children.values():
            child._post_detach(self)

    def _pre_attach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call before attaching `children`."""
        for name, child in children.items():
            child._pre_attach(self, name)

    def _post_attach_children(self: Tree, children: Mapping[str, Tree]) -> None:
        """Method call after attaching `children`."""
        for name, child in children.items():
            child._post_attach(self, name)

    def _iter_parents(self: Tree) -> Iterator[Tree]:
        """Iterate up the tree, starting from the current node's parent."""
        pass

    def iter_lineage(self: Tree) -> tuple[Tree, ...]:
        """Iterate up the tree, starting from the current node."""
        pass

    @property
    def lineage(self: Tree) -> tuple[Tree, ...]:
        """All parent nodes and their parent nodes, starting with the closest."""
        pass

    @property
    def parents(self: Tree) -> tuple[Tree, ...]:
        """All parent nodes and their parent nodes, starting with the closest."""
        pass

    @property
    def ancestors(self: Tree) -> tuple[Tree, ...]:
        """All parent nodes and their parent nodes, starting with the most distant."""
        pass

    @property
    def root(self: Tree) -> Tree:
        """Root node of the tree"""
        pass

    @property
    def is_root(self) -> bool:
        """Whether this node is the tree root."""
        pass

    @property
    def is_leaf(self) -> bool:
        """
        Whether this node is a leaf node.

        Leaf nodes are defined as nodes which have no children.
        """
        pass

    @property
    def leaves(self: Tree) -> tuple[Tree, ...]:
        """
        All leaf nodes.

        Leaf nodes are defined as nodes which have no children.
        """
        pass

    @property
    def siblings(self: Tree) -> dict[str, Tree]:
        """
        Nodes with the same parent as this node.
        """
        pass

    @property
    def subtree(self: Tree) -> Iterator[Tree]:
        """
        An iterator over all nodes in this tree, including both self and all descendants.

        Iterates depth-first.

        See Also
        --------
        DataTree.descendants
        """
        pass

    @property
    def descendants(self: Tree) -> tuple[Tree, ...]:
        """
        Child nodes and all their child nodes.

        Returned in depth-first order.

        See Also
        --------
        DataTree.subtree
        """
        pass

    @property
    def level(self: Tree) -> int:
        """
        Level of this node.

        Level means number of parent nodes above this node before reaching the root.
        The root node is at level 0.

        Returns
        -------
        level : int

        See Also
        --------
        depth
        width
        """
        pass

    @property
    def depth(self: Tree) -> int:
        """
        Maximum level of this tree.

        Measured from the root, which has a depth of 0.

        Returns
        -------
        depth : int

        See Also
        --------
        level
        width
        """
        pass

    @property
    def width(self: Tree) -> int:
        """
        Number of nodes at this level in the tree.

        Includes number of immediate siblings, but also "cousins" in other branches and so-on.

        Returns
        -------
        depth : int

        See Also
        --------
        level
        depth
        """
        pass

    def _pre_detach(self: Tree, parent: Tree) -> None:
        """Method call before detaching from `parent`."""
        pass

    def _post_detach(self: Tree, parent: Tree) -> None:
        """Method call after detaching from `parent`."""
        pass

    def _pre_attach(self: Tree, parent: Tree, name: str) -> None:
        """Method call before attaching to `parent`."""
        pass

    def _post_attach(self: Tree, parent: Tree, name: str) -> None:
        """Method call after attaching to `parent`."""
        pass

    def get(self: Tree, key: str, default: Tree | None=None) -> Tree | None:
        """
        Return the child node with the specified key.

        Only looks for the node within the immediate children of this node,
        not in other nodes of the tree.
        """
        pass

    def _get_item(self: Tree, path: str | NodePath) -> Tree | T_DataArray:
        """
        Returns the object lying at the given path.

        Raises a KeyError if there is no object at the given path.
        """
        pass

    def _set(self: Tree, key: str, val: Tree) -> None:
        """
        Set the child node with the specified key to value.

        Counterpart to the public .get method, and also only works on the immediate node, not other nodes in the tree.
        """
        pass

    def _set_item(self: Tree, path: str | NodePath, item: Tree | T_DataArray, new_nodes_along_path: bool=False, allow_overwrite: bool=True) -> None:
        """
        Set a new item in the tree, overwriting anything already present at that path.

        The given value either forms a new node of the tree or overwrites an
        existing item at that location.

        Parameters
        ----------
        path
        item
        new_nodes_along_path : bool
            If true, then if necessary new nodes will be created along the
            given path, until the tree can reach the specified location.
        allow_overwrite : bool
            Whether or not to overwrite any existing node at the location given
            by path.

        Raises
        ------
        KeyError
            If node cannot be reached, and new_nodes_along_path=False.
            Or if a node already exists at the specified path, and allow_overwrite=False.
        """
        pass

    def __delitem__(self: Tree, key: str):
        """Remove a child node from this tree object."""
        if key in self.children:
            child = self._children[key]
            del self._children[key]
            child.orphan()
        else:
            raise KeyError('Cannot delete')

    def same_tree(self, other: Tree) -> bool:
        """True if other node is in the same tree as this node."""
        pass
AnyNamedNode = TypeVar('AnyNamedNode', bound='NamedNode')

class NamedNode(TreeNode, Generic[Tree]):
    """
    A TreeNode which knows its own name.

    Implements path-like relationships to other nodes in its tree.
    """
    _name: str | None
    _parent: Tree | None
    _children: dict[str, Tree]

    def __init__(self, name=None, children=None):
        super().__init__(children=children)
        self._name = None
        self.name = name

    @property
    def name(self) -> str | None:
        """The name of this node."""
        pass

    def __repr__(self, level=0):
        repr_value = '\t' * level + self.__str__() + '\n'
        for child in self.children:
            repr_value += self.get(child).__repr__(level + 1)
        return repr_value

    def __str__(self) -> str:
        return f"NamedNode('{self.name}')" if self.name else 'NamedNode()'

    def _post_attach(self: AnyNamedNode, parent: AnyNamedNode, name: str) -> None:
        """Ensures child has name attribute corresponding to key under which it has been stored."""
        pass

    @property
    def path(self) -> str:
        """Return the file-like path from the root to this node."""
        pass

    def relative_to(self: NamedNode, other: NamedNode) -> str:
        """
        Compute the relative path from this node to node `other`.

        If other is not in this tree, or it's otherwise impossible, raise a ValueError.
        """
        pass

    def find_common_ancestor(self, other: NamedNode) -> NamedNode:
        """
        Find the first common ancestor of two nodes in the same tree.

        Raise ValueError if they are not in the same tree.
        """
        pass

    def _path_to_ancestor(self, ancestor: NamedNode) -> NodePath:
        """Return the relative path from this node to the given ancestor node"""
        pass
