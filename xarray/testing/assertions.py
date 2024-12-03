"""Testing functions exposed to the user API"""
import functools
import warnings
from collections.abc import Hashable
from typing import Union, overload
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.formatting import diff_datatree_repr
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable

@ensure_warnings
def assert_isomorphic(a: DataTree, b: DataTree, from_root: bool=False):
    """
    Two DataTrees are considered isomorphic if every node has the same number of children.

    Nothing about the data or attrs in each node is checked.

    Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
    such as tree1 + tree2.

    By default this function does not check any part of the tree above the given node.
    Therefore this function can be used as default to check that two subtrees are isomorphic.

    Parameters
    ----------
    a : DataTree
        The first object to compare.
    b : DataTree
        The second object to compare.
    from_root : bool, optional, default is False
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    See Also
    --------
    DataTree.isomorphic
    assert_equal
    assert_identical
    """
    if from_root:
        a = a.root
        b = b.root
    
    if len(a.children) != len(b.children):
        raise AssertionError("DataTrees have different number of children")
    
    for child_a, child_b in zip(a.children.values(), b.children.values()):
        assert_isomorphic(child_a, child_b, from_root=False)

def maybe_transpose_dims(a, b, check_dim_order: bool):
    """Helper for assert_equal/allclose/identical"""
    if check_dim_order:
        return a, b
    
    if hasattr(a, 'dims') and hasattr(b, 'dims'):
        if set(a.dims) == set(b.dims):
            return a.transpose(*b.dims), b
    
    return a, b

@ensure_warnings
def assert_equal(a, b, from_root=True, check_dim_order: bool=True):
    """Like :py:func:`numpy.testing.assert_array_equal`, but for xarray
    objects.

    Raises an AssertionError if two objects are not equal. This will match
    data values, dimensions and coordinates, but not names or attributes
    (except for Dataset objects for which the variable names must match).
    Arrays with NaN in the same location are considered equal.

    For DataTree objects, assert_equal is mapped over all Datasets on each node,
    with the DataTrees being equal if both are isomorphic and the corresponding
    Datasets at each node are themselves equal.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray, xarray.Variable, xarray.Coordinates
        or xarray.core.datatree.DataTree. The first object to compare.
    b : xarray.Dataset, xarray.DataArray, xarray.Variable, xarray.Coordinates
        or xarray.core.datatree.DataTree. The second object to compare.
    from_root : bool, optional, default is True
        Only used when comparing DataTree objects. Indicates whether or not to
        first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_identical, assert_allclose, Dataset.equals, DataArray.equals
    numpy.testing.assert_array_equal
    """
    from xarray.core.datatree import DataTree
    
    if isinstance(a, DataTree) and isinstance(b, DataTree):
        assert_isomorphic(a, b, from_root=from_root)
        for node_a, node_b in zip(a.nodes(), b.nodes()):
            assert_equal(node_a.ds, node_b.ds, check_dim_order=check_dim_order)
    else:
        a, b = maybe_transpose_dims(a, b, check_dim_order)
        
        if not a.equals(b):
            raise AssertionError("Objects are not equal")
        
        if isinstance(a, (xr.Dataset, xr.DataArray)):
            assert set(a.coords) == set(b.coords), "Coordinate names do not match"
            
            for name in a.coords:
                assert_equal(a.coords[name], b.coords[name], check_dim_order=check_dim_order)

@ensure_warnings
def assert_identical(a, b, from_root=True):
    """Like :py:func:`xarray.testing.assert_equal`, but also matches the
    objects' names and attributes.

    Raises an AssertionError if two objects are not identical.

    For DataTree objects, assert_identical is mapped over all Datasets on each
    node, with the DataTrees being identical if both are isomorphic and the
    corresponding Datasets at each node are themselves identical.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray, xarray.Variable or xarray.Coordinates
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray, xarray.Variable or xarray.Coordinates
        The second object to compare.
    from_root : bool, optional, default is True
        Only used when comparing DataTree objects. Indicates whether or not to
        first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_equal, assert_allclose, Dataset.equals, DataArray.equals
    """
    from xarray.core.datatree import DataTree
    
    if isinstance(a, DataTree) and isinstance(b, DataTree):
        assert_isomorphic(a, b, from_root=from_root)
        for node_a, node_b in zip(a.nodes(), b.nodes()):
            assert_identical(node_a.ds, node_b.ds)
    else:
        assert_equal(a, b, check_dim_order=True)
        assert a.name == b.name, "Names do not match"
        assert a.attrs == b.attrs, "Attributes do not match"
        
        if isinstance(a, (xr.Dataset, xr.DataArray)):
            for name in a.coords:
                assert_identical(a.coords[name], b.coords[name])

@ensure_warnings
def assert_allclose(a, b, rtol=1e-05, atol=1e-08, decode_bytes=True, check_dim_order: bool=True):
    """Like :py:func:`numpy.testing.assert_allclose`, but for xarray objects.

    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    decode_bytes : bool, optional
        Whether byte dtypes should be decoded to strings as UTF-8 or not.
        This is useful for testing serialization methods on Python 3 that
        return saved strings as bytes.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_identical, assert_equal, numpy.testing.assert_allclose
    """
    a, b = maybe_transpose_dims(a, b, check_dim_order)
    
    if not a.dims == b.dims:
        raise AssertionError(f"Dimensions do not match: {a.dims} != {b.dims}")
    
    if isinstance(a, (xr.Dataset, xr.DataArray)):
        assert set(a.coords) == set(b.coords), "Coordinate names do not match"
        
        for name in a.coords:
            assert_allclose(a.coords[name], b.coords[name], rtol=rtol, atol=atol, 
                            decode_bytes=decode_bytes, check_dim_order=check_dim_order)
    
    if decode_bytes and a.dtype.kind == 'S':
        a = a.astype(str)
        b = b.astype(str)
    
    np.testing.assert_allclose(a.values, b.values, rtol=rtol, atol=atol)

@ensure_warnings
def assert_duckarray_allclose(actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
    """Like `np.testing.assert_allclose`, but for duckarrays."""
    import numpy as np
    from xarray.core.duck_array_ops import allclose_or_equiv
    
    if not allclose_or_equiv(actual, desired, rtol=rtol, atol=atol):
        if err_msg == '':
            err_msg = 'Not equal to tolerance rtol={}, atol={}'.format(rtol, atol)
        if verbose:
            err_msg += '\n' + 'Max absolute difference: ' + str(np.max(np.abs(actual - desired)))
            err_msg += '\n' + 'Max relative difference: ' + str(np.max(np.abs((actual - desired) / desired)))
        raise AssertionError(err_msg)

@ensure_warnings
def assert_duckarray_equal(x, y, err_msg='', verbose=True):
    """Like `np.testing.assert_array_equal`, but for duckarrays"""
    import numpy as np
    from xarray.core.duck_array_ops import array_equiv
    
    if not array_equiv(x, y):
        if err_msg == '':
            err_msg = 'Arrays are not equal'
        if verbose:
            err_msg += '\n' + 'x: ' + str(x)
            err_msg += '\n' + 'y: ' + str(y)
            err_msg += '\n' + 'Difference: ' + str(np.array(x) - np.array(y))
        raise AssertionError(err_msg)

def assert_chunks_equal(a, b):
    """
    Assert that chunksizes along chunked dimensions are equal.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        The first object to compare.
    b : xarray.Dataset or xarray.DataArray
        The second object to compare.
    """
    a_chunks = a.chunks
    b_chunks = b.chunks
    
    if a_chunks is None and b_chunks is None:
        return
    
    if a_chunks is None or b_chunks is None:
        raise AssertionError("One object is chunked while the other is not")
    
    if set(a_chunks.keys()) != set(b_chunks.keys()):
        raise AssertionError("Chunked dimensions do not match")
    
    for dim in a_chunks:
        if a_chunks[dim] != b_chunks[dim]:
            raise AssertionError(f"Chunk sizes for dimension '{dim}' do not match: {a_chunks[dim]} != {b_chunks[dim]}")

def _assert_internal_invariants(xarray_obj: Union[DataArray, Dataset, Variable], check_default_indexes: bool):
    """Validate that an xarray object satisfies its own internal invariants.

    This exists for the benefit of xarray's own test suite, but may be useful
    in external projects if they (ill-advisedly) create objects using xarray's
    private APIs.
    """
    if isinstance(xarray_obj, DataArray):
        assert set(xarray_obj._coords) <= set(xarray_obj._indexes)
        assert set(xarray_obj.dims) <= set(xarray_obj._indexes)
        assert set(xarray_obj.dims) == set(xarray_obj.variable.dims)
        assert set(xarray_obj._indexes) <= set(xarray_obj._coords) | set(xarray_obj.dims)
        
        if check_default_indexes:
            for k, v in xarray_obj._indexes.items():
                assert isinstance(v, pd.Index)
                
    elif isinstance(xarray_obj, Dataset):
        assert set(xarray_obj._coord_names) <= set(xarray_obj._indexes)
        assert set(xarray_obj.dims) <= set(xarray_obj._indexes)
        assert set(xarray_obj._indexes) <= set(xarray_obj._coord_names) | set(xarray_obj.dims)
        
        if check_default_indexes:
            for k, v in xarray_obj._indexes.items():
                assert isinstance(v, pd.Index)
                
    elif isinstance(xarray_obj, Variable):
        assert set(xarray_obj.dims) == set(xarray_obj.data.shape)
        
    else:
        raise TypeError(f"Unexpected type: {type(xarray_obj)}")
