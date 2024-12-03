"""Functions for converting to and from xarray objects
"""
from collections import Counter
import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
iris_forbidden_keys = {'standard_name', 'long_name', 'units', 'bounds', 'axis', 'calendar', 'leap_month', 'leap_year', 'month_lengths', 'coordinates', 'grid_mapping', 'climatology', 'cell_methods', 'formula_terms', 'compress', 'missing_value', 'add_offset', 'scale_factor', 'valid_max', 'valid_min', 'valid_range', '_FillValue'}
cell_methods_strings = {'point', 'sum', 'maximum', 'median', 'mid_range', 'minimum', 'mean', 'mode', 'standard_deviation', 'variance'}

def _filter_attrs(attrs, ignored_attrs):
    """Return attrs that are not in ignored_attrs"""
    return {k: v for k, v in attrs.items() if k not in ignored_attrs}

def _pick_attrs(attrs, keys):
    """Return attrs with keys in keys list"""
    return {k: attrs[k] for k in keys if k in attrs}

def _get_iris_args(attrs):
    """Converts the xarray attrs into args that can be passed into Iris"""
    iris_args = {}
    if 'standard_name' in attrs:
        iris_args['standard_name'] = attrs['standard_name']
    if 'long_name' in attrs:
        iris_args['long_name'] = attrs['long_name']
    if 'units' in attrs:
        iris_args['units'] = attrs['units']
    if 'calendar' in attrs:
        iris_args['calendar'] = attrs['calendar']
    return iris_args

def to_iris(dataarray):
    """Convert a DataArray into an Iris Cube"""
    try:
        import iris
    except ImportError:
        raise ImportError("iris is required for this function")
    
    # Get the data
    data = dataarray.values

    # Get the dimensions
    dim_coords = []
    for dim in dataarray.dims:
        coord = dataarray.coords[dim]
        iris_coord = iris.coords.DimCoord(coord.values,
                                          standard_name=coord.attrs.get('standard_name'),
                                          long_name=coord.attrs.get('long_name', dim),
                                          var_name=dim,
                                          units=coord.attrs.get('units'))
        dim_coords.append((iris_coord, dataarray.get_axis_num(dim)))

    # Create the cube
    iris_args = _get_iris_args(dataarray.attrs)
    cube = iris.cube.Cube(data, dim_coords_and_dims=dim_coords, **iris_args)

    # Add any auxiliary coordinates
    for name, coord in dataarray.coords.items():
        if name not in dataarray.dims:
            iris_coord = iris.coords.AuxCoord(coord.values,
                                              standard_name=coord.attrs.get('standard_name'),
                                              long_name=coord.attrs.get('long_name', name),
                                              var_name=name,
                                              units=coord.attrs.get('units'))
            cube.add_aux_coord(iris_coord)

    return cube

def _iris_obj_to_attrs(obj):
    """Return a dictionary of attrs when given an Iris object"""
    attrs = {}
    if hasattr(obj, 'standard_name') and obj.standard_name is not None:
        attrs['standard_name'] = obj.standard_name
    if hasattr(obj, 'long_name') and obj.long_name is not None:
        attrs['long_name'] = obj.long_name
    if hasattr(obj, 'var_name') and obj.var_name is not None:
        attrs['var_name'] = obj.var_name
    if hasattr(obj, 'units') and obj.units is not None:
        attrs['units'] = str(obj.units)
    if hasattr(obj, 'attributes'):
        attrs.update(obj.attributes)
    return attrs

def _iris_cell_methods_to_str(cell_methods_obj):
    """Converts an Iris cell methods into a string"""
    cell_methods = []
    for cm in cell_methods_obj:
        method = f"{cm.method} over {' '.join(cm.coord_names)}"
        if cm.intervals:
            method += f" (interval: {' '.join(cm.intervals)})"
        if cm.comments:
            method += f" (comment: {' '.join(cm.comments)})"
        cell_methods.append(method)
    return ' '.join(cell_methods)

def _name(iris_obj, default='unknown'):
    """Mimics `iris_obj.name()` but with different name resolution order.

    Similar to iris_obj.name() method, but using iris_obj.var_name first to
    enable roundtripping.
    """
    if hasattr(iris_obj, 'var_name') and iris_obj.var_name is not None:
        return iris_obj.var_name
    elif hasattr(iris_obj, 'standard_name') and iris_obj.standard_name is not None:
        return iris_obj.standard_name
    elif hasattr(iris_obj, 'long_name') and iris_obj.long_name is not None:
        return iris_obj.long_name
    else:
        return default

def from_iris(cube):
    """Convert an Iris cube into a DataArray"""
    import xarray as xr
    import numpy as np

    # Get the main data
    data = cube.data

    # Get the dimensions and coordinates
    dims = []
    coords = {}
    for dim_coord in cube.dim_coords:
        name = _name(dim_coord)
        dims.append(name)
        coords[name] = (name, dim_coord.points, _iris_obj_to_attrs(dim_coord))

    # Add any auxiliary coordinates
    for aux_coord in cube.aux_coords:
        name = _name(aux_coord)
        if name not in coords:
            coords[name] = (aux_coord.points.shape, aux_coord.points, _iris_obj_to_attrs(aux_coord))

    # Create the DataArray
    da = xr.DataArray(data, dims=dims, coords=coords, name=_name(cube))

    # Add the attributes
    da.attrs.update(_iris_obj_to_attrs(cube))

    # Add cell methods if present
    if cube.cell_methods:
        da.attrs['cell_methods'] = _iris_cell_methods_to_str(cube.cell_methods)

    return da
