"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_aggregations.py
    pytest --doctest-modules xarray/core/_aggregations.py --accept || true
    pytest --doctest-modules xarray/core/_aggregations.py

This requires [pytest-accept](https://github.com/max-sixty/pytest-accept).
The second run of pytest is deliberate, since the first will return an error
while replacing the doctests.

"""
import collections
import textwrap
from dataclasses import dataclass, field
MODULE_PREAMBLE = '"""Mixin classes with reduction operations."""\n\n# This file was generated using xarray.util.generate_aggregations. Do not edit manually.\n\nfrom __future__ import annotations\n\nfrom collections.abc import Sequence\nfrom typing import TYPE_CHECKING, Any, Callable\n\nfrom xarray.core import duck_array_ops\nfrom xarray.core.options import OPTIONS\nfrom xarray.core.types import Dims, Self\nfrom xarray.core.utils import contains_only_chunked_or_numpy, module_available\n\nif TYPE_CHECKING:\n    from xarray.core.dataarray import DataArray\n    from xarray.core.dataset import Dataset\n\nflox_available = module_available("flox")\n'
NAMED_ARRAY_MODULE_PREAMBLE = '"""Mixin classes with reduction operations."""\n# This file was generated using xarray.util.generate_aggregations. Do not edit manually.\n\nfrom __future__ import annotations\n\nfrom collections.abc import Sequence\nfrom typing import Any, Callable\n\nfrom xarray.core import duck_array_ops\nfrom xarray.core.types import Dims, Self\n'
AGGREGATIONS_PREAMBLE = '\n\nclass {obj}{cls}Aggregations:\n    __slots__ = ()\n\n    def reduce(\n        self,\n        func: Callable[..., Any],\n        dim: Dims = None,\n        *,\n        axis: int | Sequence[int] | None = None,\n        keep_attrs: bool | None = None,\n        keepdims: bool = False,\n        **kwargs: Any,\n    ) -> Self:\n        raise NotImplementedError()'
NAMED_ARRAY_AGGREGATIONS_PREAMBLE = '\n\nclass {obj}{cls}Aggregations:\n    __slots__ = ()\n\n    def reduce(\n        self,\n        func: Callable[..., Any],\n        dim: Dims = None,\n        *,\n        axis: int | Sequence[int] | None = None,\n        keepdims: bool = False,\n        **kwargs: Any,\n    ) -> Self:\n        raise NotImplementedError()'
GROUPBY_PREAMBLE = '\n\nclass {obj}{cls}Aggregations:\n    _obj: {obj}\n\n    def reduce(\n        self,\n        func: Callable[..., Any],\n        dim: Dims = None,\n        *,\n        axis: int | Sequence[int] | None = None,\n        keep_attrs: bool | None = None,\n        keepdims: bool = False,\n        **kwargs: Any,\n    ) -> {obj}:\n        raise NotImplementedError()\n\n    def _flox_reduce(\n        self,\n        dim: Dims,\n        **kwargs: Any,\n    ) -> {obj}:\n        raise NotImplementedError()'
RESAMPLE_PREAMBLE = '\n\nclass {obj}{cls}Aggregations:\n    _obj: {obj}\n\n    def reduce(\n        self,\n        func: Callable[..., Any],\n        dim: Dims = None,\n        *,\n        axis: int | Sequence[int] | None = None,\n        keep_attrs: bool | None = None,\n        keepdims: bool = False,\n        **kwargs: Any,\n    ) -> {obj}:\n        raise NotImplementedError()\n\n    def _flox_reduce(\n        self,\n        dim: Dims,\n        **kwargs: Any,\n    ) -> {obj}:\n        raise NotImplementedError()'
TEMPLATE_REDUCTION_SIGNATURE = '\n    def {method}(\n        self,\n        dim: Dims = None,{kw_only}{extra_kwargs}{keep_attrs}\n        **kwargs: Any,\n    ) -> Self:\n        """\n        Reduce this {obj}\'s data by applying ``{method}`` along some dimension(s).\n\n        Parameters\n        ----------'
TEMPLATE_REDUCTION_SIGNATURE_GROUPBY = '\n    def {method}(\n        self,\n        dim: Dims = None,\n        *,{extra_kwargs}\n        keep_attrs: bool | None = None,\n        **kwargs: Any,\n    ) -> {obj}:\n        """\n        Reduce this {obj}\'s data by applying ``{method}`` along some dimension(s).\n\n        Parameters\n        ----------'
TEMPLATE_RETURNS = '\n        Returns\n        -------\n        reduced : {obj}\n            New {obj} with ``{method}`` applied to its data and the\n            indicated dimension(s) removed'
TEMPLATE_SEE_ALSO = '\n        See Also\n        --------\n{see_also_methods}\n        :ref:`{docref}`\n            User guide on {docref_description}.'
TEMPLATE_NOTES = '\n        Notes\n        -----\n{notes}'
_DIM_DOCSTRING = 'dim : str, Iterable of Hashable, "..." or None, default: None\n    Name of dimension[s] along which to apply ``{method}``. For e.g. ``dim="x"``\n    or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.'
_DIM_DOCSTRING_GROUPBY = 'dim : str, Iterable of Hashable, "..." or None, default: None\n    Name of dimension[s] along which to apply ``{method}``. For e.g. ``dim="x"``\n    or ``dim=["x", "y"]``. If None, will reduce over the {cls} dimensions.\n    If "...", will reduce over all dimensions.'
_SKIPNA_DOCSTRING = 'skipna : bool or None, optional\n    If True, skip missing values (as marked by NaN). By default, only\n    skips missing values for float dtypes; other dtypes either do not\n    have a sentinel missing value (int) or ``skipna=True`` has not been\n    implemented (object, datetime64 or timedelta64).'
_MINCOUNT_DOCSTRING = "min_count : int or None, optional\n    The required number of valid values to perform the operation. If\n    fewer than min_count non-NA values are present the result will be\n    NA. Only used if skipna is set to True or defaults to True for the\n    array's dtype. Changed in version 0.17.0: if specified on an integer\n    array and skipna=True, the result will be a float array."
_DDOF_DOCSTRING = 'ddof : int, default: 0\n    “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,\n    where ``N`` represents the number of elements.'
_KEEP_ATTRS_DOCSTRING = 'keep_attrs : bool or None, optional\n    If True, ``attrs`` will be copied from the original\n    object to the new one.  If False, the new object will be\n    returned without attributes.'
_KWARGS_DOCSTRING = "**kwargs : Any\n    Additional keyword arguments passed on to the appropriate array\n    function for calculating ``{method}`` on this object's data.\n    These could include dask-specific kwargs like ``split_every``."
_NUMERIC_ONLY_NOTES = 'Non-numeric variables will be removed prior to reducing.'
_FLOX_NOTES_TEMPLATE = 'Use the ``flox`` package to significantly speed up {kind} computations,\nespecially with dask arrays. Xarray will use flox by default if installed.\nPass flox-specific keyword arguments in ``**kwargs``.\nSee the `flox documentation <https://flox.readthedocs.io>`_ for more.'
_FLOX_GROUPBY_NOTES = _FLOX_NOTES_TEMPLATE.format(kind='groupby')
_FLOX_RESAMPLE_NOTES = _FLOX_NOTES_TEMPLATE.format(kind='resampling')
ExtraKwarg = collections.namedtuple('ExtraKwarg', 'docs kwarg call example')
skipna = ExtraKwarg(docs=_SKIPNA_DOCSTRING, kwarg='skipna: bool | None = None,', call='skipna=skipna,', example='\n\n        Use ``skipna`` to control whether NaNs are ignored.\n\n        >>> {calculation}(skipna=False)')
min_count = ExtraKwarg(docs=_MINCOUNT_DOCSTRING, kwarg='min_count: int | None = None,', call='min_count=min_count,', example='\n\n        Specify ``min_count`` for finer control over when NaNs are ignored.\n\n        >>> {calculation}(skipna=True, min_count=2)')
ddof = ExtraKwarg(docs=_DDOF_DOCSTRING, kwarg='ddof: int = 0,', call='ddof=ddof,', example='\n\n        Specify ``ddof=1`` for an unbiased estimate.\n\n        >>> {calculation}(skipna=True, ddof=1)')

@dataclass
class DataStructure:
    name: str
    create_example: str
    example_var_name: str
    numeric_only: bool = False
    see_also_modules: tuple[str] = tuple

class Method:

    def __init__(self, name, bool_reduce=False, extra_kwargs=tuple(), numeric_only=False, see_also_modules=('numpy', 'dask.array'), min_flox_version=None):
        self.name = name
        self.extra_kwargs = extra_kwargs
        self.numeric_only = numeric_only
        self.see_also_modules = see_also_modules
        self.min_flox_version = min_flox_version
        if bool_reduce:
            self.array_method = f'array_{name}'
            self.np_example_array = '\n        ...     np.array([True, True, True, True, True, False], dtype=bool)'
        else:
            self.array_method = name
            self.np_example_array = '\n        ...     np.array([1, 2, 3, 0, 2, np.nan])'

@dataclass
class AggregationGenerator:
    _dim_docstring = _DIM_DOCSTRING
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE
    cls: str
    datastructure: DataStructure
    methods: tuple[Method, ...]
    docref: str
    docref_description: str
    example_call_preamble: str
    definition_preamble: str
    has_keep_attrs: bool = True
    notes: str = ''
    preamble: str = field(init=False)

    def __post_init__(self):
        self.preamble = self.definition_preamble.format(
            obj=self.datastructure.name,
            cls=self.cls
        )

class GroupByAggregationGenerator(AggregationGenerator):
    _dim_docstring = _DIM_DOCSTRING_GROUPBY
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE_GROUPBY

class GenericAggregationGenerator(AggregationGenerator):
    pass
AGGREGATION_METHODS = (Method('count', see_also_modules=('pandas.DataFrame', 'dask.dataframe.DataFrame')), Method('all', bool_reduce=True), Method('any', bool_reduce=True), Method('max', extra_kwargs=(skipna,)), Method('min', extra_kwargs=(skipna,)), Method('mean', extra_kwargs=(skipna,), numeric_only=True), Method('prod', extra_kwargs=(skipna, min_count), numeric_only=True), Method('sum', extra_kwargs=(skipna, min_count), numeric_only=True), Method('std', extra_kwargs=(skipna, ddof), numeric_only=True), Method('var', extra_kwargs=(skipna, ddof), numeric_only=True), Method('median', extra_kwargs=(skipna,), numeric_only=True, min_flox_version='0.9.2'), Method('cumsum', extra_kwargs=(skipna,), numeric_only=True), Method('cumprod', extra_kwargs=(skipna,), numeric_only=True))
DATASET_OBJECT = DataStructure(name='Dataset', create_example='\n        >>> da = xr.DataArray({example_array},\n        ...     dims="time",\n        ...     coords=dict(\n        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),\n        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),\n        ...     ),\n        ... )\n        >>> ds = xr.Dataset(dict(da=da))', example_var_name='ds', numeric_only=True, see_also_modules=('DataArray',))
DATAARRAY_OBJECT = DataStructure(name='DataArray', create_example='\n        >>> da = xr.DataArray({example_array},\n        ...     dims="time",\n        ...     coords=dict(\n        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),\n        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),\n        ...     ),\n        ... )', example_var_name='da', numeric_only=False, see_also_modules=('Dataset',))
DATASET_GENERATOR = GenericAggregationGenerator(cls='', datastructure=DATASET_OBJECT, methods=AGGREGATION_METHODS, docref='agg', docref_description='reduction or aggregation operations', example_call_preamble='', definition_preamble=AGGREGATIONS_PREAMBLE)
DATAARRAY_GENERATOR = GenericAggregationGenerator(cls='', datastructure=DATAARRAY_OBJECT, methods=AGGREGATION_METHODS, docref='agg', docref_description='reduction or aggregation operations', example_call_preamble='', definition_preamble=AGGREGATIONS_PREAMBLE)
DATAARRAY_GROUPBY_GENERATOR = GroupByAggregationGenerator(cls='GroupBy', datastructure=DATAARRAY_OBJECT, methods=AGGREGATION_METHODS, docref='groupby', docref_description='groupby operations', example_call_preamble='.groupby("labels")', definition_preamble=GROUPBY_PREAMBLE, notes=_FLOX_GROUPBY_NOTES)
DATAARRAY_RESAMPLE_GENERATOR = GroupByAggregationGenerator(cls='Resample', datastructure=DATAARRAY_OBJECT, methods=AGGREGATION_METHODS, docref='resampling', docref_description='resampling operations', example_call_preamble='.resample(time="3ME")', definition_preamble=RESAMPLE_PREAMBLE, notes=_FLOX_RESAMPLE_NOTES)
DATASET_GROUPBY_GENERATOR = GroupByAggregationGenerator(cls='GroupBy', datastructure=DATASET_OBJECT, methods=AGGREGATION_METHODS, docref='groupby', docref_description='groupby operations', example_call_preamble='.groupby("labels")', definition_preamble=GROUPBY_PREAMBLE, notes=_FLOX_GROUPBY_NOTES)
DATASET_RESAMPLE_GENERATOR = GroupByAggregationGenerator(cls='Resample', datastructure=DATASET_OBJECT, methods=AGGREGATION_METHODS, docref='resampling', docref_description='resampling operations', example_call_preamble='.resample(time="3ME")', definition_preamble=RESAMPLE_PREAMBLE, notes=_FLOX_RESAMPLE_NOTES)
NAMED_ARRAY_OBJECT = DataStructure(name='NamedArray', create_example='\n        >>> from xarray.namedarray.core import NamedArray\n        >>> na = NamedArray(\n        ...     "x",{example_array},\n        ... )', example_var_name='na', numeric_only=False, see_also_modules=('Dataset', 'DataArray'))
NAMED_ARRAY_GENERATOR = GenericAggregationGenerator(cls='', datastructure=NAMED_ARRAY_OBJECT, methods=AGGREGATION_METHODS, docref='agg', docref_description='reduction or aggregation operations', example_call_preamble='', definition_preamble=NAMED_ARRAY_AGGREGATIONS_PREAMBLE, has_keep_attrs=False)
if __name__ == '__main__':
    import os
    from pathlib import Path
    p = Path(os.getcwd())
    write_methods(filepath=p.parent / 'xarray' / 'xarray' / 'core' / '_aggregations.py', generators=[DATASET_GENERATOR, DATAARRAY_GENERATOR, DATASET_GROUPBY_GENERATOR, DATASET_RESAMPLE_GENERATOR, DATAARRAY_GROUPBY_GENERATOR, DATAARRAY_RESAMPLE_GENERATOR], preamble=MODULE_PREAMBLE)
    write_methods(filepath=p.parent / 'xarray' / 'xarray' / 'namedarray' / '_aggregations.py', generators=[NAMED_ARRAY_GENERATOR], preamble=NAMED_ARRAY_MODULE_PREAMBLE)
