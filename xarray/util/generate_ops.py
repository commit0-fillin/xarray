"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_ops.py > xarray/core/_typed_ops.py

"""
from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Optional
BINOPS_EQNE = (('__eq__', 'nputils.array_eq'), ('__ne__', 'nputils.array_ne'))
BINOPS_CMP = (('__lt__', 'operator.lt'), ('__le__', 'operator.le'), ('__gt__', 'operator.gt'), ('__ge__', 'operator.ge'))
BINOPS_NUM = (('__add__', 'operator.add'), ('__sub__', 'operator.sub'), ('__mul__', 'operator.mul'), ('__pow__', 'operator.pow'), ('__truediv__', 'operator.truediv'), ('__floordiv__', 'operator.floordiv'), ('__mod__', 'operator.mod'), ('__and__', 'operator.and_'), ('__xor__', 'operator.xor'), ('__or__', 'operator.or_'), ('__lshift__', 'operator.lshift'), ('__rshift__', 'operator.rshift'))
BINOPS_REFLEXIVE = (('__radd__', 'operator.add'), ('__rsub__', 'operator.sub'), ('__rmul__', 'operator.mul'), ('__rpow__', 'operator.pow'), ('__rtruediv__', 'operator.truediv'), ('__rfloordiv__', 'operator.floordiv'), ('__rmod__', 'operator.mod'), ('__rand__', 'operator.and_'), ('__rxor__', 'operator.xor'), ('__ror__', 'operator.or_'))
BINOPS_INPLACE = (('__iadd__', 'operator.iadd'), ('__isub__', 'operator.isub'), ('__imul__', 'operator.imul'), ('__ipow__', 'operator.ipow'), ('__itruediv__', 'operator.itruediv'), ('__ifloordiv__', 'operator.ifloordiv'), ('__imod__', 'operator.imod'), ('__iand__', 'operator.iand'), ('__ixor__', 'operator.ixor'), ('__ior__', 'operator.ior'), ('__ilshift__', 'operator.ilshift'), ('__irshift__', 'operator.irshift'))
UNARY_OPS = (('__neg__', 'operator.neg'), ('__pos__', 'operator.pos'), ('__abs__', 'operator.abs'), ('__invert__', 'operator.invert'))
OTHER_UNARY_METHODS = (('round', 'ops.round_'), ('argsort', 'ops.argsort'), ('conj', 'ops.conj'), ('conjugate', 'ops.conjugate'))
required_method_binary = '\n    def _binary_op(\n        self, other: {other_type}, f: Callable, reflexive: bool = False\n    ) -> {return_type}:\n        raise NotImplementedError'
template_binop = '\n    def {method}(self, other: {other_type}) -> {return_type}:{type_ignore}\n        return self._binary_op(other, {func})'
template_binop_overload = '\n    @overload{overload_type_ignore}\n    def {method}(self, other: {overload_type}) -> {overload_type}: ...\n\n    @overload\n    def {method}(self, other: {other_type}) -> {return_type}: ...\n\n    def {method}(self, other: {other_type}) -> {return_type} | {overload_type}:{type_ignore}\n        return self._binary_op(other, {func})'
template_reflexive = '\n    def {method}(self, other: {other_type}) -> {return_type}:\n        return self._binary_op(other, {func}, reflexive=True)'
required_method_inplace = '\n    def _inplace_binary_op(self, other: {other_type}, f: Callable) -> Self:\n        raise NotImplementedError'
template_inplace = '\n    def {method}(self, other: {other_type}) -> Self:{type_ignore}\n        return self._inplace_binary_op(other, {func})'
required_method_unary = '\n    def _unary_op(self, f: Callable, *args: Any, **kwargs: Any) -> Self:\n        raise NotImplementedError'
template_unary = '\n    def {method}(self) -> Self:\n        return self._unary_op({func})'
template_other_unary = '\n    def {method}(self, *args: Any, **kwargs: Any) -> Self:\n        return self._unary_op({func}, *args, **kwargs)'
unhashable = '\n    # When __eq__ is defined but __hash__ is not, then an object is unhashable,\n    # and it should be declared as follows:\n    __hash__: None  # type:ignore[assignment]'
FuncType = Sequence[tuple[Optional[str], Optional[str]]]
OpsType = tuple[FuncType, str, dict[str, str]]
ops_info = {}
ops_info['DatasetOpsMixin'] = binops(other_type='DsCompatible') + inplace(other_type='DsCompatible') + unops()
ops_info['DataArrayOpsMixin'] = binops(other_type='DaCompatible') + inplace(other_type='DaCompatible') + unops()
ops_info['VariableOpsMixin'] = binops_overload(other_type='VarCompatible', overload_type='T_DA') + inplace(other_type='VarCompatible', type_ignore='misc') + unops()
ops_info['DatasetGroupByOpsMixin'] = binops(other_type='Dataset | DataArray', return_type='Dataset')
ops_info['DataArrayGroupByOpsMixin'] = binops(other_type='T_Xarray', return_type='T_Xarray')
MODULE_PREAMBLE = '"""Mixin classes with arithmetic operators."""\n\n# This file was generated using xarray.util.generate_ops. Do not edit manually.\n\nfrom __future__ import annotations\n\nimport operator\nfrom typing import TYPE_CHECKING, Any, Callable, overload\n\nfrom xarray.core import nputils, ops\nfrom xarray.core.types import (\n    DaCompatible,\n    DsCompatible,\n    Self,\n    T_Xarray,\n    VarCompatible,\n)\n\nif TYPE_CHECKING:\n    from xarray.core.dataarray import DataArray\n    from xarray.core.dataset import Dataset\n    from xarray.core.types import T_DataArray as T_DA'
CLASS_PREAMBLE = '{newline}\nclass {cls_name}:\n    __slots__ = ()'
COPY_DOCSTRING = '    {method}.__doc__ = {func}.__doc__'

def render(ops_info: dict[str, list[OpsType]]) -> Iterator[str]:
    """Render the module or stub file."""
    yield MODULE_PREAMBLE

    for cls_name, ops_list in ops_info.items():
        yield CLASS_PREAMBLE.format(newline='\n', cls_name=cls_name)

        for ops, required_method, kwargs in ops_list:
            yield required_method.format(**kwargs)

            for method, func in ops:
                if method is None or func is None:
                    continue

                if 'overload' in kwargs:
                    template = template_binop_overload
                elif method.startswith('__r'):
                    template = template_reflexive
                elif method.startswith('__i'):
                    template = template_inplace
                elif method in dict(UNARY_OPS):
                    template = template_unary
                elif method in dict(OTHER_UNARY_METHODS):
                    template = template_other_unary
                else:
                    template = template_binop

                yield template.format(method=method, func=func, **kwargs)
                yield COPY_DOCSTRING.format(method=method, func=func)

        if '__eq__' in dict(ops):
            yield unhashable
if __name__ == '__main__':
    for line in render(ops_info):
        print(line)
