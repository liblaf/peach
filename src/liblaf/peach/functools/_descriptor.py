# ruff: noqa: SLF001
from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, overload

import equinox as eqx

from liblaf import grapes
from liblaf.peach import tree

from ._context import FunctionContext


@tree.define(kw_only=True)
class MethodDescriptor:
    name: str | None = None
    n_outputs: int = 1
    in_structures: Mapping[int, str] = {}
    out_structures: Mapping[int, str] = {}
    _wrapped_name: str | None = tree.field(default=None, repr=False, alias="wrapped")
    _wrapper_name: str | None = tree.field(default=None, repr=False, alias="wrapper")

    @overload
    def __get__(self, instance: None, owner: type, /) -> Self: ...
    @overload
    def __get__(
        self, instance: FunctionContext, owner: type | None = None, /
    ) -> Callable: ...
    def __get__(
        self, instance: FunctionContext | None, owner: type | None = None
    ) -> Self | Callable | None:
        assert self.name is not None
        if instance is None:
            return self
        if (cached := getattr(instance, self.wrapper_name, None)) is not None:
            return cached
        wrapped: Callable | None = getattr(instance, self.wrapped_name, None)
        if wrapped is None:
            return None

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            with_aux: bool = kwargs.pop("_with_aux", instance._with_aux)
            flatten: bool = kwargs.pop("_flatten", instance._flatten)
            args: Sequence[Any] = (*instance._args, *args)
            kwargs = {**instance._kwargs, **kwargs}
            if flatten:
                args = self._unflatten_inputs(args, instance)
            outputs: Sequence[Any] = _as_tuple(wrapped(*args, **kwargs))
            if flatten:
                outputs = self._flatten_outputs(outputs, instance)
            outputs = self._with_aux(outputs, with_aux=with_aux)
            return outputs[0] if len(outputs) == 1 else outputs

        wrapper = self._with_jit(wrapper, instance)
        wrapper = self._with_timer(wrapper, instance)
        functools.update_wrapper(wrapper, wrapped)
        setattr(instance, self.wrapper_name, wrapper)
        return wrapper

    def __set__(self, instance: FunctionContext, value: Callable | None) -> None:
        setattr(instance, self.wrapped_name, value)
        setattr(instance, self.wrapper_name, None)

    def __set_name__(self, owner: type, name: str) -> None:
        if self.name is None:
            self.name = name

    @functools.cached_property
    def wrapped_name(self) -> str:
        if self._wrapped_name is not None:
            return self._wrapped_name
        assert self.name is not None
        return f"_{self.name}_wrapped"

    @functools.cached_property
    def wrapper_name(self) -> str:
        if self._wrapper_name is not None:
            return self._wrapper_name
        assert self.name is not None
        return f"_{self.name}_wrapper"

    def _unflatten_inputs(
        self, inputs: Sequence[Any], context: FunctionContext
    ) -> list[Any]:
        return _unflatten_inputs(
            inputs,
            structures=context._structures,
            structure_mapping=self.in_structures,
        )

    def _flatten_outputs(
        self, outputs: Sequence[Any], context: FunctionContext
    ) -> list[Any]:
        return _flatten_outputs(
            outputs,
            structures=context._structures,
            structure_mapping=self.out_structures,
        )

    def _with_aux(self, outputs: Sequence[Any], *, with_aux: bool) -> Sequence[Any]:
        if with_aux:
            if len(outputs) == self.n_outputs:
                return *outputs, None
            if len(outputs) == self.n_outputs + 1:
                return outputs
            raise ValueError(outputs)
        if len(outputs) == self.n_outputs:
            return outputs
        if len(outputs) == self.n_outputs + 1:
            return outputs[:-1]
        raise ValueError(outputs)

    def _with_jit[C: Callable](self, wrapped: C, context: FunctionContext) -> C:
        if context._jit:
            return eqx.filter_jit(wrapped)  # pyright: ignore[reportReturnType]
        return wrapped

    def _with_timer[C: Callable](self, wrapped: C, context: FunctionContext) -> C:
        if context._timer:
            label: str
            if context.name and self.name:
                label = f"{context.name}.{self.name}"
            elif context.name:
                label = context.name
            elif self.name:
                label = self.name
            else:
                label = wrapped.__name__
            label += "()"
            return grapes.timer(wrapped, label=label)
        return wrapped


def _as_tuple(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, tuple):
        return outputs
    return (outputs,)


@eqx.filter_jit
def _unflatten_inputs(
    inputs: Sequence[Any],
    *,
    structures: Mapping[str, tree.Structure],
    structure_mapping: Mapping[int, str],
) -> list[Any]:
    inputs = list(inputs)
    for i, structure_name in structure_mapping.items():
        structure: tree.Structure = structures[structure_name]
        inputs[i] = structure.unflatten(inputs[i])
    return inputs


@eqx.filter_jit
def _flatten_outputs(
    outputs: Sequence[Any],
    *,
    structures: Mapping[str, tree.Structure],
    structure_mapping: Mapping[int, str],
) -> list[Any]:
    outputs = list(outputs)
    for i, structure_name in structure_mapping.items():
        structure: tree.Structure = structures[structure_name]
        outputs[i] = structure.flatten(outputs[i])
    return outputs
