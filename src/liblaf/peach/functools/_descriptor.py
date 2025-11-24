# ruff: noqa: SLF001

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Self, overload

import attrs
import equinox as eqx
from jaxtyping import Array, PyTree

from liblaf import grapes

if TYPE_CHECKING:
    from ._wrapper import FunctionWrapper


@attrs.define(kw_only=True)
class FunctionDescriptor:
    name: str | None = None
    n_outputs: int = 1
    flatten_outputs: Iterable[int] = (0,)
    unflatten_inputs: Iterable[int] = (0,)

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(
        self, instance: FunctionWrapper, owner: type | None = None
    ) -> Callable | None: ...
    def __get__(
        self, instance: FunctionWrapper | None, owner: type | None = None
    ) -> Callable | Self | None:
        assert self.name is not None
        if instance is None:
            return None
        if (cached := getattr(instance, self.wrapper_name, None)) is not None:
            return cached
        wrapped: Callable | None = getattr(instance, self.wrapped_name, None)
        if wrapped is None:
            return None

        def wrapper(
            *args: Any,
            flatten: bool = instance._flatten,
            with_aux: bool = instance._with_aux,
            **kwargs: Any,
        ) -> Any:
            __tracebackhide__ = True
            args = (*args, *instance._args)
            kwargs = {**instance._kwargs, **kwargs}
            if flatten:
                if self.name == "hess":
                    raise NotImplementedError
                assert instance.flat_def is not None
                args = _unflatten_inputs(
                    args,
                    unflatten=instance.flat_def.unflatten,
                    indices=self.unflatten_inputs,
                )
            outputs: Sequence[Any] = _as_tuple(wrapped(*args, **kwargs))
            if flatten:
                assert instance.flat_def is not None
                outputs = _flatten_outputs(
                    outputs,
                    flatten=instance.flat_def.flatten,
                    indices=self.flatten_outputs,
                )
            outputs = _with_aux(outputs, n_outputs=self.n_outputs, with_aux=with_aux)
            return outputs[0] if len(outputs) == 1 else outputs

        if instance._jit:
            wrapper = eqx.filter_jit(wrapper)
        if instance._timer:
            wrapper = grapes.timer(wrapper, label=f"{self.name}()")
        setattr(instance, self.wrapper_name, wrapper)
        return wrapper

    def __set__(self, instance: FunctionWrapper, value: Callable | None) -> None:
        setattr(instance, self.wrapped_name, value)
        setattr(instance, self.wrapper_name, None)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @functools.cached_property
    def wrapped_name(self) -> str:
        assert self.name is not None
        return f"_{self.name}_wrapped"

    @functools.cached_property
    def wrapper_name(self) -> str:
        assert self.name is not None
        return f"_{self.name}_wrapper"


def _as_tuple(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, tuple):
        return outputs
    return (outputs,)


def _flatten_outputs(
    outputs: Sequence[PyTree],
    flatten: Callable[[PyTree], Array],
    indices: Iterable[int],
) -> tuple[PyTree, ...]:
    outputs = list(outputs)
    for i in indices:
        outputs[i] = flatten(outputs[i])
    return tuple(outputs)


def _unflatten_inputs(
    inputs: Sequence[Array],
    unflatten: Callable[[Array], PyTree],
    indices: Iterable[int],
) -> tuple[PyTree, ...]:
    inputs = list(inputs)
    for i in indices:
        inputs[i] = unflatten(inputs[i])
    return tuple(inputs)


def _with_aux(
    outputs: Sequence[Any], n_outputs: int, *, with_aux: bool
) -> Sequence[Any]:
    if with_aux:
        if len(outputs) == n_outputs:
            return *outputs, None
        if len(outputs) == n_outputs + 1:
            return outputs
        raise ValueError(outputs)
    if len(outputs) == n_outputs:
        return outputs
    if len(outputs) == n_outputs + 1:
        return outputs[:-1]
    raise ValueError(outputs)
