# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import attrs
import equinox as eqx

from liblaf import grapes
from liblaf.peach import tree_utils

if TYPE_CHECKING:
    from ._objective import Objective


@attrs.define(kw_only=True)
class FunctionWrapper:
    name: str | None = None
    n_outputs: int = 1
    unflatten_inputs: Iterable[int] = (0,)
    flatten_outputs: Iterable[int] = (0,)

    def __get__(
        self, instance: Objective, owner: type | None = None
    ) -> Callable | None:
        assert self.name is not None
        if (cached := getattr(instance._cache, self.name, None)) is not None:
            return cached
        wrapped: Callable | None = getattr(instance.wrapped, self.name, None)
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
                assert instance._unflatten is not None
                args = _unflatten_inputs(
                    args, unflatten=instance._unflatten, indices=self.unflatten_inputs
                )
            outputs: Sequence[Any] = _as_tuple(wrapped(*args, **kwargs))
            if flatten:
                outputs = _flatten_outputs(outputs, indices=self.flatten_outputs)
            outputs = _with_aux(outputs, n_outputs=self.n_outputs, with_aux=with_aux)
            return outputs[0] if len(outputs) == 1 else outputs

        if instance._jit:
            wrapper = eqx.filter_jit(wrapper)
        if instance._timer:
            wrapper = grapes.timer(wrapper, name=f"{self.name}()")
        setattr(instance._cache, self.name, wrapper)
        return wrapper

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name


def _as_tuple(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, tuple):
        return outputs
    return (outputs,)


def _flatten_outputs(outputs: Sequence[Any], indices: Iterable[int]) -> tuple[Any, ...]:
    outputs = list(outputs)
    for i in indices:
        outputs[i], _ = tree_utils.flatten(outputs[i])
    return tuple(outputs)


def _unflatten_inputs(
    inputs: Sequence[Any], unflatten: Callable, indices: Iterable[int]
) -> tuple[Any, ...]:
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
