from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import attrs
import cytoolz as toolz

from liblaf.peach import tree_utils

from ._struct import ObjectiveStruct
from ._wrapper import FunctionWrapper


@tree_utils.tree(frozen=True)
class Objective:
    wrapped: ObjectiveStruct
    _wrapper: ObjectiveStruct = attrs.field(factory=ObjectiveStruct)

    fun = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=())
    """X -> Scalar"""

    grad = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> X"""

    hess = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> H"""

    hess_diag = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""

    hess_prod = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=(0,)
    )
    """X, P -> X"""

    hess_quad = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=()
    )
    """X, P -> Scalar"""

    value_and_grad = FunctionWrapper(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(1,)
    )
    """X -> Scalar, X"""

    grad_and_hess_diag = FunctionWrapper(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(0, 1)
    )
    """X -> X, X"""

    def __init__(self, *args, **kwargs) -> None:
        wrapped_fields: set[str] = _field_aliases(ObjectiveStruct)
        wrapped_kwargs: Mapping[str, Any] = toolz.keyfilter(
            lambda k: k in wrapped_fields, kwargs
        )
        objective_fields: set[str] = _field_aliases(type(self))
        objective_kwargs: Mapping[str, Any] = toolz.keyfilter(
            lambda k: k in objective_fields, kwargs
        )
        if "wrapped" not in objective_kwargs:
            objective_kwargs["wrapped"] = ObjectiveStruct(*args, **wrapped_kwargs)
        objective_kwargs.pop("cache", None)
        self.__attrs_init__(**objective_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def __replace__(self, **changes: Any) -> Self:
        inst: Self = object.__new__(type(self))
        changes = toolz.keymap(lambda k: f"_{k}", changes)
        changes = toolz.merge(attrs.asdict(self, recurse=False), changes)
        for k, v in changes.items():
            object.__setattr__(inst, k, v)
        return inst

    _flatten: bool = False
    _unflatten: Callable[[Any], Any] | None = None

    def flatten(self, unflatten: Callable) -> Self:
        return self.__replace__(flatten=True, unflatten=unflatten)

    _jit: bool = False

    def jit(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return self.__replace__(jit=enable)

    _args: Sequence[Any] = ()
    _kwargs: Mapping[str, Any] = {}

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        return self.__replace__(
            args=(*self._args, *args), kwargs={**self._kwargs, **kwargs}
        )

    _timer: bool = False

    def timer(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return self.__replace__(timer=enable)

    _with_aux: bool = False

    def with_aux(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return self.__replace__(with_aux=enable)


@functools.lru_cache
def _field_aliases(cls: type) -> set[str]:
    return {f.alias for f in attrs.fields(cls)}
