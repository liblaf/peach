from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import attrs
from jaxtyping import Array, PyTree, Shaped

from liblaf.peach import tree_utils
from liblaf.peach.tree_utils import Unflatten

from ._wrapper import FunctionWrapper


@tree_utils.define
class Objective:
    fun = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=())
    """X -> Scalar"""
    _fun_wrapped: Callable | None = tree_utils.field(default=None, alias="fun")
    _fun_wrapper: Callable | None = None

    grad = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> X"""
    _grad_wrapped: Callable | None = tree_utils.field(default=None, alias="grad")
    _grad_wrapper: Callable | None = None

    hess = FunctionWrapper(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> H"""
    _hess_wrapped: Callable | None = tree_utils.field(default=None, alias="hess")
    _hess_wrapper: Callable | None = None

    hess_diag = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _hess_diag_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_diag"
    )
    _hess_diag_wrapper: Callable | None = None

    hess_prod = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=(0,)
    )
    """X, P -> X"""
    _hess_prod_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_prod"
    )
    _hess_prod_wrapper: Callable | None = None

    hess_quad = FunctionWrapper(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=()
    )
    """X, P -> Scalar"""
    _hess_quad_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_quad"
    )
    _hess_quad_wrapper: Callable | None = None

    value_and_grad = FunctionWrapper(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(1,)
    )
    """X -> Scalar, X"""
    _value_and_grad_wrapped: Callable | None = tree_utils.field(
        default=None, alias="value_and_grad"
    )
    _value_and_grad_wrapper: Callable | None = None

    grad_and_hess_diag = FunctionWrapper(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(0, 1)
    )
    """X -> X, X"""
    _grad_and_hess_diag_wrapped: Callable | None = tree_utils.field(
        default=None, alias="grad_and_hess_diag"
    )
    _grad_and_hess_diag_wrapper: Callable | None = None

    # def __replace__(self, **changes: Any) -> Self:
    #     inst: Self = object.__new__(type(self))
    #     changes = toolz.keymap(lambda k: f"_{k}", changes)
    #     changes = toolz.merge(attrs.asdict(self, recurse=False), changes)
    #     for k, v in changes.items():
    #         object.__setattr__(inst, k, v)
    #     return inst

    @property
    def bounds(
        self,
    ) -> tuple[Shaped[Array, " free"] | None, Shaped[Array, " free"] | None]:
        return self._lower_bound_flat, self._upper_bound_flat

    _flatten: bool = False
    unflatten: Unflatten[PyTree] | None = None
    _lower_bound_flat: Shaped[Array, " free"] | None = None
    _upper_bound_flat: Shaped[Array, " free"] | None = None

    def flatten[T](
        self,
        params: T,
        *,
        fixed_mask: T | None = None,
        n_fixed: int | None = None,
        lower_bound: T | None = None,
        upper_bound: T | None = None,
    ) -> tuple[Self, Shaped[Array, " free"]]:
        flat: Shaped[Array, " free"]
        unflatten: Unflatten[T]
        flat, unflatten = tree_utils.flatten(
            params, fixed_mask=fixed_mask, n_fixed=n_fixed
        )
        lower_bound_flat: Shaped[Array, " free"] | None = (
            None if lower_bound is None else unflatten.flatten(lower_bound)
        )
        upper_bound_flat: Shaped[Array, " free"] | None = (
            None if upper_bound is None else unflatten.flatten(upper_bound)
        )
        return attrs.evolve(
            self,
            flatten=True,
            unflatten=unflatten,
            lower_bound_flat=lower_bound_flat,
            upper_bound_flat=upper_bound_flat,
        ), flat

    _jit: bool = False

    def jit(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, jit=enable)

    _args: Sequence[Any] = ()
    _kwargs: Mapping[str, Any] = {}

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        return attrs.evolve(
            self, args=(*self._args, *args), kwargs={**self._kwargs, **kwargs}
        )

    _timer: bool = False

    def timer(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, timer=enable)

    _with_aux: bool = False

    def with_aux(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, with_aux=enable)


@functools.lru_cache
def _field_aliases(cls: type) -> set[str]:
    return {f.alias for f in attrs.fields(cls)}
