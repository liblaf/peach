from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import attrs
from jaxtyping import Array, PyTree, Shaped

from liblaf.peach import tree_utils
from liblaf.peach.tree_utils import Unflatten


@tree_utils.define
class FunctionWrapper:
    @property
    def bounds(
        self,
    ) -> tuple[Shaped[Array, " free"] | None, Shaped[Array, " free"] | None]:
        return self._lower_bound_flat, self._upper_bound_flat

    _flatten: bool = tree_utils.field(default=False, kw_only=True, alias="flatten")
    unflatten: Unflatten[PyTree] | None = tree_utils.field(default=None, kw_only=True)
    _lower_bound_flat: Shaped[Array, " free"] | None = tree_utils.field(
        default=None, kw_only=True, alias="lower_bound_flat"
    )
    _upper_bound_flat: Shaped[Array, " free"] | None = tree_utils.field(
        default=None, kw_only=True, alias="upper_bound_flat"
    )

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

    _jit: bool = tree_utils.field(default=False, kw_only=True, alias="jit")

    def jit(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, jit=enable)

    _args: Sequence[Any] = tree_utils.field(default=(), kw_only=True, alias="args")
    _kwargs: Mapping[str, Any] = tree_utils.field(
        factory=dict, kw_only=True, alias="kwargs"
    )

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        return attrs.evolve(
            self, args=(*self._args, *args), kwargs={**self._kwargs, **kwargs}
        )

    _timer: bool = tree_utils.field(default=False, kw_only=True, alias="timer")

    def timer(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, timer=enable)

    _with_aux: bool = tree_utils.field(default=False, kw_only=True, alias="with_aux")

    def with_aux(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, with_aux=enable)
