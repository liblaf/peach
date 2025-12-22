from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self

import attrs
from jaxtyping import Array, PyTree, Shaped

from liblaf import grapes
from liblaf.peach import tree
from liblaf.peach.constraints import Constraint, FixedConstraint
from liblaf.peach.tree import Structure


@tree.define
class FunctionWrapper:
    structure: Structure[PyTree] | None = tree.field(default=None, kw_only=True)
    _flatten: bool = tree.field(default=False, kw_only=True, alias="flatten")

    def flatten(
        self, params: PyTree, *, constraints: Iterable[Constraint] = ()
    ) -> tuple[Self, Shaped[Array, " free"], list[Constraint]]:
        fixed_constr: list[FixedConstraint] = []
        other_constr: list[Constraint] = []
        for c in constraints:
            if isinstance(c, FixedConstraint):
                fixed_constr.append(c)
            else:
                other_constr.append(c)
        if len(fixed_constr) > 1:
            raise NotImplementedError
        fixed_mask: PyTree | None = fixed_constr[0].mask if fixed_constr else None
        params_flat: Shaped[Array, " free"]
        structure: Structure[PyTree]
        params_flat, structure = tree.flatten(params, fixed_mask=fixed_mask)
        self_new: Self = attrs.evolve(self, flatten=True, structure=structure)
        constr_flat: list[Constraint] = [
            constr.flatten(structure) for constr in other_constr
        ]
        return self_new, params_flat, constr_flat

    _jit: bool = tree.field(default=False, kw_only=True, alias="jit")

    def jit(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, jit=enable)

    _args: Sequence[Any] = tree.field(default=(), kw_only=True, alias="args")
    _kwargs: Mapping[str, Any] = tree.field(factory=dict, kw_only=True, alias="kwargs")

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        return attrs.evolve(
            self, args=(*self._args, *args), kwargs={**self._kwargs, **kwargs}
        )

    _timer: bool = tree.field(default=False, kw_only=True, alias="timer")

    def timer(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, timer=enable)

    def timer_finish(self) -> None:
        _logging_hide = True
        for name, member in inspect.getmembers(self):
            if name.startswith("_"):
                continue
            timer: grapes.BaseTimer | None = grapes.get_timer(member, None)
            if timer is not None and len(timer) > 0:
                timer.finish()

    _with_aux: bool = tree.field(default=False, kw_only=True, alias="with_aux")

    def with_aux(self, enable: bool = True) -> Self:  # noqa: FBT001, FBT002
        return attrs.evolve(self, with_aux=enable)
