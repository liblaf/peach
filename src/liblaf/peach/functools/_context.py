import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Self

import attrs

from liblaf import grapes
from liblaf.peach import tree


@tree.define
class FunctionContext:
    name: str | None = tree.field(default=None, kw_only=True)

    _jit: bool = tree.field(default=False, kw_only=True, alias="jit")

    def jit(self, *, jit: bool = True) -> Self:
        return attrs.evolve(self, jit=jit)

    _args: Sequence[Any] = tree.field(default=(), kw_only=True, alias="args")
    _kwargs: Mapping[str, Any] = tree.field(default={}, kw_only=True, alias="kwargs")

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        args = (*self._args, *args)
        kwargs = {**self._kwargs, **kwargs}
        return attrs.evolve(self, args=args, kwargs=kwargs)

    _timer: bool = tree.field(default=False, kw_only=True, alias="timer")

    def timer(self, *, timer: bool = True) -> Self:
        return attrs.evolve(self, timer=timer)

    def timer_finish(self) -> None:
        _logging_hide = True
        for name, member in inspect.getmembers(self):
            if name.startswith("_"):
                continue
            timer: grapes.BaseTimer | None = grapes.get_timer(member, None)
            if timer is not None and len(timer) > 0:
                timer.finish()

    _with_aux: bool = tree.field(default=False, kw_only=True, alias="with_aux")

    def with_aux(self, *, with_aux: bool = True) -> Self:
        return attrs.evolve(self, with_aux=with_aux)

    _flatten: bool = tree.field(default=False, kw_only=True, alias="flatten")
    _structures: dict[str, tree.Structure] = tree.field(
        factory=dict, kw_only=True, alias="structures"
    )

    def with_structures(
        self, structures: Mapping[str, tree.Structure] = {}, *, flatten: bool = True
    ) -> Self:
        return attrs.evolve(self, flatten=flatten, structures=structures)
