import enum
import time
from typing import Protocol

import jax.tree_util as jtu
import wadler_lindig as wl
from jaxtyping import Array, Float
from liblaf.grapes.rich.repr import rich_repr_fieldz
from liblaf.grapes.wadler_lindig import pdoc_rich_repr
from rich.repr import RichReprResult

from liblaf.peach import tree

type Vector = Float[Array, " N"]


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


@jtu.register_static
class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    NAN = enum.auto()
    STAGNATION = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree.define
class State:
    params: Vector = tree.field(default=None, kw_only=True)


@tree.define
class Stats:
    end_time: float | None = tree.field(repr=False, default=None, kw_only=True)
    n_steps: int = tree.field(default=0, kw_only=True)
    start_time: float = tree.field(repr=False, factory=time.perf_counter, kw_only=True)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc | None:
        return pdoc_rich_repr(self, **kwargs)

    def __rich_repr__(self) -> RichReprResult:
        yield from rich_repr_fieldz(self)
        yield "time", self.time

    @property
    def time(self) -> float:
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


@tree.define
class OptimizeSolution[StateT: State, StatsT: Stats]:
    result: Result
    state: StateT
    stats: StatsT

    @property
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result is Result.SUCCESS
