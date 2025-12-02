import enum
import time
from typing import Protocol

import liblaf.grapes.rich.repr as grr
import liblaf.grapes.wadler_lindig as gwd
import wadler_lindig as wl
from jaxtyping import PyTree
from rich.repr import RichReprResult

from liblaf.peach import tree

type Params = PyTree


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree.define
class State:
    @property
    def params(self) -> Params:
        raise NotImplementedError


@tree.define
class Stats:
    end_time: float | None = tree.field(repr=False, default=None, kw_only=True)
    n_steps: int = tree.field(default=0, kw_only=True)
    start_time: float = tree.field(repr=False, factory=time.perf_counter, kw_only=True)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc | None:
        return gwd.pdoc_rich_repr(self, **kwargs)

    def __rich_repr__(self) -> RichReprResult:
        yield from grr.rich_repr_fieldz(self)
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
    def params(self) -> Params:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESS
