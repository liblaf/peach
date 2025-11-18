import enum
import time
from typing import Protocol

from jaxtyping import PyTree

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
    end_time: float | None = tree.field(default=None, kw_only=True)
    n_steps: int = tree.field(default=0, kw_only=True)
    start_time: float = tree.field(factory=time.perf_counter, kw_only=True)

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
