from __future__ import annotations

import enum
import time
from typing import Protocol

from jaxtyping import PyTree

from liblaf.peach import tree_utils

type Params = PyTree


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


class State(Protocol):
    @property
    def params(self) -> Params: ...
    @property
    def b(self) -> Params: ...


class Stats(Protocol):
    start_time: float
    end_time: float | None

    @property
    def time(self) -> float:
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


@tree_utils.define
class LinearSolution[StateT: State, StatsT: Stats]:
    result: Result
    state: StateT
    stats: StatsT

    @property
    def params(self) -> Params:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESS
