from __future__ import annotations

import enum
from typing import Protocol

from jaxtyping import PyTree

from liblaf.peach import tree

type Params = PyTree


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    UNKNOWN_ERROR = enum.auto()


class State(Protocol):
    @property
    def params(self) -> Params: ...


class Stats(Protocol):
    start_time: float
    end_time: float | None

    @property
    def time(self) -> float: ...


@tree.define
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
