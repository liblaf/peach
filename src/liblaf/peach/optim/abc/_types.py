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
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


class State(Protocol):
    @property
    def params(self) -> Params:
        raise NotImplementedError


class Stats(Protocol):
    n_steps: int = 0
    time: float = 0.0


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
