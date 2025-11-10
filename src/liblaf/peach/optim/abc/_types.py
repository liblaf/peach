from __future__ import annotations

import enum
from typing import Protocol

from jaxtyping import PyTree

from liblaf.peach import tree_utils

type Params = PyTree


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree_utils.tree
class State:
    pass


@tree_utils.tree
class Stats:
    n_steps: int = 0
    time: float = 0.0


@tree_utils.tree
class OptimizeSolution[StateT: State, StatsT: Stats]:
    result: Result
    params: Params
    state: StateT
    stats: StatsT

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESS
