from __future__ import annotations

import enum
import time
from typing import Protocol

from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree
from liblaf.peach.tree import FlatDef, TreeView

type Params = PyTree
type Vector = Float[Array, " free"]


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree.define
class State:
    flat_def: FlatDef[Params] | None = tree.field(default=None, kw_only=True)

    params = TreeView[Params]()
    """x"""
    params_flat: Vector = tree.array(default=None, kw_only=True)


@tree.define
class Stats:
    end_time: float | None = tree.field(default=None, kw_only=True)
    start_time: float = tree.field(factory=time.perf_counter, kw_only=True)

    @property
    def time(self) -> float:
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


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
