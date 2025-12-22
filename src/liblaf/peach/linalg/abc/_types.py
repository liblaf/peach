from __future__ import annotations

import enum
import time
from typing import Protocol

import liblaf.grapes.rich.repr as grr
import liblaf.grapes.wadler_lindig as gwd
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from rich.repr import RichReprResult

from liblaf.peach import tree
from liblaf.peach.tree import Structure, TreeView

type Params = PyTree
type Vector = Float[Array, " free"]


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, state: StateT, stats: StatsT, /) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    BREAKDOWN = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree.define
class State:
    structure: Structure[Params] | None = tree.field(default=None, kw_only=True)

    params = TreeView[Params]()
    """x"""
    params_flat: Vector = tree.array(default=None, kw_only=True)


@tree.define
class Stats:
    end_time: float | None = tree.field(repr=False, default=None, kw_only=True)
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
class LinearSolution[StateT: State, StatsT: Stats]:
    result: Result
    state: StateT
    stats: StatsT

    @property
    def params(self) -> Params:
        return self.state.params

    @property
    def params_flat(self) -> Vector:
        return self.state.params_flat

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESS
