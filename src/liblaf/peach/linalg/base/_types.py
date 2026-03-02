from __future__ import annotations

import enum
import time
from typing import Protocol

import jarp
import liblaf.grapes.rich.repr as grr
import liblaf.grapes.wadler_lindig as gwd
import wadler_lindig as wl
from jaxtyping import Array, Float
from rich.repr import RichReprResult

from ._system import LinearSystem

type Vector = Float[Array, " free"]


class Callback[Params, StateT: State, StatsT: Stats](Protocol):
    def __call__(
        self, system: LinearSystem[Params], state: StateT, stats: StatsT, /
    ) -> None: ...


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    BREAKDOWN = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@jarp.define
class State:
    params: Vector = jarp.array(default=None, kw_only=True)


@jarp.define
class Stats:
    _end_time: float | None = jarp.field(repr=False, default=None, kw_only=True)
    _start_time: float = jarp.field(repr=False, factory=time.perf_counter, kw_only=True)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc | None:
        return gwd.pdoc_rich_repr(self, **kwargs)

    def __rich_repr__(self) -> RichReprResult:
        yield from grr.rich_repr_fieldz(self)
        yield "time", self.time

    @property
    def time(self) -> float:
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time


@jarp.define
class LinearSolution[StateT: State, StatsT: Stats]:
    result: Result
    state: StateT
    stats: StatsT

    @property
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESS
