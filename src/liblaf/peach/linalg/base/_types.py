from __future__ import annotations

import enum
import time
from typing import Protocol

from jaxtyping import Array, Float

from liblaf import jarp

from ._system import LinearSystem

type Vector = Float[Array, " free"]


class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    BREAKDOWN = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    def __bool__(self) -> bool:
        return self is Result.SUCCESS


@jarp.define
class State:
    params: Vector = jarp.array(default=None, kw_only=True)


@jarp.define
class Stats:
    _end_time: float | None = jarp.field(repr=False, default=None, kw_only=True)
    _start_time: float = jarp.field(repr=False, factory=time.perf_counter, kw_only=True)

    @property
    def time(self) -> float:
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time


class Callback[P: LinearSystem, S: State, T: Stats](Protocol):
    def __call__(self, system: P, state: S, stats: T, /) -> None: ...


@jarp.define
class LinearSolution[S: State, T: Stats]:
    result: Result
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result is Result.SUCCESS
