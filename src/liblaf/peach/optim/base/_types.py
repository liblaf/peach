import enum
import time
from typing import Protocol

import jarp
import jax.tree_util as jtu
import wadler_lindig as wl
from jaxtyping import Array, Float
from liblaf.grapes.rich.repr import rich_repr_fieldz
from liblaf.grapes.wadler_lindig import pdoc_rich_repr
from rich.repr import RichReprResult

from ._objective import Objective

type Vector = Float[Array, " N"]


@jtu.register_static
class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    MAX_STEPS_REACHED = enum.auto()
    NAN = enum.auto()
    STAGNATION = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    def __bool__(self) -> bool:
        return self in {
            Result.SUCCESS,
            Result.PRIMARY_SUCCESS,
            Result.SECONDARY_SUCCESS,
        }


@jarp.define
class State:
    params: Vector = jarp.array(default=None, kw_only=True)


@jarp.define
class Stats:
    _end_time: float | None = jarp.field(repr=False, default=None, kw_only=True)
    _start_time: float = jarp.field(repr=False, factory=time.perf_counter, kw_only=True)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc | None:
        return pdoc_rich_repr(self, **kwargs)

    def __rich_repr__(self) -> RichReprResult:
        yield from rich_repr_fieldz(self)
        yield "time", self.time

    @property
    def time(self) -> float:
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time


class Callback[X, S: State, T: Stats](Protocol):
    def __call__(
        self, objective: Objective[X], model_state: X, opt_state: S, opt_stats: T, /
    ) -> None: ...


@jarp.define
class Solution[S: State, T: Stats]:
    result: Result = jarp.static()
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> bool:
        return bool(self.result)
