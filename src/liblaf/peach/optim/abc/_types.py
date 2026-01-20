import enum
import time
from typing import TYPE_CHECKING, Protocol

import jax.tree_util as jtu
import wadler_lindig as wl
from jaxtyping import Array, Float, PyTree
from liblaf.grapes.rich.repr import rich_repr_fieldz
from liblaf.grapes.wadler_lindig import pdoc_rich_repr
from rich.repr import RichReprResult

from liblaf.peach import tree
from liblaf.peach.constraints import Constraints
from liblaf.peach.functools import Objective
from liblaf.peach.transforms import IdentityTransform, LinearTransform

type Vector = Float[Array, " N"]
type Params = PyTree


@tree.define
class Problem:
    objective: Objective
    constraints: Constraints = tree.field(factory=Constraints, kw_only=True)
    transform: LinearTransform[Vector, Params] = tree.field(
        factory=IdentityTransform, kw_only=True
    )


class Callback[StateT: State, StatsT: Stats](Protocol):
    def __call__(self, problem: Problem, state: StateT, stats: StatsT, /) -> None: ...


@jtu.register_static
class Result(enum.StrEnum):
    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    MAX_STEPS_REACHED = enum.auto()
    NAN = enum.auto()
    STAGNATION = enum.auto()
    UNKNOWN_ERROR = enum.auto()


@tree.define
class State:
    if TYPE_CHECKING:

        @property
        def params(self) -> Vector: ...


@tree.define
class Stats:
    end_time: float | None = tree.field(repr=False, default=None, kw_only=True)
    n_steps: int = tree.field(default=0, kw_only=True)
    start_time: float = tree.field(repr=False, factory=time.perf_counter, kw_only=True)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc | None:
        return pdoc_rich_repr(self, **kwargs)

    def __rich_repr__(self) -> RichReprResult:
        yield from rich_repr_fieldz(self)
        yield "time", self.time

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
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> bool:
        return self.result in {
            Result.SUCCESS,
            Result.PRIMARY_SUCCESS,
            Result.SECONDARY_SUCCESS,
        }
