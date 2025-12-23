import abc
import functools
import time
from collections.abc import Iterable
from typing import NamedTuple

from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg.system import LinearSystem

from ._types import Callback, LinearSolution, Params, Result, State, Stats

type Vector = Float[Array, " free"]


class SetupResult[StateT: State, StatsT: Stats](NamedTuple):
    system: LinearSystem
    constraints: list[Constraint]
    state: StateT
    stats: StatsT


@tree.define
class LinearSolver[StateT: State, StatsT: Stats](abc.ABC):
    from ._types import LinearSolution as Solution
    from ._types import State, Stats

    jit: bool = tree.field(default=False, kw_only=True)
    timer: bool = tree.field(default=False, kw_only=True)

    @functools.cached_property
    def name(self) -> str:
        return type(self).__name__

    def setup(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[StateT, StatsT]:
        params_flat: Vector
        system, params_flat, constraints = system.flatten(
            params, constraints=constraints
        )
        state = self.State(params_flat=params_flat, structure=system.structure)
        if self.jit:
            system = system.jit()
        if self.timer:
            system = system.timer()
        return SetupResult(system, constraints, state, self.Stats())  # pyright: ignore[reportReturnType]

    def finalize(
        self, system: LinearSystem, state: StateT, stats: StatsT, result: Result
    ) -> LinearSolution[StateT, StatsT]:
        stats.end_time = time.perf_counter()
        system.timer_finish()
        return LinearSolution(state=state, stats=stats, result=result)

    def solve(
        self,
        system: LinearSystem,
        params: Params,
        *,
        callback: Callback[StateT, StatsT] | None = None,
        constraints: Iterable[Constraint] = (),
    ) -> LinearSolution[StateT, StatsT]:
        state: StateT
        stats: StatsT
        system, constraints, state, stats = self.setup(
            system, params, constraints=constraints
        )
        result: Result
        state, stats, result = self._solve(
            system, state, stats, callback=callback, constraints=constraints
        )
        return self.finalize(system, state, stats, result)

    @abc.abstractmethod
    def _solve(
        self,
        system: LinearSystem,
        state: StateT,
        stats: StatsT,
        *,
        callback: Callback[StateT, StatsT] | None = None,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[StateT, StatsT, Result]:
        raise NotImplementedError
