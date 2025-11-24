import abc
import time
from collections.abc import Iterable
from typing import NamedTuple

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg.system import LinearSystem

from ._types import Callback, LinearSolution, Params, Result, State, Stats


class SetupResult[StateT: State, StatsT: Stats](NamedTuple):
    system: LinearSystem
    constraints: list[Constraint]
    state: StateT
    stats: StatsT


@tree.define
class LinearSolver[StateT: State, StatsT: Stats](abc.ABC):
    from ._types import LinearSolution as Solution
    from ._types import State, Stats

    jit: bool = False
    timer: bool = False

    @abc.abstractmethod
    def setup(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[StateT, StatsT]:
        raise NotImplementedError

    def finalize(
        self,
        system: LinearSystem,  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> LinearSolution[StateT, StatsT]:
        stats.end_time = time.perf_counter()
        return LinearSolution(state=state, stats=stats, result=result)

    @abc.abstractmethod
    def solve(
        self,
        system: LinearSystem,
        params: Params,
        *,
        callback: Callback[StateT, StatsT] | None = None,
        constraints: Iterable[Constraint] = (),
    ) -> LinearSolution[StateT, StatsT]:
        raise NotImplementedError
