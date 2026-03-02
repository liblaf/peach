import time

import jarp
from jaxtyping import PyTree

from ._system import LinearSystem
from ._types import Result, State, Stats


@jarp.define
class LinearSolver[StateT: State, StatsT: Stats]:
    from ._types import LinearSolution, State, Stats

    def init[Params](
        self, system: LinearSystem[Params], params: PyTree
    ) -> tuple[StateT, StatsT]:
        raise NotImplementedError

    def compute[Params](
        self, system: LinearSystem[Params], state: StateT, stats: StatsT
    ) -> tuple[StateT, StatsT, Result]:
        raise NotImplementedError

    def postprocess[Params](
        self,
        system: LinearSystem[Params],  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> LinearSolution[StateT, StatsT]:
        stats._end_time = time.perf_counter()  # noqa: SLF001
        return LinearSolver.LinearSolution(result=result, state=state, stats=stats)

    def solve[Params](
        self,
        system: LinearSystem[Params],
        params: PyTree,
    ) -> LinearSolution[StateT, StatsT]:
        state: StateT
        stats: StatsT
        state, stats = self.init(system, params)
        result: Result
        state, stats, result = self.compute(system, state, stats)
        return self.postprocess(system, state, stats, result)
