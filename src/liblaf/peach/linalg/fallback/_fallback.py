from typing import override

import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import LinearSolution, LinearSolver, LinearSystem, Result

from ._types import FallbackState, FallbackStats

type Vector = Float[Array, " N"]


@jarp.define
class FallbackSolver(LinearSolver):
    from ._types import FallbackState as State
    from ._types import FallbackStats as Stats

    type Solution = LinearSolution[State, Stats]

    @staticmethod
    def _default_solvers() -> list[LinearSolver]:
        from liblaf.peach.linalg.cupy import CupyMinRes
        from liblaf.peach.linalg.jax import JaxCG

        return [JaxCG(), CupyMinRes()]

    solvers: list[LinearSolver] = jarp.field(factory=_default_solvers)

    @override
    def init(self, system: LinearSystem, params: Vector) -> tuple[State, Stats]:
        state: FallbackState = FallbackState(params=params)
        stats: FallbackStats = FallbackStats()
        return state, stats

    @override
    def compute(
        self, system: LinearSystem, state: State, stats: Stats
    ) -> tuple[State, Stats, Result]:
        params: Vector = state.params
        result: Result = Result.UNKNOWN_ERROR
        for solver in self.solvers:
            solution: LinearSolution = solver.solve(system, state.params)
            state.state.append(solution.state)
            stats.stats.append(solution.stats)
            params = solution.params
            result = solution.result
            if solution.success:
                break
        state.params = params
        return state, stats, result
