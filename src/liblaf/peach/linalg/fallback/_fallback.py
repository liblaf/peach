from typing import cast, override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.peach.linalg import utils
from liblaf.peach.linalg.base import (
    LinearSolution,
    LinearSolver,
    LinearSystem,
    Result,
    SupportsMatvec,
)

from ._types import FallbackState, FallbackStats

type Scalar = Float[Array, ""]
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
        result: Result = Result.UNKNOWN_ERROR
        absolute_residuals: list[Scalar] = []
        for solver in self.solvers:
            solution: LinearSolution = solver.solve(system, state.params)
            state.state.append(solution.state)
            stats.stats.append(solution.stats)
            absolute_residuals.append(
                utils.absolute_residual(
                    cast("SupportsMatvec", system).matvec,
                    solution.state.params,
                    system.b,
                )
            )
            result = solution.result
            if solution.success:
                break
        stats.absolute_residuals = jnp.asarray(absolute_residuals)
        idx: Integer[Array, ""] = jnp.argmin(stats.absolute_residuals)
        state.params = state.state[idx].params
        stats.absolute_residual = stats.absolute_residuals[idx]
        return state, stats, result
