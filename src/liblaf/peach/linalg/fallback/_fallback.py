from typing import cast, override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import (
    BaseProblem,
    LinearSolver,
    Problem,
    Result,
    Solution,
)

from ._types import FallbackState, FallbackStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class FallbackSolver(LinearSolver[FallbackState, FallbackStats]):
    """Try a sequence of linear solvers and keep the best residual."""

    from ._types import FallbackState as State
    from ._types import FallbackStats as Stats

    type Solution = LinearSolver.Solution[State, Stats]

    @staticmethod
    def _default_solvers() -> list[LinearSolver]:
        try:
            import cupy as cp
        except ImportError:
            pass
        else:
            if cp.is_available():
                from liblaf.peach.linalg.cupy import CupyMinRes
                from liblaf.peach.linalg.jax import JaxCG

                return [JaxCG(), CupyMinRes()]

        from liblaf.peach.linalg.jax import JaxCG

        return [JaxCG()]

    solvers: list[LinearSolver] = jarp.field(factory=_default_solvers)

    @override
    def init(self, problem: BaseProblem, params: Vector) -> State:
        """Initialize fallback state with a shared starting vector."""
        return self.State(init_params=params)

    @override
    def compute(self, problem: BaseProblem, state: State) -> tuple[State, Result]:
        """Run solvers until one succeeds, recording residuals for each attempt."""
        problem: Problem = cast("Problem", problem)
        results: list[Result] = []
        absolute_residuals: list[Scalar] = []
        relative_residuals: list[Scalar] = []
        for solver in self.solvers:
            solution: Solution = solver.solve(problem, state.init_params)
            state.solutions.append(solution)
            results.append(solution.result)
            absolute_residual: Scalar = jnp.linalg.norm(
                problem.matvec(solution.state.params) - problem.b
            )
            relative_residual: Scalar = absolute_residual / jnp.linalg.norm(problem.b)
            absolute_residuals.append(absolute_residual)
            relative_residuals.append(relative_residual)
            if solution.success:
                break
        state.absolute_residuals = jnp.asarray(absolute_residuals)
        state.relative_residuals = jnp.asarray(relative_residuals)
        state.best_index = jnp.argmin(state.absolute_residuals)
        return state, state.result
