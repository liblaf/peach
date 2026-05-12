import abc
from typing import Any, cast, override

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import BaseProblem, LinearSolver, Problem, Result
from liblaf.peach.utils import implemented

from ._types import JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class JaxSolver(LinearSolver[JaxState, JaxStats]):
    """Base class for solvers backed by `jax.scipy.sparse.linalg`."""

    from ._types import JaxState as State
    from ._types import JaxStats as Stats

    type Solution = LinearSolver.Solution[JaxState, JaxStats]

    maxiter: int | None = None

    atol_secondary: Scalar = jarp.array(default=jnp.asarray(0.0))
    rtol_secondary: Scalar = jarp.array(default=jnp.asarray(1e-5))

    def _default_atol_primary(self) -> Scalar:
        return 1e-1 * self.atol_secondary

    def _default_rtol_primary(self) -> Scalar:
        return 1e-1 * self.rtol_secondary

    atol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_primary, takes_self=True)
    )
    rtol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_primary, takes_self=True)
    )

    @override
    def init(self, problem: BaseProblem, params: Vector) -> State:
        """Initialize a JAX solver state."""
        return self.State(params=params)

    @override
    def compute(self, problem: BaseProblem, state: State) -> tuple[State, Result]:
        """Run the wrapped JAX solver and record residual diagnostics."""
        problem: Problem = cast("Problem", problem)
        state.params, state.info = self._wrapped(
            problem.matvec, problem.b, state.params, **self._options(problem)
        )
        residual: Scalar = jnp.linalg.norm(problem.matvec(state.params) - problem.b)
        b_norm: Scalar = jnp.linalg.norm(problem.b)
        state.absolute_residual = residual
        state.relative_residual = residual / b_norm
        result: Result = Result.select(
            [
                residual <= self.atol_primary + self.rtol_primary * b_norm,
                residual <= self.atol_secondary + self.rtol_secondary * b_norm,
            ],
            [Result.PRIMARY_SUCCESS, Result.SECONDARY_SUCCESS],
            default=Result.UNKNOWN_ERROR,
        )
        return state, result

    def _options(self, problem: BaseProblem) -> dict[str, Any]:
        """Build keyword options for the wrapped solver."""
        problem: Problem = cast("Problem", problem)
        maxiter: int = problem.b.size if self.maxiter is None else self.maxiter
        options: dict[str, Any] = {
            "tol": self.rtol_primary,
            "atol": self.atol_primary,
            "maxiter": maxiter,
        }
        if implemented(problem, Problem.precondition):
            options["M"] = problem.precondition
        return options

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Any]:
        """Call the concrete JAX solver implementation."""
        raise NotImplementedError
