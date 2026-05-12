import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf import jarp
from liblaf.peach.linalg.base import Result, Solution, State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class FallbackState(State):
    """State collected while trying multiple linear solvers."""

    init_params: Vector = jarp.array()
    solutions: list[Solution] = jarp.field(factory=list)
    absolute_residuals: Float[Array, " S"] = jarp.field(factory=list)
    relative_residuals: Float[Array, " S"] = jarp.field(factory=list)
    best_index: Integer[Array, ""] = jarp.array(default=jnp.zeros((), dtype=jnp.int32))

    @property
    def best_solution(self) -> Solution:
        """Solution with the smallest absolute residual."""
        return self.solutions[self.best_index]

    @property
    def params(self) -> Vector:
        """Parameters from [`best_solution`][liblaf.peach.linalg.fallback.FallbackState.best_solution]."""
        return self.best_solution.params

    @property
    def result(self) -> Result:
        """Result code from [`best_solution`][liblaf.peach.linalg.fallback.FallbackState.best_solution]."""
        return self.best_solution.result


@jarp.define
class FallbackStats(Stats):
    """Stats placeholder for fallback linear solves."""
