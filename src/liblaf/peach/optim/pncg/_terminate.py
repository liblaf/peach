import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from liblaf import jarp
from liblaf.peach.optim.base import Result

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class ConvergenceState:
    """Gradient-norm convergence state."""

    grad: Vector = jarp.array()
    grad_norm: Scalar = jarp.array(default=jnp.asarray(jnp.nan))
    grad_norm_first: Scalar = jarp.array(default=jnp.asarray(jnp.nan))
    n_steps: Integer[Array, ""] = jarp.array(default=jnp.zeros((), jnp.int32))


@jarp.define(kw_only=True)
class ConvergenceCriteria:
    """Gradient-norm stopping criteria for PNCG."""

    max_steps: Integer[Array, ""] = jarp.array(default=jnp.asarray(1000, jnp.int32))

    atol_primary: Scalar = jarp.array(default=jnp.asarray(0.0))
    rtol_primary: Scalar = jarp.array(default=jnp.asarray(1e-6))

    def _default_atol_secondary(self) -> Scalar:
        return 1e3 * self.atol_primary

    atol_secondary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_secondary, takes_self=True)
    )

    def _default_rtol_secondary(self) -> Scalar:
        return 1e3 * self.rtol_primary

    rtol_secondary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_secondary, takes_self=True)
    )

    def init(self, params: Vector) -> ConvergenceState:
        """Create an empty convergence state shaped like `params`."""
        return ConvergenceState(grad=jnp.full_like(params, jnp.nan))

    @jax.jit(inline=True)
    def update(self, state: ConvergenceState, g: Vector) -> ConvergenceState:
        """Record the current gradient and gradient norms."""
        grad_norm: Scalar = jnp.linalg.norm(g)
        first_grad_norm: Scalar = jnp.where(
            state.n_steps == 0, grad_norm, state.grad_norm_first
        )
        return ConvergenceState(
            grad=g,
            grad_norm=grad_norm,
            grad_norm_first=first_grad_norm,
            n_steps=state.n_steps + 1,
        )

    @jax.jit(inline=True)
    def terminate(self, state: ConvergenceState) -> tuple[Bool[Array, ""], Result]:
        """Return the convergence decision and result code."""
        max_steps_reached: Bool[Array, ""] = state.n_steps >= self.max_steps
        primary_success: Bool[Array, ""] = self.primary_success(state)
        secondary_success: Bool[Array, ""] = self.secondary_success(state)
        done: Bool[Array, ""] = primary_success | max_steps_reached
        result: Result = Result.select(
            [primary_success, secondary_success, max_steps_reached],
            [
                Result.PRIMARY_SUCCESS,
                Result.SECONDARY_SUCCESS,
                Result.MAX_STEPS_REACHED,
            ],
            default=Result.UNKNOWN_ERROR,
        )
        return done, result

    def primary_success(self, state: ConvergenceState) -> Bool[Array, ""]:
        """Check the primary absolute-relative gradient tolerance."""
        return (
            state.grad_norm
            <= self.atol_primary + self.rtol_primary * state.grad_norm_first
        )

    def secondary_success(self, state: ConvergenceState) -> Bool[Array, ""]:
        """Check the looser secondary gradient tolerance."""
        return (
            state.grad_norm
            <= self.atol_secondary + self.rtol_secondary * state.grad_norm_first
        )
