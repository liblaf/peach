# ruff: noqa: N803, N806
import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf import jarp

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class HessianDampingState:
    """Adaptive diagonal Hessian-damping state."""

    factor: Scalar = jarp.array()
    hess_diag_mean: Scalar = jarp.array(default=jnp.asarray(jnp.nan))


@jarp.define
class HessianDamping:
    """Adaptive Levenberg-style diagonal Hessian damping."""

    factor_max: Scalar = jarp.array(default=jnp.asarray(0.0))
    factor_min: Scalar = jarp.array(default=jnp.asarray(0.0))
    initial: Scalar = jarp.array(default=jnp.asarray(0.0))

    def init(self) -> HessianDampingState:
        """Create damping state with the initial factor."""
        return HessianDampingState(factor=self.initial)

    @jax.jit(inline=True)
    def hess_diag(
        self, state: HessianDampingState, H_diag: Vector
    ) -> tuple[Vector, HessianDampingState]:
        """Return `abs(H_diag) + factor * mean_positive(abs(H_diag))`."""
        H_diag: Vector = jnp.abs(H_diag)
        H_diag_mean: Scalar = jnp.nanmean(H_diag, where=H_diag > 0)
        state: HessianDampingState = attrs.evolve(state, hess_diag_mean=H_diag_mean)
        return H_diag + state.factor * H_diag_mean, state

    @jax.jit(inline=True)
    def hess_quad(self, state: HessianDampingState, p: Vector, pHp: Scalar) -> Scalar:
        """Add the damping contribution to a Hessian quadratic form."""
        return pHp + state.factor * state.hess_diag_mean * jnp.vdot(p, p)

    @jax.jit(inline=True)
    def update(
        self,
        state: HessianDampingState,
        *,
        actual_decrease: Scalar,
        line_search_steps: Integer[Array, ""],
        predicted_decrease: Scalar,
    ) -> HessianDampingState:
        """Adapt the damping factor from line-search behavior."""
        factor: Scalar = state.factor
        factor *= jnp.where(
            (line_search_steps == 0) & (actual_decrease > predicted_decrease),
            0.5,
            jnp.square(line_search_steps + 1.0),
        )
        factor: Scalar = jnp.clip(factor, self.factor_min, self.factor_max)
        return attrs.evolve(state, factor=factor)
