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
    factor: Scalar = jarp.array()
    hess_diag_mean: Scalar = jarp.array(default=jnp.asarray(jnp.nan))


@jarp.define
class HessianDamping:
    factor_max: Scalar = jarp.array(default=jnp.asarray(1e1))
    initial: Scalar = jarp.array(default=jnp.asarray(1.0))

    def init(self) -> HessianDampingState:
        return HessianDampingState(factor=self.initial)

    @jax.jit(inline=True)
    def hess_diag(
        self, state: HessianDampingState, H_diag: Vector
    ) -> tuple[Vector, HessianDampingState]:
        H_diag: Vector = jnp.abs(H_diag)
        H_diag_mean: Scalar = jnp.nanmean(H_diag, where=H_diag > 0)
        state: HessianDampingState = attrs.evolve(state, hess_diag_mean=H_diag_mean)
        return H_diag + state.factor * H_diag_mean, state

    @jax.jit(inline=True)
    def hess_quad(self, state: HessianDampingState, p: Vector, pHp: Scalar) -> Scalar:
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
        factor: Scalar = state.factor
        factor *= jnp.select(
            [
                (line_search_steps == 0) & (actual_decrease > predicted_decrease),
                line_search_steps > 2,
            ],
            [0.5, jnp.asarray(line_search_steps, float)],
            default=1.0,
        )
        factor: Scalar = jnp.minimum(factor, self.factor_max)
        return attrs.evolve(state, factor=factor)
