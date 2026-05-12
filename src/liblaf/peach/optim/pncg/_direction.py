# ruff: noqa: N803, N806
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class DirectionUpdate:
    """Dai-Kou nonlinear conjugate-gradient direction update."""

    @jax.jit(inline=True)
    def __call__(
        self,
        g: Vector,
        g_prev: Vector,
        P: Vector,
        p_prev: Vector,
        *,
        restart: bool = False,
    ) -> Vector:
        """Compute a descent direction, restarting when requested."""
        return jax.lax.cond(
            restart,
            self._compute_direction_restart,
            self._compute_direction,
            g,
            g_prev,
            P,
            p_prev,
        )

    @jax.jit(inline=True)
    def _compute_direction(
        self, g: Vector, g_prev: Vector, P: Vector, p_prev: Vector
    ) -> Scalar:
        beta: Scalar = dai_kou_plus(g, g_prev, P, p_prev)
        p: Vector = jnp.where(beta == 0.0, -P * g, -P * g + beta * p_prev)
        p: Vector = jnp.where(jnp.vdot(p, g) < 0, p, -P * g)
        return p

    @jax.jit(inline=True)
    def _compute_direction_restart(
        self, g: Vector, g_prev: Vector, P: Vector, p_prev: Vector
    ) -> Vector:
        del g_prev, p_prev
        p: Vector = -P * g
        return p


@jax.jit(inline=True)
def dai_kou(g: Vector, g_prev: Vector, P: Vector, p_prev: Vector) -> Scalar:
    """Compute the Dai-Kou conjugacy coefficient."""
    y: Vector = g - g_prev
    Py: Vector = P * y
    yTp: Scalar = jnp.vdot(y, p_prev)
    beta: Scalar = (jnp.vdot(g, Py) - jnp.vdot(y, Py) * jnp.vdot(p_prev, g) / yTp) / yTp
    return beta


@jax.jit(inline=True)
def dai_kou_plus(g: Vector, g_prev: Vector, P: Vector, p_prev: Vector) -> Scalar:
    """Compute the safeguarded nonnegative Dai-Kou coefficient."""
    beta: Scalar = dai_kou(g, g_prev, P, p_prev)
    beta: Scalar = jnp.maximum(beta, 0.0)
    beta: Scalar = jnp.where(beta > 10.0, 0.0, beta)
    return beta
