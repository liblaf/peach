import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class JaxState(State):
    """State recorded by JAX-backed linear solvers."""

    params: Vector
    info: int | None = None
    absolute_residual: Scalar = jarp.array(default=jnp.asarray(jnp.nan))
    relative_residual: Scalar = jarp.array(default=jnp.asarray(jnp.nan))


@jarp.define
class JaxStats(Stats):
    """Stats placeholder for JAX-backed linear solvers."""
