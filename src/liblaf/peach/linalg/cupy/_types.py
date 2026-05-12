import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class CupyState(State):
    """State recorded by CuPy-backed linear solvers."""

    params: Vector = jarp.array()
    info: int = -1
    n_steps: int | None = None
    absolute_residual: Scalar = jarp.array(default=jnp.asarray(jnp.nan))
    relative_residual: Scalar = jarp.array(default=jnp.asarray(jnp.nan))


@jarp.define
class CupyStats(Stats):
    """Stats placeholder for CuPy-backed linear solvers."""
