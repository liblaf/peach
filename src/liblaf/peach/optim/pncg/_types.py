import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.peach.optim.base import State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class PNCGState(State):
    n_steps: Integer[Array, ""] = jarp.array(default=0)

    alpha: Scalar = jarp.array()
    beta: Scalar = jarp.array()
    decrease: Scalar = jarp.array()
    first_decrease: Scalar = jarp.array()

    grad: Vector = jarp.array()
    hess_diag: Vector = jarp.array()
    hess_quad: Scalar = jarp.array()
    params: Vector = jarp.array()
    preconditioner: Vector = jarp.array()
    search_direction: Vector = jarp.array()

    # best so far
    best_decrease: Scalar = jarp.array(default=jnp.inf)
    best_params: Vector = jarp.array()

    # stagnation
    stagnation_counter: Integer[Array, ""] = jarp.array(default=0)
    stagnation_restarts: Integer[Array, ""] = jarp.array(default=0)


@jarp.define(kw_only=True)
class PNCGStats(Stats):
    relative_decrease: Scalar
