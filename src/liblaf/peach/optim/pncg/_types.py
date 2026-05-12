import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf import jarp
from liblaf.peach.optim.base import State, Stats

from ._hess_damping import HessianDampingState
from ._line_search import LineSearchState
from ._terminate import ConvergenceState

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class PNCGState(State):
    """State tracked by [`PNCG`][liblaf.peach.optim.pncg.PNCG]."""

    params: Vector = jarp.array()
    grad: Vector = jarp.array()
    direction: Vector = jarp.array()
    convergence_state: ConvergenceState = jarp.field()
    hess_damping_state: HessianDampingState = jarp.field()
    line_search_state: LineSearchState = jarp.field()
    n_steps: Integer[Array, ""] = jarp.array(default=jnp.zeros((), jnp.int32))


@jarp.define(kw_only=True)
class PNCGStats(Stats):
    """Stats placeholder for [`PNCG`][liblaf.peach.optim.pncg.PNCG]."""
