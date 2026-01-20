import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PyTree

from liblaf.peach import tree
from liblaf.peach.optim.abc import State

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]
type Params = PyTree


@tree.define(kw_only=True)
class PNCGState(State):
    alpha: Scalar = tree.array(default=None)
    """line search step size"""

    beta: Scalar = tree.array(default=None)
    """Dai-Kou (DK) algorithm"""

    decrease: Scalar = tree.array(default=None)
    """Delta E"""

    first_decrease: Scalar = tree.array(default=None)
    """Delta E_0"""

    grad: Vector = tree.array(default=None)
    """g"""

    hess_diag: Vector = tree.array(default=None)
    """diag(H)"""

    hess_quad: Scalar = tree.array(default=None)
    """pHp"""

    params: Vector = tree.array()  # pyright: ignore[reportIncompatibleMethodOverride]
    """x"""

    preconditioner: Vector = tree.array(default=None)
    """P"""

    search_direction: Vector = tree.array()
    """p"""

    # best so far
    best_decrease: Scalar = tree.array(default=jnp.inf)
    # best_grad_norm: Scalar = tree.array(default=jnp.inf)
    best_params: Vector = tree.array(default=None)
    # stagnation detection
    stagnation_counter: Integer[Array, ""] = tree.array(default=0)
    stagnation_restarts: Integer[Array, ""] = tree.array(default=0)
