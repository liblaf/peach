import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PyTree

from liblaf.peach import tree
from liblaf.peach.optim.abc import State
from liblaf.peach.tree import TreeView

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]
type Params = PyTree


@tree.define(kw_only=True)
class PNCGState(State):
    alpha: Scalar = tree.array(default=None)
    """line search step size"""

    beta: Scalar = tree.array(default=0.0)
    """Dai-Kou (DK) algorithm"""

    decrease: Scalar = tree.array(default=None)
    """Delta E"""

    first_decrease: Scalar = tree.array(default=None)
    """Delta E_0"""

    grad = TreeView[Params]()
    """g"""
    grad_flat: Vector = tree.array(default=None)

    hess_diag = TreeView[Params]()
    """diag(H)"""
    hess_diag_flat: Vector = tree.array(default=None)

    hess_quad: Scalar = tree.array(default=None)
    """pHp"""

    params = TreeView[Params]()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    """x"""
    params_flat: Vector = tree.array(default=None)

    preconditioner = TreeView[Params]()
    """P"""
    preconditioner_flat: Vector = tree.array(default=None)

    search_direction = TreeView[Params]()
    """p"""
    search_direction_flat: Vector = tree.array(default=None)

    # best so far
    best_grad_norm: Scalar = tree.array(default=jnp.inf)
    best_params = TreeView[Params]()
    best_params_flat: Vector = tree.array(default=None)

    # stagnation detection
    stagnation_counter: Integer[Array, ""] = tree.array(default=0)
    stagnation_restarts: Integer[Array, ""] = tree.array(default=0)
