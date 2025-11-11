import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree
from liblaf.peach.tree import TreeView, Unflatten

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]
type Params = PyTree


@tree.define
class PNCGState:
    alpha: Scalar = tree.array(default=None)
    """line search step size"""

    beta: Scalar = tree.array(default=jnp.zeros(()))
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

    params = TreeView[Params]()
    """x"""
    params_flat: Vector = tree.array(default=None)

    preconditioner = TreeView[Params]()
    """P"""
    preconditioner_flat: Vector = tree.array(default=None)

    search_direction = TreeView[Params]()
    """p"""
    search_direction_flat: Vector = tree.array(default=None)

    unflatten: Unflatten[Params] | None = None
