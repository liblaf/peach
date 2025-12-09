import optax
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.abc import Params, State, Stats
from liblaf.peach.tree import TreeView

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class OptaxState(State):
    wrapped: optax.OptState = tree.field(default=None, kw_only=True)

    grad = TreeView[Params]()
    grad_flat: Vector = tree.array(default=None, kw_only=True)

    updates = TreeView[Params]()
    updates_flat: Vector = tree.array(default=None, kw_only=True)

    first_grad_norm: Scalar = tree.array(default=None, kw_only=True)


@tree.define
class OptaxStats(Stats):
    pass
