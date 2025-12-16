import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer

from liblaf.peach import tree
from liblaf.peach.optim.abc import Params, State, Stats
from liblaf.peach.tree import TreeView

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class OptaxState(State):
    wrapped: optax.OptState = tree.field(default=None, kw_only=True)

    value: Scalar = tree.array(default=None, kw_only=True)

    grad = TreeView[Params]()
    grad_flat: Vector = tree.array(default=None, kw_only=True)

    updates = TreeView[Params]()
    updates_flat: Vector = tree.array(default=None, kw_only=True)

    best_params = TreeView[Params]()
    best_params_flat: Vector = tree.array(default=None, kw_only=True)
    best_value_so_far: Scalar = tree.array(default=jnp.inf, kw_only=True)
    steps_from_best: Integer[Array, ""] = tree.array(default=0, kw_only=True)


@tree.define
class OptaxStats(Stats):
    pass
