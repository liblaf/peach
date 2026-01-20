import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer

from liblaf.peach import tree
from liblaf.peach.optim.abc import State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class OptaxState(State):
    wrapped: optax.OptState = tree.field()
    params: Vector = tree.array(kw_only=True)  # pyright: ignore[reportIncompatibleMethodOverride]

    value: Scalar = tree.array(default=None, kw_only=True)
    grad: Vector = tree.array(default=None, kw_only=True)
    updates: Vector = tree.array(default=None, kw_only=True)

    # early stop
    best_params: Vector = tree.array(default=None, kw_only=True)
    best_value: Scalar = tree.array(default=jnp.inf, kw_only=True)
    no_decrease_steps: Integer[Array, ""] = tree.array(default=0, kw_only=True)


@tree.define
class OptaxStats(Stats):
    pass
