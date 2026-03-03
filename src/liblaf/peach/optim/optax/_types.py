from typing import Protocol

import jarp
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer

from liblaf.peach.optim.base import Objective, State, Stats, SupportsValueAndGrad

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class OptaxObjective[X](Objective[X], SupportsValueAndGrad[X], Protocol): ...


@jarp.define
class OptaxState(State):
    __wrapped__: optax.OptState = jarp.field(default=None, alias="wrapped")

    n_steps: Integer[Array, ""] = jarp.array(default=0, kw_only=True)

    value: Scalar = jarp.array(default=jnp.inf, kw_only=True)
    grad: Vector = jarp.array(default=None, kw_only=True)
    updates: Vector = jarp.array(default=None, kw_only=True)

    # best so far
    best_params: Vector = jarp.array(default=None, kw_only=True)
    best_value: Scalar = jarp.array(default=jnp.inf, kw_only=True)
    steps_from_best: Integer[Array, ""] = jarp.array(default=0, kw_only=True)


@jarp.define
class OptaxStats(Stats): ...
