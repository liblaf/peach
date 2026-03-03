import time
from typing import override

import jarp
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Integer

from liblaf.peach.optim.base import Optimizer, Result

from ._types import OptaxObjective, OptaxState, OptaxStats

type BooleanNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class Optax(Optimizer[OptaxObjective, OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    type Callback[X] = Optimizer.Callback[X, Optax.State, Optax.Stats]
    type Solution = Optimizer.Solution[State, Stats]

    __wrapped__: optax.GradientTransformation = jarp.field(alias="wrapped")

    # termination criteria
    max_steps: Integer[Array, ""] = jarp.array(default=1000, kw_only=True)
    patience: Integer[Array, ""] = jarp.array(default=20, kw_only=True)
    rtol: Scalar = jarp.array(default=0.0, kw_only=True)

    # miscellaneous
    jit: bool = jarp.static(default=False, kw_only=True)

    @override
    def init[X](
        self, objective: OptaxObjective[X], model_state: X, params: Vector
    ) -> tuple[State, Stats]:
        wrapped: optax.OptState = self.__wrapped__.init(params)
        state: OptaxState = OptaxState(wrapped, params=params)
        stats: OptaxStats = OptaxStats()
        return state, stats

    @override
    def step[X](
        self, objective: OptaxObjective[X], model_state: X, opt_state: State
    ) -> tuple[X, State]:
        assert objective.value_and_grad is not None
        model_state = objective.update(model_state, opt_state.params)
        opt_state.value, opt_state.grad = objective.value_and_grad(model_state)
        opt_state.updates, opt_state.__wrapped__ = self.__wrapped__.update(  # pyright: ignore[reportAttributeAccessIssue]
            opt_state.grad, opt_state.__wrapped__, opt_state.params
        )
        opt_state.params = optax.apply_updates(opt_state.params, opt_state.updates)  # pyright: ignore[reportAttributeAccessIssue]
        opt_state.n_steps += 1
        return model_state, opt_state

    @override
    def terminate[X](
        self,
        objective: OptaxObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> BooleanNumeric:
        if opt_state.value <= opt_state.best_value:
            opt_state.best_params = opt_state.params
            opt_state.best_value = opt_state.value
            opt_state.steps_from_best = jnp.zeros_like(opt_state.steps_from_best)
        else:
            opt_state.steps_from_best += 1
        if (
            opt_state.steps_from_best > self.patience
            and opt_state.value >= opt_state.best_value * (1.0 - self.rtol)
        ):
            return jnp.ones((), bool)
        if opt_state.n_steps > self.max_steps:
            return jnp.ones((), bool)
        return jnp.zeros((), bool)

    @override
    def postprocess[X](
        self,
        objective: OptaxObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        result: Result = (
            Result.SUCCESS
            if opt_state.n_steps <= self.max_steps
            else Result.UNKNOWN_ERROR
        )
        opt_stats._end_time = time.perf_counter()  # noqa: SLF001
        opt_state.params = opt_state.best_params
        return Optimizer.Solution(result=result, state=opt_state, stats=opt_stats)
