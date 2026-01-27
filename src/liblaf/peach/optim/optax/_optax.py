import time
from typing import override

import jarp
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Integer

from liblaf.peach.optim.base import Objective, Optimizer, Result
from liblaf.peach.transforms import Transform

from ._types import OptaxState, OptaxStats

type BooleanNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class Optax(Optimizer[OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    type Callback[ModelState, Params] = Optimizer.Callback[
        ModelState, Params, Optax.State, Optax.Stats
    ]
    type Solution = Optimizer.Solution[State, Stats]

    __wrapped__: optax.GradientTransformation = jarp.field(alias="wrapped")

    # termination criteria
    max_steps: Integer[Array, ""] = jarp.array(default=1000, kw_only=True)
    patience: Integer[Array, ""] = jarp.array(default=20, kw_only=True)
    rtol: Scalar = jarp.array(default=0.0, kw_only=True)

    # miscellaneous
    jit: bool = jarp.static(default=False, kw_only=True)

    @override
    def init[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
    ) -> tuple[State, Stats]:
        params_flat: Vector = objective.transform.backward_primals(params)
        wrapped: optax.OptState = self.__wrapped__.init(params_flat)
        state: OptaxState = OptaxState(wrapped, params=params_flat)
        stats: OptaxStats = OptaxStats()
        return state, stats

    @override
    def step[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
    ) -> tuple[ModelState, State]:
        assert objective.value_and_grad is not None
        transform: Transform = objective.transform
        params_tree: Params = transform.forward_primals(opt_state.params)
        model_state = objective.update(model_state, params_tree)
        grad_tree: Params
        opt_state.value, grad_tree = objective.value_and_grad(model_state)
        opt_state.grad = transform.backward_primals(grad_tree)
        opt_state.updates, opt_state.__wrapped__ = self.__wrapped__.update(  # pyright: ignore[reportAttributeAccessIssue]
            opt_state.grad, opt_state.__wrapped__, opt_state.params
        )
        opt_state.params = optax.apply_updates(opt_state.params, opt_state.updates)  # pyright: ignore[reportAttributeAccessIssue]
        opt_state.n_steps += 1
        return model_state, opt_state

    @override
    def terminate[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
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
    def postprocess[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
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
