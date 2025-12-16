import warnings
from collections.abc import Iterable
from typing import override

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim.abc import (
    Callback,
    Optimizer,
    OptimizeSolution,
    Params,
    Result,
    SetupResult,
)
from liblaf.peach.optim.objective import Objective

from ._types import OptaxState, OptaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class Optax(Optimizer[OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    Solution = OptimizeSolution[State, Stats]
    Callback = Callback[State, Stats]

    wrapped: optax.GradientTransformation

    patience: Integer[Array, ""] = tree.array(
        default=20, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )

    @override
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[State, Stats]:
        state: OptaxState
        stats: OptaxStats
        objective, constraints, state, stats = super().init(
            objective, params, constraints=constraints
        )
        state.wrapped = self.wrapped.init(state.params_flat)
        return SetupResult(objective, constraints, state, stats)

    @override
    def step(
        self,
        objective: Objective,
        state: State,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> State:
        if constraints:
            warnings.warn(
                "Optax optimizer does not support constraints. Ignoring them.",
                RuntimeWarning,
                stacklevel=3,
            )
        assert objective.value_and_grad is not None
        state.value, state.grad_flat = objective.value_and_grad(state.params_flat)
        state.updates_flat, state.wrapped = self.wrapped.update(  # pyright: ignore[reportAttributeAccessIssue]
            state.grad_flat, state.wrapped, state.params_flat
        )
        state.params_flat = optax.apply_updates(state.params_flat, state.updates_flat)  # pyright: ignore[reportAttributeAccessIssue]
        return state

    @override
    def terminate(
        self,
        objective: Objective,
        state: State,
        stats: Stats,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[bool, Result]:
        if state.value <= state.best_value_so_far:
            state.best_params_flat = state.params_flat
            state.best_value_so_far = state.value
            state.steps_from_best = jnp.zeros_like(state.steps_from_best)
        else:
            state.steps_from_best += 1
        if (
            state.steps_from_best > self.patience
            and state.value >= state.best_value_so_far * (1.0 - self.rtol)
        ):
            return True, Result.SUCCESS
        return False, Result.UNKNOWN_ERROR

    @override
    def postprocess(
        self,
        objective: Objective,
        state: OptaxState,
        stats: OptaxStats,
        result: Result,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> OptimizeSolution[OptaxState, OptaxStats]:
        state.params_flat = state.best_params_flat
        return super().postprocess(
            objective, state, stats, result, constraints=constraints
        )
