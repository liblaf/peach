from collections.abc import Iterable
from typing import override

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.functools import Objective, ObjectiveProtocol
from liblaf.peach.optim.abc import Callback, Optimizer, OptimizeSolution, Result

from ._types import OptaxState, OptaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]
type OptaxSolution = OptimizeSolution[OptaxState, OptaxStats]


@tree.define
class Optax(Optimizer[OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    Solution = OptaxSolution
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
        params: Vector,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[State, Stats]:
        state: OptaxState = self.State(params=params, wrapped=self.wrapped.init(params))
        stats: OptaxStats = self.Stats()
        return state, stats

    @override
    def step(
        self,
        objective: Objective,
        state: State,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> State:
        self._warn_unsupported_constraints(constraints)
        assert objective.value_and_grad is not None
        state.value, state.grad = objective.value_and_grad(state.params)
        state.updates, state.wrapped = self.wrapped.update(  # pyright: ignore[reportAttributeAccessIssue]
            state.grad, state.wrapped, state.params
        )
        state.params = optax.apply_updates(state.params, state.updates)  # pyright: ignore[reportAttributeAccessIssue]
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
            state.best_params = state.params
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
        objective: ObjectiveProtocol,
        state: State,
        stats: Stats,
        result: Result,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> Solution:
        state.params = state.best_params
        return super().postprocess(
            objective, state, stats, result, constraints=constraints
        )
