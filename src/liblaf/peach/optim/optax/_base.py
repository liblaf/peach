from typing import override

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Integer, PyTree

from liblaf.peach import tree
from liblaf.peach.constraints import Constraints
from liblaf.peach.functools import Objective
from liblaf.peach.optim.abc import (
    Callback,
    Optimizer,
    OptimizeSolution,
    Problem,
    Result,
)
from liblaf.peach.transforms import LinearTransform

from ._types import OptaxState, OptaxStats

type OptaxSolution = OptimizeSolution[OptaxState, OptaxStats]
type Params = PyTree
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class Optax(Optimizer[OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    Callback = Callback[State, Stats]
    Solution = OptaxSolution

    wrapped: optax.GradientTransformation
    patience: Integer[Array, ""] = tree.array(
        default=400, converter=tree.converters.asarray, kw_only=True
    )
    atol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )

    @override
    def init[T](
        self,
        objective: Objective,
        params: T,
        *,
        constraints: Constraints | None = None,
        transform: LinearTransform[Vector, T] | None = None,
    ) -> tuple[Problem, State, Stats]:
        self._warn_unsupported_constraints(constraints)
        if constraints is None:
            constraints = Constraints()
        params_flat: Vector
        params_flat, transform = self._transform_params(params, transform)
        problem = Problem(
            objective=objective, constraints=constraints, transform=transform
        )
        state: OptaxState = self.State(
            self.wrapped.init(params_flat), params=params_flat
        )
        stats: OptaxStats = self.Stats()
        return problem, state, stats

    @override
    def step(self, problem: Problem, state: State) -> State:
        objective: Objective = problem.objective
        transform: LinearTransform[Vector, Params] = problem.transform
        assert objective.value_and_grad is not None
        params_tree: Params = transform.forward_primals(state.params)
        grad_tree: Params
        state.value, grad_tree = objective.value_and_grad(params_tree)
        state.grad = transform.linear_transpose(grad_tree)
        state.updates, state.wrapped = self.wrapped.update(  # pyright: ignore[reportAttributeAccessIssue]
            state.grad, state.wrapped, state.params
        )
        state.params = optax.apply_updates(state.params, state.updates)  # pyright: ignore[reportAttributeAccessIssue]
        return state

    @override
    def terminate(
        self, problem: Problem, state: State, stats: Stats
    ) -> tuple[bool, Result]:
        if state.value <= state.best_value:
            state.best_params = state.params
            state.best_value = state.value
            state.no_decrease_steps = jnp.zeros_like(state.no_decrease_steps)
        else:
            state.no_decrease_steps += 1
        if (
            state.no_decrease_steps > self.patience
            and state.value >= state.best_value * (1.0 - self.rtol)
        ):
            return True, Result.SUCCESS
        return False, Result.UNKNOWN_ERROR

    @override
    def postprocess(
        self, problem: Problem, state: State, stats: Stats, result: Result
    ) -> Solution:
        state.params = state.best_params
        return super().postprocess(problem, state, stats, result)
