from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Never, override

import numpy as np
import scipy
from jaxtyping import Array, ArrayLike, Float
from scipy.optimize import Bounds, OptimizeResult

from liblaf import grapes
from liblaf.peach import tree
from liblaf.peach.constraints import BoundConstraint, Constraints, FixedConstraint
from liblaf.peach.functools import Objective
from liblaf.peach.optim.abc import (
    Callback,
    Optimizer,
    OptimizeSolution,
    Problem,
    Result,
)
from liblaf.peach.transforms import LinearTransform

from ._state import ScipyState
from ._stats import ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult, _MinimizeOptions

type Vector = Float[Array, " N"]
type ScipySolution = OptimizeSolution[ScipyState, ScipyStats]


@tree.define
class ScipyOptimizer(Optimizer[ScipyState, ScipyStats]):
    from ._state import ScipyState as State
    from ._stats import ScipyStats as Stats

    Solution = ScipySolution

    method: str | None = tree.field(default=None, kw_only=True)
    tol: float | None = tree.field(default=None, kw_only=True)
    options: _MinimizeOptions | None = tree.field(default=None, kw_only=True)

    @override
    def init[T](
        self,
        objective: Objective,
        params: T,
        *,
        constraints: Constraints | None,
        transform: LinearTransform[Vector, T] | None,
    ) -> tuple[Problem, State, Stats]:
        self._warn_unsupported_constraints(constraints)
        if constraints is None:
            constraints = Constraints()
        params_flat: Vector
        params_flat, transform = self._transform_params(params, transform)
        problem = Problem(
            objective=objective, constraints=constraints, transform=transform
        )
        state: ScipyState = self.State(OptimizeResult({"x": params_flat}))  # pyright: ignore[reportCallIssue]
        stats: ScipyStats = self.Stats()
        return problem, state, stats

    @override
    def step(self, problem: Problem, state: State) -> Never:
        raise NotImplementedError

    @override
    def terminate(self, problem: Problem, state: State, stats: Stats) -> Never:
        raise NotImplementedError

    @override
    def minimize[T](
        self,
        objective: Objective,
        params: T,
        *,
        callback: Callback[State, Stats] | None = None,
        constraints: Constraints | None = None,
        transform: LinearTransform[Vector, T] | None = None,
    ) -> Solution:
        problem: Problem
        state: ScipyState
        stats: ScipyStats
        problem, state, stats = self.init(
            objective, params, constraints=constraints, transform=transform
        )

        # bound_constr: BoundConstraint | None
        # fixed_constr: FixedConstraint | None
        # other_constr: list[Constraint]
        # bound_constr, other_constr = pop_constraint(constraints, BoundConstraint)
        # fixed_constr, other_constr = pop_constraint(other_constr, FixedConstraint)
        # self._warn_unsupported_constraints(other_constr)
        # bounds: Bounds | None = self._make_bounds(bound_constr, fixed_constr)
        # if fixed_constr is not None:
        #     objective: ObjectiveProtocol = fixed_constr.wraps_objective(objective)
        cb_wrapper: _CallbackResult = self._make_callback(problem, callback, stats)
        options: _MinimizeOptions = self._make_options()

        objective = objective.apply_transform(transform)
        fun: Callable | None = objective.fun
        grad: Callable | bool | None = objective.grad
        if objective.value_and_grad is not None:
            fun = objective.value_and_grad
            grad = True

        result_wrapped: OptimizeResult = scipy.optimize.minimize(  # pyright: ignore[reportCallIssue]
            # bounds=bounds,
            callback=cb_wrapper,
            fun=fun,  # pyright: ignore[reportArgumentType]
            hessp=objective.hess_prod,
            jac=grad,  # pyright: ignore[reportArgumentType]
            method=self.method,  # pyright: ignore[reportArgumentType]
            options=options,
            tol=self.tol,
            x0=state.result["x"],
        )
        state: ScipyState = self.State(result_wrapped)  # pyright: ignore[reportCallIssue]
        result: Result = Result.SUCCESS if state["success"] else Result.UNKNOWN_ERROR
        solution: ScipySolution = self.postprocess(problem, state, stats, result)
        return solution

    def _make_bounds(
        self, bound: BoundConstraint | None, fixed: FixedConstraint | None
    ) -> Bounds | None:
        if bound is None:
            return None
        lower: ArrayLike = -np.inf
        upper: ArrayLike = np.inf
        if fixed is not None:
            if bound.lower_flat is not None:
                lower = fixed.get_free(bound.lower_flat)
            if bound.upper_flat is not None:
                upper = fixed.get_free(bound.upper_flat)
        else:
            if bound.lower_flat is not None:
                lower = bound.lower_flat
            if bound.upper_flat is not None:
                upper = bound.upper_flat
        return Bounds(lower, upper)

    def _make_callback(
        self, problem: Problem, callback: Callback[State, Stats] | None, stats: Stats
    ) -> _CallbackResult:
        @grapes.timer(label=f"{self.name}.callback()")
        def wrapper(intermediate_result: OptimizeResult) -> None:
            nonlocal stats
            if callback is not None:
                state: ScipyState = self.State(intermediate_result)
                stats.n_steps = len(grapes.get_timer(wrapper)) + 1
                stats = self.update_stats(problem, state, stats)
                callback(problem, state, stats)

        return wrapper

    def _make_options(self) -> _MinimizeOptions:
        options: _MinimizeOptions = {}
        if self.max_steps is not None:
            options["maxiter"] = self.max_steps
        if self.options is not None:
            options.update(self.options)
        return options
