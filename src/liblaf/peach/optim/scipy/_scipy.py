from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Never, override

import scipy
from jaxtyping import Array
from scipy.optimize import Bounds, OptimizeResult

from liblaf import grapes
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
from liblaf.peach.tree import Unflatten

from ._state import ScipyState
from ._stats import ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


@tree.define
class ScipyOptimizer(Optimizer[ScipyState, ScipyStats]):
    from ._state import ScipyState as State
    from ._stats import ScipyStats as Stats

    method: str | None = tree.field(default=None, kw_only=True)
    tol: float | None = tree.field(default=None, kw_only=True)
    options: Mapping[str, Any] | None = tree.field(default=None, kw_only=True)

    @override
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[ScipyState, ScipyStats]:
        params_flat: Array
        objective, params_flat, constraints = objective.flatten(
            params, constraints=constraints
        )
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        assert objective.unflatten is not None
        state = ScipyState(
            unflatten=objective.unflatten, result=OptimizeResult({"x": params_flat})
        )
        stats = ScipyStats()
        return SetupResult(objective, constraints, state, stats)

    @override
    def step(
        self,
        objective: Objective,
        state: ScipyState,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> Never:
        raise NotImplementedError

    @override
    def terminate(
        self,
        objective: Objective,
        state: ScipyState,
        stats: ScipyStats,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> Never:
        raise NotImplementedError

    @override
    def minimize(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback[ScipyState, ScipyStats] | None = None,
    ) -> OptimizeSolution[ScipyState, ScipyStats]:
        options: dict[str, Any] = {"maxiter": self.max_steps}
        if self.options is not None:
            options.update(self.options)
        state: ScipyState
        stats: ScipyStats
        objective, constraints, state, stats = self.init(
            objective, params, constraints=constraints
        )
        callback_wrapper: _CallbackResult = self._make_callback(
            objective, callback, stats, state.unflatten
        )
        fun: Callable | None
        jac: Callable | bool | None
        if objective.value_and_grad is None:
            fun = objective.fun
            jac = objective.grad
        else:
            fun = objective.value_and_grad
            jac = True
        raw: OptimizeResult = scipy.optimize.minimize(  # pyright: ignore[reportCallIssue]
            bounds=self._make_bounds(objective, constraints),
            callback=callback_wrapper,
            fun=fun,  # pyright: ignore[reportArgumentType]
            hess=objective.hess,
            hessp=objective.hess_prod,
            jac=jac,  # pyright: ignore[reportArgumentType]
            method=self.method,  # pyright: ignore[reportArgumentType]
            options=options,  # pyright: ignore[reportArgumentType]
            tol=self.tol,
            x0=state.result["x"],
        )
        state: ScipyState = self._unflatten_state(raw, state.unflatten)
        result: Result = (
            Result.SUCCESS if state.result["success"] else Result.UNKNOWN_ERROR
        )
        solution: OptimizeSolution[ScipyState, ScipyStats] = self.postprocess(
            objective, state, stats, result
        )
        return solution

    def _make_bounds(
        self,
        objective: Objective,  # noqa: ARG002
        constraints: list[Constraint],
    ) -> Bounds | None:
        # TODO: implement bound constraints
        if constraints:
            raise NotImplementedError

    def _make_callback(
        self,
        objective: Objective,
        callback: Callback[ScipyState, ScipyStats] | None,
        stats: ScipyStats,
        unflatten: Unflatten[Params],
    ) -> _CallbackResult:
        @grapes.timer(label="callback()")
        def wrapper(intermediate_result: OptimizeResult) -> None:
            nonlocal stats
            if callback is not None:
                state: ScipyState = self._unflatten_state(
                    intermediate_result, unflatten
                )
                stats.n_steps = len(grapes.get_timer(wrapper)) + 1
                stats = self.update_stats(objective, state, stats)
                callback(state, stats)

        return wrapper

    def _unflatten_state(
        self, result: OptimizeResult, unflatten: Unflatten[Params]
    ) -> ScipyState:
        state = ScipyState(result=result, unflatten=unflatten)
        return state
