from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Never, override

import scipy
from jaxtyping import Array
from scipy.optimize import OptimizeResult

from liblaf import grapes
from liblaf.peach import tree_utils
from liblaf.peach.optim.abc import Callback, Optimizer, OptimizeSolution, Params, Result
from liblaf.peach.optim.objective import Objective

from ._state import ScipyState
from ._stats import ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


@tree_utils.tree
class ScipyOptimizer(Optimizer[ScipyState, ScipyStats]):
    method: str | None = None
    tol: float | None = None
    options: Mapping[str, Any] | None = None

    @override
    def init(
        self, objective: Objective, params: Params
    ) -> tuple[Objective, ScipyState, ScipyStats]:
        params_flat: Array
        unflatten: Callable[[Array], Params]
        params_flat, unflatten = tree_utils.flatten(params)
        objective = objective.flatten(unflatten)
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        state = ScipyState(
            result=OptimizeResult({"x": params_flat}), unflatten=unflatten
        )
        stats = ScipyStats()
        return objective, state, stats

    @override
    def step(self, objective: Objective, state: ScipyState) -> Never:
        raise NotImplementedError

    @override
    def terminate(
        self, objective: Objective, state: ScipyState, stats: ScipyStats
    ) -> Never:
        raise NotImplementedError

    @override
    def postprocess(
        self, objective: Objective, state: ScipyState, stats: ScipyStats, result: Result
    ) -> OptimizeSolution[ScipyState, ScipyStats]:
        solution: OptimizeSolution[ScipyState, ScipyStats] = OptimizeSolution(
            result=result, state=state, stats=stats
        )
        return solution

    @override
    def minimize(
        self,
        objective: Objective,
        params: Params,
        callback: Callback[ScipyState, ScipyStats] | None = None,
    ) -> OptimizeSolution[ScipyState, ScipyStats]:
        with grapes.timer(label=str(self)) as timer:
            options: dict[str, Any] = {"maxiter": self.max_steps}
            if self.options is not None:
                options.update(self.options)
            state: ScipyState
            stats: ScipyStats
            objective, state, stats = self.init(objective, params)
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
                bounds=objective.bounds,
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
        solution.stats.time = timer.elapsed()
        return solution

    def _make_callback(
        self,
        objective: Objective,
        callback: Callback[ScipyState, ScipyStats] | None,
        stats: ScipyStats,
        unflatten: Callable[[Array], Any],
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
        self, result: OptimizeResult, unflatten: Callable[[Array], Any]
    ) -> ScipyState:
        state = ScipyState(result=result, unflatten=unflatten)
        return state
