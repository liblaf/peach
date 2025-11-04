import copy
from collections.abc import Callable, Mapping
from typing import Any, Never, override

import scipy
from jaxtyping import Array
from scipy.optimize import OptimizeResult

from liblaf import grapes
from liblaf.peach import tree_utils
from liblaf.peach.optim.abc import Aux, Callback, Optimizer, OptimizeSolution, Result, X
from liblaf.peach.optim.objective import Objective

type State = OptimizeResult


@tree_utils.tree
class OptimizerScipy(Optimizer[State]):
    method: str | None = None
    tol: float | None = None
    options: Mapping[str, Any] | None = None

    @override
    def init(self, objective: Objective, x: X) -> Never:
        raise NotImplementedError

    @override
    def step(self, objective: Objective, x: X, state: State) -> Never:
        raise NotImplementedError

    @override
    def terminate(self, objective: Objective, x: X, state: State) -> Never:
        raise NotImplementedError

    @override
    def postprocess(
        self,
        objective: Objective,
        x: X,
        aux: Aux,
        state: State,
        result: Result,
    ) -> OptimizeSolution[State]:
        solution: OptimizeSolution[State] = OptimizeSolution(
            aux=aux, state=state, stats={}, x=x, result=result
        )
        return solution

    @override
    def minimize(
        self,
        objective: Objective,
        x: X,
        callback: Callback[State] | None = None,
    ) -> OptimizeSolution[State]:
        with grapes.timer(name=str(self)) as timer:
            options: dict[str, Any] = {"maxiter": self.max_steps}
            if self.options is not None:
                options.update(self.options)
            x_flat: Array
            unflatten: Callable[[Any], Any]
            x_flat, unflatten = tree_utils.flatten(x)
            objective = objective.flatten(unflatten)
            callback_wrapper: Callable[[State], None] = self._make_callback(
                callback, unflatten
            )
            raw: State = scipy.optimize.minimize(  # pyright: ignore[reportCallIssue]
                bounds=objective.bounds,
                callback=callback_wrapper,  # pyright: ignore[reportArgumentType]
                fun=objective.fun,  # pyright: ignore[reportArgumentType]
                hess=objective.hess,
                hessp=objective.hess_prod,
                jac=objective.grad,
                method=self.method,  # pyright: ignore[reportArgumentType]
                options=options,  # pyright: ignore[reportArgumentType]
                tol=self.tol,
                x0=x_flat,  # pyright: ignore[reportArgumentType]
            )
            state: State = self._unflatten_state(raw, unflatten)
            result: Result = Result.SUCCESSFUL if state["success"] else Result.FAILURE
            x = state["x"]
            solution: OptimizeSolution[State] = self.postprocess(
                objective, x, None, state, result
            )  # pyright: ignore[reportAssignmentType]
            solution.stats["n_steps"] = len(grapes.get_timer(callback_wrapper))
            solution.stats["time"] = timer.elapsed()
        return solution

    def _make_callback(
        self, callback: Callback[State] | None, unflatten: Callable[[Any], Any]
    ) -> Callable[[State], None]:
        @grapes.timer(name="callback()")
        def wrapper(intermediate_result: State) -> None:
            if callback is not None:
                state: State = self._unflatten_state(intermediate_result, unflatten)
                n_steps: int = len(grapes.get_timer(wrapper)) + 1
                callback(state, n_steps)

        return wrapper

    def _unflatten_state(self, state: State, unflatten: Callable[[Any], Any]) -> State:
        state = copy.copy(state)
        if "x" in state:
            state["x"] = unflatten(state["x"])
        return state
