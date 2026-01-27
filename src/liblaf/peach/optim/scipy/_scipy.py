from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, override

import jarp
import jax.numpy as jnp
import scipy
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach.optim.base import Callback, Objective, Optimizer, Result, Solution
from liblaf.peach.transforms import Transform

from ._types import ScipyState, ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class ScipyOptimizer(Optimizer):
    from ._types import ScipyState as State
    from ._types import ScipyStats as Stats

    type Callback = Optimizer.Callback[State, Stats]
    type Solution = Optimizer.Solution[State, Stats]

    method: str | None = jarp.static(default=None)
    options: Mapping[str, Any] | None = jarp.field(default=None)
    tol: float | None = jarp.static(default=None)

    @override
    def init[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
    ) -> tuple[State, Stats]:
        params_flat: Vector = objective.transform.backward_primals(params)
        res: OptimizeResult = OptimizeResult({"x": params_flat})  # pyright: ignore[reportCallIssue]
        return ScipyState(res), ScipyStats()

    @override
    def postprocess[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        result: Result = (
            Result.SUCCESS if opt_state["success"] else Result.UNKNOWN_ERROR
        )
        return Solution(result=result, state=opt_state, stats=opt_stats)

    @override
    def minimize[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
        callback: Callback | None = None,
    ) -> tuple[Solution, ModelState]:
        opt_state: ScipyState
        opt_stats: ScipyStats
        opt_state, opt_stats = self.init(objective, model_state, params)
        objective_wrapper: _ObjectiveWrapper[ModelState, Params] = _ObjectiveWrapper(
            objective, model_state=model_state
        )
        fun: Callable | None
        jac: Callable | Literal[True] | None
        fun, jac = (
            (objective_wrapper.fun, objective_wrapper.grad)
            if objective_wrapper.value_and_grad is None
            else (objective_wrapper.value_and_grad, True)
        )
        res: OptimizeResult = scipy.optimize.minimize(  # pyright: ignore[reportCallIssue]
            fun=fun,  # pyright: ignore[reportArgumentType]
            x0=opt_state.params,  # pyright: ignore[reportArgumentType]
            method=self.method,  # pyright: ignore[reportArgumentType]
            jac=jac,  # pyright: ignore[reportArgumentType]
            hessp=objective_wrapper.hessp,
            tol=self.tol,
            options=self.options,  # pyright: ignore[reportArgumentType]
        )
        opt_state = ScipyState(res)
        solution: ScipyOptimizer.Solution = self.postprocess(
            objective, model_state, opt_state, opt_stats
        )
        return solution, objective_wrapper.model_state

    def _wraps_callback(
        self,
        objective_wrapper: _ObjectiveWrapper,
        callback: ScipyOptimizer.Callback | None,
        state: ScipyState,
        stats: ScipyStats,
    ) -> _CallbackResult | None:
        if callback is None:
            return None

        def wrapped_callback(intermediate_result: OptimizeResult) -> None:
            nonlocal state, stats
            state.__wrapped__ = intermediate_result
            stats = self.update_stats(
                objective_wrapper.__wrapped__,
                objective_wrapper.model_state,
                state,
                stats,
            )
            callback(objective_wrapper.__wrapped__, state, stats)

        return wrapped_callback


@jarp.define
class _ObjectiveWrapper[ModelState, Params]:
    __wrapped__: Objective[ModelState, Params] = jarp.field(alias="wrapped")
    model_state: ModelState

    @property
    def transform(self) -> Transform[Vector, Params]:
        return self.__wrapped__.transform

    @property
    def fun(self) -> Callable | None:
        if self.__wrapped__.fun is None:
            return None

        def fun(params_flat: Vector) -> Scalar:
            assert self.__wrapped__.fun is not None
            params_flat = jnp.asarray(params_flat, float)
            params: Params = self.transform.forward_primals(params_flat)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            return self.__wrapped__.fun(self.model_state)

        return fun

    @property
    def grad(self) -> Callable | None:
        if self.__wrapped__.grad is None:
            return None

        def grad(params_flat: Vector) -> Vector:
            assert self.__wrapped__.grad is not None
            params_flat = jnp.asarray(params_flat, float)
            params: Params = self.transform.forward_primals(params_flat)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            grad_params: Params = self.__wrapped__.grad(self.model_state)
            grad_flat: Vector = self.transform.backward_tangents(grad_params)
            return grad_flat

        return grad

    @property
    def hessp(self) -> Callable | None:
        if self.__wrapped__.hess_prod is None:
            return None

        def hessp(params_flat: Vector, vector_flat: Vector) -> Vector:
            assert self.__wrapped__.hess_prod is not None
            params_flat = jnp.asarray(params_flat, float)
            vector_flat = jnp.asarray(vector_flat, float)
            params: Params = self.transform.forward_primals(params_flat)
            vector: Params = self.transform.forward_primals(vector_flat)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            hess_prod_params: Params = self.__wrapped__.hess_prod(
                self.model_state, vector
            )
            hess_prod_flat: Vector = self.transform.backward_tangents(hess_prod_params)
            return hess_prod_flat

        return hessp

    @property
    def value_and_grad(self) -> Callable | None:
        if self.__wrapped__.value_and_grad is None:
            return None

        def value_and_grad(params_flat: Vector) -> tuple[Scalar, Vector]:
            assert self.__wrapped__.value_and_grad is not None
            params_flat = jnp.asarray(params_flat, float)
            params: Params = self.transform.forward_primals(params_flat)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            value: Scalar
            grad_tree: Params
            value, grad_tree = self.__wrapped__.value_and_grad(self.model_state)
            grad_flat: Vector = self.transform.backward_tangents(grad_tree)
            return value, grad_flat

        return value_and_grad
