from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, override

import jarp
import jax.numpy as jnp
import scipy
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach.optim.base import (
    Callback,
    Objective,
    Optimizer,
    Result,
    Solution,
    SupportsFun,
    SupportsGrad,
    SupportsHessProd,
    SupportsValueAndGrad,
)

from ._types import ScipyState, ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class ScipyOptimizer(Optimizer):
    from ._types import ScipyState as State
    from ._types import ScipyStats as Stats

    type Callback[X] = Optimizer.Callback[X, ScipyOptimizer.State, ScipyOptimizer.Stats]
    type Solution = Optimizer.Solution[State, Stats]

    method: str | None = jarp.static(default=None)
    options: Mapping[str, Any] | None = jarp.field(default=None)
    tol: float | None = jarp.static(default=None)

    @override
    def init[X](
        self, objective: Objective[X], model_state: X, params: Vector
    ) -> tuple[State, Stats]:
        res: OptimizeResult = OptimizeResult({"x": params})  # ty:ignore[too-many-positional-arguments]
        return ScipyState(res), ScipyStats()

    @override
    def postprocess[X](
        self,
        objective: Objective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        result: Result = (
            Result.SUCCESS if opt_state["success"] else Result.UNKNOWN_ERROR
        )
        opt_stats._end_time = time.perf_counter()  # noqa: SLF001
        return Solution(result=result, state=opt_state, stats=opt_stats)

    @override
    def minimize[X](
        self,
        objective: Objective[X],
        model_state: X,
        params: Vector,
        callback: Callback | None = None,
    ) -> tuple[Solution, X]:
        opt_state: ScipyState
        opt_stats: ScipyStats
        opt_state, opt_stats = self.init(objective, model_state, params)
        objective_wrapper: _ObjectiveWrapper[X] = _ObjectiveWrapper(
            objective, model_state=model_state
        )
        fun: Callable | None
        jac: Callable | Literal[True] | None
        fun, jac = (
            (objective_wrapper.fun, objective_wrapper.grad)
            if objective_wrapper.value_and_grad is None
            else (objective_wrapper.value_and_grad, True)
        )
        res: OptimizeResult = scipy.optimize.minimize(
            fun=fun,
            x0=opt_state.params,
            method=self.method,
            jac=jac,
            hessp=objective_wrapper.hessp,
            tol=self.tol,
            options=self.options,
        )  # ty:ignore[no-matching-overload]
        opt_state = ScipyState(res)
        solution: ScipyOptimizer.Solution = self.postprocess(
            objective, model_state, opt_state, opt_stats
        )
        return solution, objective_wrapper.model_state

    def _wraps_callback[X](
        self,
        objective_wrapper: _ObjectiveWrapper[X],
        callback: ScipyOptimizer.Callback[X] | None,
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
            callback(
                objective_wrapper.__wrapped__,
                objective_wrapper.model_state,
                state,
                stats,
            )

        return wrapped_callback


@jarp.define
class _ObjectiveWrapper[X]:
    __wrapped__: Objective[X] = jarp.field(alias="wrapped")
    model_state: X

    @property
    def fun(self) -> Callable | None:
        if getattr(self.__wrapped__, "fun", None) is None:
            return None

        def fun(params: Vector) -> Scalar:
            params = jnp.asarray(params, float)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            wrapped: SupportsFun[X] = cast("SupportsFun[X]", self.__wrapped__)
            return wrapped.fun(self.model_state)

        return fun

    @property
    def grad(self) -> Callable | None:
        if getattr(self.__wrapped__, "grad", None) is None:
            return None

        def grad(params: Vector) -> Vector:
            self.model_state = self.__wrapped__.update(self.model_state, params)
            wrapped: SupportsGrad[X] = cast("SupportsGrad[X]", self.__wrapped__)
            return wrapped.grad(self.model_state)

        return grad

    @property
    def hessp(self) -> Callable | None:
        if getattr(self.__wrapped__, "hess_prod", None) is None:
            return None

        def hessp(params: Vector, vector: Vector) -> Vector:
            self.model_state = self.__wrapped__.update(self.model_state, params)
            wrapped: SupportsHessProd[X] = cast("SupportsHessProd[X]", self.__wrapped__)
            return wrapped.hess_prod(self.model_state, vector)

        return hessp

    @property
    def value_and_grad(self) -> Callable | None:
        if getattr(self.__wrapped__, "value_and_grad", None) is None:
            return None

        def value_and_grad(params: Vector) -> tuple[Scalar, Vector]:
            self.model_state = self.__wrapped__.update(self.model_state, params)
            wrapped: SupportsValueAndGrad[X] = cast(
                "SupportsValueAndGrad[X]", self.__wrapped__
            )
            return wrapped.value_and_grad(self.model_state)

        return value_and_grad
