from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast, override

import jax.numpy as jnp
import scipy
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf import jarp
from liblaf.peach.optim.base import BaseProblem, Optimizer, Problem, Result, Solution
from liblaf.peach.utils import implemented

from ._types import ScipyState, ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class ScipyOptimizer(Optimizer[ScipyState, ScipyStats]):
    from ._types import ScipyState as State
    from ._types import ScipyStats as Stats

    type Solution = Optimizer.Solution[State, Stats]

    method: str | None = jarp.static(default=None)
    options: Mapping[str, Any] | None = jarp.field(default=None)
    tol: float | None = jarp.static(default=None)

    @override
    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> State:
        res: OptimizeResult = OptimizeResult({"x": params})  # ty:ignore[too-many-positional-arguments]
        return self.State(res)

    @override
    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State, result: Result
    ) -> Solution:
        return Solution(result=result, state=opt_state, stats=self.Stats())

    @override
    def minimize[X](
        self, problem: BaseProblem[X], model_state: X, params: Vector
    ) -> tuple[Solution, X]:
        problem: Problem[X] = cast("Problem[X]", problem)
        opt_state: ScipyState = self.init(problem, model_state, params)
        wrapper: _ProblemWrapper[X] = _ProblemWrapper(problem, model_state=model_state)
        fun, jac = (
            (wrapper.fun, wrapper.grad)
            if wrapper.value_and_grad is None
            else (wrapper.value_and_grad, True)
        )
        scipy_result: OptimizeResult = scipy.optimize.minimize(
            fun=fun,
            x0=opt_state.params,
            method=self.method,
            jac=jac,
            hessp=wrapper.hessp,
            tol=self.tol,
            callback=self._wraps_callback(wrapper, opt_state),
            options=self.options,
        )  # ty:ignore[no-matching-overload]
        opt_state: ScipyState = self.State(scipy_result)
        result: Result = (
            Result.SUCCESS if scipy_result.success else Result.UNKNOWN_ERROR
        )
        solution: ScipyOptimizer.Solution = self.postprocess(
            problem, model_state, opt_state, result
        )
        return solution, wrapper.model_state

    def _wraps_callback[X](
        self, problem: _ProblemWrapper[X], state: ScipyState
    ) -> _CallbackResult | None:
        if not implemented(problem.__wrapped__, Problem.callback):
            return None

        def callback(intermediate_result: OptimizeResult) -> None:
            state.__wrapped__ = intermediate_result
            if implemented(problem.__wrapped__, Problem.callback):
                problem.__wrapped__.callback(problem.model_state, state)

        return callback


@jarp.define
class _ProblemWrapper[X]:
    __wrapped__: Problem[X] = jarp.field(alias="__wrapped__")
    model_state: X

    @property
    def fun(self) -> Callable | None:
        if not implemented(self.__wrapped__, Problem.fun):
            return None

        def fun(params: Vector) -> Scalar:
            params: Array = jnp.asarray(params, float)
            self.model_state = self.__wrapped__.before_step(self.model_state, params)
            return self.__wrapped__.fun(self.model_state)

        return fun

    @property
    def grad(self) -> Callable | None:
        if not implemented(self.__wrapped__, Problem.grad):
            return None

        def grad(params: Vector) -> Vector:
            self.model_state = self.__wrapped__.before_step(self.model_state, params)
            return self.__wrapped__.grad(self.model_state)

        return grad

    @property
    def hessp(self) -> Callable | None:
        if not implemented(self.__wrapped__, Problem.hess_prod):
            return None

        def hessp(params: Vector, vector: Vector) -> Vector:
            self.model_state = self.__wrapped__.before_step(self.model_state, params)
            return self.__wrapped__.hess_prod(self.model_state, vector)

        return hessp

    @property
    def value_and_grad(self) -> Callable | None:
        if not implemented(self.__wrapped__, Problem.value_and_grad):
            return None

        def value_and_grad(params: Vector) -> tuple[Scalar, Vector]:
            self.model_state = self.__wrapped__.before_step(self.model_state, params)
            return self.__wrapped__.value_and_grad(self.model_state)

        return value_and_grad
