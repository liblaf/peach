from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast, override

import attrs
import scipy
import torch
from jaxtyping import Float
from scipy.optimize import OptimizeResult
from torch import Tensor

from liblaf import jarp
from liblaf.peach.optim.base import BaseProblem, Optimizer, Problem, Result, Solution
from liblaf.peach.utils import is_implemented

from ._types import ScipyState, ScipyStats

if TYPE_CHECKING:
    from scipy.optimize._minimize import _CallbackResult


type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@jarp.define(kw_only=True)
class ScipyOptimizer(Optimizer[ScipyState, ScipyStats]):
    """Adapter around `scipy.optimize.minimize`."""

    from ._types import ScipyState as State
    from ._types import ScipyStats as Stats

    type Solution = Optimizer.Solution[State, Stats]

    method: str | None = jarp.static(default=None)
    options: Mapping[str, Any] | None = jarp.field(default=None)
    tol: float | None = jarp.static(default=None)

    @override
    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> State:
        """Initialize SciPy state from the starting parameters."""
        res: OptimizeResult = OptimizeResult({"x": params})  # ty:ignore[too-many-positional-arguments]
        return self.State(res)

    @override
    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State, result: Result
    ) -> Solution:
        """Build the SciPy optimizer solution object."""
        return Solution(result=result, state=opt_state, stats=self.Stats())

    @override
    def minimize[X](
        self, problem: BaseProblem[X], model_state: X, params: Vector
    ) -> tuple[Solution, X]:
        """Run `scipy.optimize.minimize` against a Peach problem."""
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
        if not is_implemented(problem.__wrapped__, Problem.callback):
            return None

        def callback(intermediate_result: OptimizeResult) -> None:
            state.__wrapped__ = intermediate_result
            if is_implemented(problem.__wrapped__, Problem.callback):
                problem.__wrapped__.callback(problem.model_state, state)

        return callback


@attrs.define
class _ProblemWrapper[X]:
    """Mutable adapter that exposes Peach problem hooks to SciPy."""

    __wrapped__: Problem[X]
    model_state: X

    @property
    def fun(self) -> Callable | None:
        """SciPy-compatible objective callable, when implemented."""
        if not is_implemented(self.__wrapped__, Problem.fun):
            return None

        def fun(params: Vector) -> Scalar:
            params: Tensor = torch.as_tensor(params)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            return self.__wrapped__.fun(self.model_state)

        return fun

    @property
    def grad(self) -> Callable | None:
        """SciPy-compatible gradient callable, when implemented."""
        if not is_implemented(self.__wrapped__, Problem.grad):
            return None

        def grad(params: Vector) -> Vector:
            params: Tensor = torch.as_tensor(params)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            return self.__wrapped__.grad(self.model_state)

        return grad

    @property
    def hessp(self) -> Callable | None:
        """SciPy-compatible Hessian-product callable, when implemented."""
        if not is_implemented(self.__wrapped__, Problem.hess_prod):
            return None

        def hessp(params: Vector, vector: Vector) -> Vector:
            params: Tensor = torch.as_tensor(params)
            vector: Tensor = torch.as_tensor(vector)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            return self.__wrapped__.hess_prod(self.model_state, vector)

        return hessp

    @property
    def value_and_grad(self) -> Callable | None:
        """SciPy-compatible combined value-and-gradient callable, when implemented."""
        if not is_implemented(self.__wrapped__, Problem.value_and_grad):
            return None

        def value_and_grad(params: Vector) -> tuple[Scalar, Vector]:
            params: Tensor = torch.as_tensor(params)
            self.model_state = self.__wrapped__.update(self.model_state, params)
            return self.__wrapped__.value_and_grad(self.model_state)

        return value_and_grad
