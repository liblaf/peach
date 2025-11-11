import abc
from typing import Any, override

import jax.numpy as jnp
from jaxtyping import Array, Shaped

from liblaf.peach import tree_utils
from liblaf.peach.linalg.abc import Callback, LinearSolution, LinearSolver, Result
from liblaf.peach.linalg.op import LinearOperator
from liblaf.peach.optim.abc import Params

from ._types import JaxState, JaxStats

type Vector = Shaped[Array, " free"]


@tree_utils.define
class JaxSolver(LinearSolver[JaxState, JaxStats]):
    atol: float = 0.0
    max_steps: int | None = None
    rtol: float = 1e-5

    @override
    def setup(
        self,
        op: LinearOperator,
        b: Params,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
    ) -> tuple[LinearOperator, JaxState, JaxStats]:
        flat: Vector
        op, flat = op.flatten(
            params,
            fixed_mask=fixed_mask,
            n_fixed=n_fixed,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        state = JaxState(params_flat=flat, unflatten=op.unflatten)
        state.b = b
        if self.jit:
            op = op.jit()
        if self.timer:
            op = op.timer()
        return op, state, JaxStats()

    def solve(
        self,
        op: LinearOperator,
        b: Params,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
        callback: Callback[JaxState, JaxStats] | None = None,
    ) -> LinearSolution[JaxState, JaxStats]:
        state: JaxState
        stats: JaxStats
        op, state, stats = self.setup(
            op,
            b,
            params,
            fixed_mask=fixed_mask,
            n_fixed=n_fixed,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        if op.bounds != (None, None):
            raise NotImplementedError
        if callback is not None:
            raise NotImplementedError
        state.params_flat, _info = self._wrapped(
            op, state.b_flat, state.params_flat, **self._options(op, state)
        )
        residual: Vector = op(state.params_flat) - state.b_flat
        residual_norm: float = jnp.linalg.norm(residual)
        b_norm: float = jnp.linalg.norm(state.b_flat)
        result: Result
        if residual_norm <= self.atol + self.rtol * b_norm:
            result = Result.SUCCESS
        else:
            result = Result.MAX_STEPS_REACHED
        stats.residual_relative = residual_norm / b_norm
        return self.finalize(op, state, stats, result)

    def _options(self, op: LinearOperator, state: JaxState) -> dict[str, Any]:
        max_steps: int = state.b_flat.size if self.max_steps is None else self.max_steps
        return {
            "tol": self.rtol,
            "atol": self.atol,
            "maxiter": max_steps,
            "M": op.preconditioner,
        }

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Any]:
        raise NotImplementedError
