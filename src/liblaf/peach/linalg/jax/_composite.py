from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.grapes.errors import UnreachableError

from liblaf.peach import tree
from liblaf.peach.linalg.abc import (
    Callback,
    LinearSolution,
    LinearSolver,
    Params,
    Result,
)
from liblaf.peach.linalg.op import LinearOperator

from ._base import JaxSolver
from ._cg import JaxCG
from ._gmres import JaxGMRES
from ._types import JaxState, JaxStats

type Vector = Float[Array, " N"]


@tree.define
class JaxCompositeSolver(LinearSolver[JaxState, JaxStats]):
    solvers: list[JaxSolver] = tree.field(factory=lambda: [JaxCG(), JaxGMRES()])

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
        return self.solvers[0].setup(
            op,
            b,
            params,
            fixed_mask=fixed_mask,
            n_fixed=n_fixed,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @override
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
        for i, solver in enumerate(self.solvers):
            params_flat: Vector
            params_flat, _info = solver._wrapped(  # noqa: SLF001
                op,
                state.b_flat,
                state.params_flat,
                **solver._options(op, state),  # noqa: SLF001
            )
            residual: Vector = op(params_flat) - state.b_flat
            residual_norm: float = jnp.linalg.norm(residual)
            b_norm: float = jnp.linalg.norm(state.b_flat)
            stats.residual_relative = residual_norm / b_norm
            if residual_norm <= solver.atol + solver.rtol * b_norm:
                state.params_flat = params_flat
                return solver.finalize(op, state, stats, Result.SUCCESS)
            if i + 1 == len(self.solvers):
                state.params_flat = params_flat
                return solver.finalize(op, state, stats, Result.MAX_STEPS_REACHED)
        raise UnreachableError
