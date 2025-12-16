# ruff: noqa: N803, N806

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import override

import attrs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim.abc import (
    Optimizer,
    OptimizeSolution,
    Params,
    Result,
    SetupResult,
)
from liblaf.peach.optim.linesearch import LineSearch
from liblaf.peach.optim.objective import Objective

from ._state import PNCGState
from ._stats import PNCGStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


def _default_line_search(self: PNCG) -> LineSearch:
    return self.default_line_search()


@tree.define
class PNCG(Optimizer[PNCGState, PNCGStats]):
    from ._state import PNCGState as State
    from ._stats import PNCGStats as Stats

    Solution = OptimizeSolution[PNCGState, PNCGStats]

    norm: Callable[[Params], Scalar] | None = tree.field(default=None, kw_only=True)
    line_search: LineSearch = tree.field(
        default=attrs.Factory(_default_line_search, takes_self=True), kw_only=True
    )

    max_steps: Integer[Array, ""] = tree.array(
        default=256, converter=tree.converters.asarray, kw_only=True
    )
    clamp_beta: Bool[Array, ""] = tree.array(
        default=False, converter=tree.converters.asarray, kw_only=True
    )
    atol: Scalar = tree.array(
        default=1e-15, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=1e-5, converter=tree.converters.asarray, kw_only=True
    )

    @classmethod
    def default_line_search(
        cls, *, d_hat: Float[ArrayLike, ""] | None = None
    ) -> LineSearch:
        from liblaf.peach.optim.linesearch import (
            LineSearchCollisionRepulsionThreshold,
            LineSearchMin,
            LineSearchNaive,
            LineSearchSingleNewton,
        )

        method: LineSearch = LineSearchSingleNewton()
        if d_hat is not None:
            method = LineSearchMin(
                [method, LineSearchCollisionRepulsionThreshold(d_hat)]
            )
        method = LineSearchNaive(method, max_steps=2)
        return method

    @override
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[PNCGState, PNCGStats]:
        params_flat: Vector
        objective, params_flat, constraints = objective.flatten(
            params, constraints=constraints
        )
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        assert objective.flat_def is not None
        state = PNCGState(params_flat=params_flat, flat_def=objective.flat_def)
        return SetupResult(objective, constraints, state, PNCGStats())

    @override
    def step(
        self,
        objective: Objective,
        state: PNCGState,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> PNCGState:
        if constraints:
            raise NotImplementedError
        assert objective.grad_and_hess_diag is not None
        assert objective.hess_quad is not None
        g: Vector
        H_diag: Vector
        g, H_diag = objective.grad_and_hess_diag(state.params_flat)
        H_diag = jnp.where(H_diag <= 0.0, 1.0, H_diag)
        P: Vector = jnp.reciprocal(H_diag)
        beta: Scalar
        p: Vector
        if state.search_direction_flat is None:
            beta = jnp.zeros(())
            p = -P * g
        else:
            beta = self._compute_beta(
                g_prev=state.grad_flat, g=g, p=state.search_direction_flat, P=P
            )
            p = -P * g + beta * state.search_direction_flat
        pHp: Scalar = objective.hess_quad(state.params_flat, p)
        alpha: Scalar = self.line_search.search(objective, state.params_flat, g, p)
        state.params_flat += alpha * p
        DeltaE: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * alpha**2 * pHp
        if state.first_decrease is None:
            state.first_decrease = DeltaE
        state.alpha = alpha
        state.beta = beta
        state.decrease = DeltaE
        state.grad_flat = g
        state.hess_diag_flat = H_diag
        state.hess_quad = pHp
        state.preconditioner_flat = P
        state.search_direction_flat = p
        return state

    @override
    def terminate(
        self,
        objective: Objective,
        state: PNCGState,
        stats: PNCGStats,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[bool, Result]:
        if constraints:
            raise NotImplementedError
        assert state.first_decrease is not None
        if (
            not jnp.isfinite(state.decrease)
            or (state.alpha is not None and not jnp.isfinite(state.alpha))
            or (state.beta is not None and not jnp.isfinite(state.beta))
        ):
            return False, Result.NAN
        if state.decrease < self.atol + self.rtol * state.first_decrease:
            return True, Result.SUCCESS
        if stats.n_steps >= self.max_steps:
            return True, Result.MAX_STEPS_REACHED
        return False, Result.UNKNOWN_ERROR

    # @eqx.filter_jit
    # def _compute_alpha(
    #     self,
    #     g: Vector,
    #     p: Vector,
    #     pHp: Scalar,
    #     unflatten: Callable[[Array], Params] | None = None,
    # ) -> Scalar:
    #     p_norm: Scalar
    #     if self.norm is None:
    #         p_norm = jnp.linalg.norm(p, ord=jnp.inf)
    #     else:
    #         p_tree: Params = p if unflatten is None else unflatten(p)
    #         p_norm = self.norm(p_tree)
    #     alpha_1: Scalar = self.d_hat / (2.0 * p_norm)  # pyright: ignore[reportAssignmentType]
    #     alpha_2: Scalar = -jnp.vdot(g, p) / pHp
    #     alpha: Scalar = jnp.minimum(alpha_1, alpha_2)
    #     alpha = jnp.nan_to_num(alpha, nan=1.0)
    #     return alpha

    @eqx.filter_jit
    def _compute_beta(self, g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
        y: Vector = g - g_prev
        yTp: Scalar = jnp.vdot(y, p)
        Py: Scalar = P * y
        beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        beta = jnp.nan_to_num(beta, nan=0.0)
        beta = jnp.where(self.clamp_beta, jnp.maximum(beta, 0.0), beta)
        return beta
