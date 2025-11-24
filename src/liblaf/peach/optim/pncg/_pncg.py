# ruff: noqa: N803, N806

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim.abc import (
    Optimizer,
    OptimizeSolution,
    Params,
    Result,
    SetupResult,
)
from liblaf.peach.optim.objective import Objective

from ._state import PNCGState
from ._stats import PNCGStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class PNCG(Optimizer[PNCGState, PNCGStats]):
    from ._state import PNCGState as State
    from ._stats import PNCGStats as Stats

    Solution = OptimizeSolution[PNCGState, PNCGStats]

    max_steps: int = tree.field(default=256, kw_only=True)
    norm: Callable[[Params], Scalar] | None = tree.field(default=None, kw_only=True)

    if TYPE_CHECKING:
        atol: ScalarLike = tree.field(default=1e-28, kw_only=True)
        rtol: ScalarLike = tree.field(default=1e-5, kw_only=True)
        d_hat: ScalarLike = tree.field(default=jnp.inf, kw_only=True)
    else:
        atol: Scalar = tree.array(default=1e-28, kw_only=True)
        rtol: Scalar = tree.array(default=1e-5, kw_only=True)
        d_hat: Scalar = tree.array(default=jnp.inf, kw_only=True)

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
        alpha: Scalar = self._compute_alpha(g=g, p=p, pHp=pHp, unflatten=state.flat_def)
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
        if state.decrease < self.atol + self.rtol * state.first_decrease:
            return True, Result.SUCCESS
        if stats.n_steps >= self.max_steps:
            return True, Result.MAX_STEPS_REACHED
        return False, Result.UNKNOWN_ERROR

    @eqx.filter_jit
    def _compute_alpha(
        self,
        g: Vector,
        p: Vector,
        pHp: Scalar,
        unflatten: Callable[[Array], Params] | None = None,
    ) -> Scalar:
        p_norm: Scalar
        if self.norm is None:
            p_norm = jnp.linalg.norm(p, ord=jnp.inf)
        else:
            p_tree: Params = p if unflatten is None else unflatten(p)
            p_norm = self.norm(p_tree)
        alpha_1: Scalar = self.d_hat / (2.0 * p_norm)  # pyright: ignore[reportAssignmentType]
        alpha_2: Scalar = -jnp.vdot(g, p) / pHp
        alpha: Scalar = jnp.minimum(alpha_1, alpha_2)
        alpha = jnp.nan_to_num(alpha)
        return alpha

    @eqx.filter_jit
    def _compute_beta(self, g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
        y: Vector = g - g_prev
        yTp: Scalar = jnp.vdot(y, p)
        Py: Scalar = P * y
        beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        return beta
