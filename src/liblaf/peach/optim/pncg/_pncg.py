# ruff: noqa: N803, N806

from collections.abc import Callable
from typing import TYPE_CHECKING, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike

from liblaf.peach import tree_utils
from liblaf.peach.optim.abc import Optimizer, Params, Result
from liblaf.peach.optim.objective import Objective

from ._state import PNCGState
from ._stats import PNCGStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree_utils.define
class PNCG(Optimizer[PNCGState, PNCGStats]):
    max_steps: int = 500
    norm: Callable[[Params], Scalar] | None = None

    if TYPE_CHECKING:
        atol: ScalarLike = 1e-28
        rtol: ScalarLike = 1e-5
        d_hat: ScalarLike = jnp.inf
    else:
        atol: Scalar = tree_utils.array(default=1e-28)
        rtol: Scalar = tree_utils.array(default=1e-5)
        d_hat: Scalar = tree_utils.array(default=jnp.inf)

    @override
    def init(
        self, objective: Objective, params: Params
    ) -> tuple[Objective, PNCGState, PNCGStats]:
        params_flat: Vector
        unflatten: Callable[[Vector], Params]
        params_flat, unflatten = tree_utils.flatten(params)
        objective = objective.flatten(unflatten)
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        state = PNCGState(params_flat=params_flat, unflatten=unflatten)
        return objective, state, PNCGStats()

    @override
    def step(self, objective: Objective, state: PNCGState) -> PNCGState:
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
        alpha: Scalar = self._compute_alpha(
            g=g, p=p, pHp=pHp, unflatten=state.unflatten
        )
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
        self, objective: Objective, state: PNCGState, stats: PNCGStats
    ) -> tuple[bool, Result]:
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
