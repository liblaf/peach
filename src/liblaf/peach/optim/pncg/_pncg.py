# ruff: noqa: N803, N806

from __future__ import annotations

from collections.abc import Iterable
from typing import override

import attrs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

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

    Solution = OptimizeSolution[State, Stats]

    max_steps: Integer[Array, ""] = tree.array(
        default=256, converter=tree.converters.asarray, kw_only=True
    )
    atol: Scalar = tree.array(
        default=1e-5, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=1e-3, converter=tree.converters.asarray, kw_only=True
    )

    def _default_atol_primary(self) -> Scalar:
        return 1e-5 * self.atol

    atol_primary: Scalar = tree.array(
        default=attrs.Factory(_default_atol_primary, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

    def _default_rtol_primary(self) -> Scalar:
        return 1e-2 * self.rtol

    rtol_primary: Scalar = tree.array(
        default=attrs.Factory(_default_rtol_primary, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

    stagnation_patience: Integer[Array, ""] = tree.array(
        default=20, converter=tree.converters.asarray, kw_only=True
    )
    stagnation_max_restarts: Integer[Array, ""] = tree.array(
        default=5, converter=tree.converters.asarray, kw_only=True
    )

    beta_non_negative: Bool[Array, ""] = tree.array(
        default=False, converter=tree.converters.asarray, kw_only=True
    )
    beta_restart_threshold: Scalar = tree.array(
        default=jnp.inf, converter=tree.converters.asarray, kw_only=True
    )
    max_delta: Scalar = tree.array(
        default=jnp.inf, converter=tree.converters.asarray, kw_only=True
    )

    @override
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[State, Stats]:
        params_flat: Vector
        objective, params_flat, constraints = objective.flatten(
            params, constraints=constraints
        )
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        assert objective.structure is not None
        state: PNCG.State = self.State(
            params_flat=params_flat,
            structure=objective.structure,
            best_params_flat=params_flat,
        )
        return SetupResult(objective, constraints, state, self.Stats())

    @override
    def postprocess(
        self,
        objective: Objective,
        state: State,
        stats: Stats,
        result: Result,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> OptimizeSolution[State, Stats]:
        state.params_flat = state.best_params_flat
        return super().postprocess(
            objective, state, stats, result, constraints=constraints
        )

    @override
    def step(
        self,
        objective: Objective,
        state: State,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> State:
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
        elif state.stagnation_counter >= self.stagnation_patience:
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
            state.stagnation_restarts += 1
            beta = jnp.zeros(())
            p = -P * g
        else:
            beta = self._compute_beta(
                g_prev=state.grad_flat, g=g, p=state.search_direction_flat, P=P
            )
            p = -P * g + beta * state.search_direction_flat
        pHp: Scalar = objective.hess_quad(state.params_flat, p)
        alpha: Scalar = self._compute_alpha(g, p, pHp)
        delta_x: Vector = alpha * p
        delta_x = jnp.clip(delta_x, -self.max_delta, self.max_delta)
        state.params_flat += delta_x
        DeltaE: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * alpha**2 * pHp
        if state.first_decrease is None:
            state.first_decrease = DeltaE
        if DeltaE > state.best_decrease:
            state.stagnation_counter += 1
        else:
            state.best_decrease = DeltaE
            state.best_params_flat = state.params_flat
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
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
        state: State,
        stats: Stats,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[bool, Result]:
        if constraints:
            raise NotImplementedError
        assert state.first_decrease is not None
        stats.relative_decrease = state.decrease / state.first_decrease
        if (
            not jnp.isfinite(state.decrease)
            or (state.alpha is not None and not jnp.isfinite(state.alpha))
            or (state.beta is not None and not jnp.isfinite(state.beta))
        ):
            return False, Result.NAN
        if (
            state.decrease
            < self.atol_primary + self.rtol_primary * state.first_decrease
        ):
            return True, Result.SUCCESS
        if stats.n_steps >= self.max_steps:
            if state.best_decrease < self.atol + self.rtol * state.first_decrease:
                return True, Result.SUCCESS
            return True, Result.MAX_STEPS_REACHED
        if state.stagnation_restarts >= self.stagnation_max_restarts:
            return True, Result.STAGNATION
        return False, Result.UNKNOWN_ERROR

    @eqx.filter_jit
    def _compute_alpha(self, g: Vector, p: Vector, pHp: Scalar) -> Scalar:
        alpha: Scalar = -jnp.vdot(g, p) / pHp
        alpha = jnp.nan_to_num(alpha, nan=1.0)
        return alpha

    @eqx.filter_jit
    def _compute_beta(self, g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
        y: Vector = g - g_prev
        yTp: Scalar = jnp.vdot(y, p)
        Py: Scalar = P * y
        beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        beta = jnp.nan_to_num(beta, nan=0.0)
        beta = jnp.where(self.beta_non_negative, jnp.maximum(beta, 0.0), beta)
        beta = jnp.where(self.beta_restart_threshold < beta, 0.0, beta)
        return beta
