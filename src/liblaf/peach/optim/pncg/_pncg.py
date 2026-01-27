# ruff: noqa: N803, N806

from typing import override

import attrs
import jarp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from liblaf.peach.optim.base import Objective, Optimizer, Result, Solution
from liblaf.peach.transforms import Transform

from ._types import PNCGState, PNCGStats

type BooleanNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class PNCG(Optimizer[PNCGState, PNCGStats]):
    from ._types import PNCGState as State
    from ._types import PNCGStats as Stats

    type Callback = Optimizer.Callback[State, Stats]
    type Solution = Optimizer.Solution[State, Stats]

    # termination criteria
    max_steps: Integer[Array, ""] = jarp.array(default=1000)
    atol: Scalar = jarp.array(default=0.0)
    rtol: Scalar = jarp.array(default=1e-5)

    def _default_atol_primary(self) -> Scalar:
        return self.atol

    atol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_primary, takes_self=True)
    )

    def _default_rtol_primary(self) -> Scalar:
        return self.rtol

    rtol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_primary, takes_self=True)
    )

    # line search
    max_delta: Scalar = jarp.array(default=jnp.inf)

    # beta
    beta_non_negative: bool = jarp.static(default=True)
    beta_reset_threshold: Scalar = jarp.array(default=jnp.inf)

    # stagnation
    stagnation_max_restarts: Integer[Array, ""] = jarp.array(default=5)
    stagnation_patience: Integer[Array, ""] = jarp.array(default=20)

    # miscellaneous
    jit: bool = jarp.static(default=True)

    @override
    def init[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
    ) -> tuple[State, Stats]:
        transform: Transform = objective.transform
        params_flat: Vector = transform.backward_primals(params)
        state = PNCGState(
            n_steps=jnp.zeros((), jnp.int32),
            alpha=jnp.empty(()),
            beta=jnp.empty(()),
            decrease=jnp.asarray(jnp.inf),
            first_decrease=jnp.asarray(jnp.inf),
            grad=jnp.empty_like(params_flat),
            hess_diag=jnp.empty_like(params_flat),
            hess_quad=jnp.empty(()),
            params=params_flat,
            preconditioner=jnp.empty_like(params_flat),
            search_direction=jnp.empty_like(params_flat),
            best_decrease=jnp.asarray(jnp.inf),
            best_params=params_flat,
            stagnation_counter=jnp.zeros((), jnp.int32),
            stagnation_restarts=jnp.zeros((), jnp.int32),
        )
        stats = PNCGStats(relative_decrease=jnp.asarray(jnp.inf))
        return state, stats

    @override
    def step[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
    ) -> tuple[ModelState, State]:
        assert objective.update is not None
        assert objective.grad is not None
        assert objective.hess_diag is not None
        assert objective.hess_quad is not None

        transform: Transform = objective.transform

        x_tree: Params = transform.forward_primals(opt_state.params)
        model_state = objective.update(model_state, x_tree)
        g_tree: Params = objective.grad(model_state)
        g_flat: Vector = transform.backward_tangents(g_tree)
        H_diag_tree: Params = objective.hess_diag(model_state)
        H_diag_flat: Vector = transform.backward_hess_diag(H_diag_tree)
        P_flat: Vector = _make_preconditioner(H_diag_flat)

        beta: Scalar
        beta, opt_state = self._compute_beta(
            grad=g_flat, preconditioner=P_flat, state=opt_state
        )
        p_flat: Vector = -P_flat * g_flat + beta * opt_state.search_direction
        p_tree: Params = transform.forward_tangents(p_flat)
        pHp: Scalar = objective.hess_quad(model_state, p_tree)
        alpha: Scalar = _compute_alpha(g_flat, p_flat, pHp)
        delta_x: Vector = alpha * p_flat
        delta_x = jnp.clip(delta_x, -self.max_delta, self.max_delta)
        decrease: Scalar = (
            -alpha * jnp.vdot(g_flat, p_flat) - 0.5 * jnp.square(alpha) * pHp
        )

        opt_state.first_decrease = jax.lax.select(
            opt_state.n_steps == 0, decrease, opt_state.first_decrease
        )
        opt_state.alpha = alpha
        opt_state.beta = beta
        opt_state.decrease = decrease
        opt_state.grad = g_flat
        opt_state.hess_diag = H_diag_flat
        opt_state.hess_quad = pHp
        opt_state.params += delta_x
        opt_state.preconditioner = P_flat
        opt_state.search_direction = p_flat
        opt_state = self._detect_stagnation(opt_state)
        opt_state.n_steps += 1
        return model_state, opt_state

    @override
    def update_stats[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
        stats: Stats,
    ) -> Stats:
        stats.relative_decrease = opt_state.decrease / opt_state.first_decrease
        return stats

    @override
    def terminate[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
        opt_stats: Stats,
    ) -> BooleanNumeric:
        return (
            (opt_state.n_steps > self.max_steps)
            | (opt_state.stagnation_restarts > self.stagnation_max_restarts)
            | (
                jnp.isfinite(opt_state.first_decrease)
                & (
                    opt_state.decrease
                    <= self.atol_primary + self.rtol_primary * opt_state.first_decrease
                )
            )
        )

    @override
    def postprocess[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        result: Optimizer.Result = Result.UNKNOWN_ERROR
        if (
            opt_state.best_decrease
            <= self.atol_primary + self.rtol_primary * opt_state.first_decrease
        ):
            result = Result.PRIMARY_SUCCESS
        elif opt_state.decrease <= self.atol + self.rtol * opt_state.first_decrease:
            result = Result.SECONDARY_SUCCESS
        elif opt_state.n_steps > self.max_steps:
            result = Result.MAX_STEPS_REACHED
        elif opt_state.stagnation_restarts > self.stagnation_max_restarts:
            result = Result.STAGNATION
        return Solution(result=result, state=opt_state, stats=opt_stats)

    def _compute_beta(
        self, grad: Vector, preconditioner: Vector, state: PNCGState
    ) -> tuple[Scalar, PNCGState]:
        def _first_step(
            _grad: Vector, _preconditioner: Vector, state: PNCGState
        ) -> tuple[Scalar, PNCGState]:
            beta: Scalar = jnp.zeros_like(state.beta)
            return beta, state

        def _stagnation(
            _grad: Vector, _preconditioner: Vector, state: PNCGState
        ) -> tuple[Scalar, PNCGState]:
            beta: Scalar = jnp.zeros_like(state.beta)
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
            state.stagnation_restarts += 1
            return beta, state

        def _normal_step(
            grad: Vector, preconditioner: Vector, state: PNCGState
        ) -> tuple[Scalar, PNCGState]:
            beta: Scalar = _compute_beta(
                g=grad,
                g_prev=state.grad,
                p_prev=state.search_direction,
                P=preconditioner,
            )
            return beta, state

        index: Integer[Array, ""] = jax.lax.select(
            state.n_steps == 0,
            0,
            jax.lax.select(state.stagnation_counter >= self.stagnation_patience, 1, 2),
        )
        return jax.lax.switch(
            index, [_first_step, _stagnation, _normal_step], grad, preconditioner, state
        )

    def _detect_stagnation(self, state: PNCGState) -> PNCGState:
        def true_fun(state: PNCGState) -> PNCGState:
            state.stagnation_counter += 1
            return state

        def false_fun(state: PNCGState) -> PNCGState:
            state.best_decrease = state.decrease
            state.best_params = state.params
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
            return state

        return jax.lax.cond(
            state.decrease > state.best_decrease, true_fun, false_fun, state
        )


@jarp.jit(inline=True)
def _make_preconditioner(hess_diag: Vector) -> Vector:
    hess_diag = jnp.abs(hess_diag)
    hess_diag_mean: Scalar = jnp.mean(hess_diag, where=hess_diag > 0.0)
    hess_diag = jnp.where(hess_diag > 0.0, hess_diag, hess_diag_mean)
    return jnp.reciprocal(hess_diag)


@jarp.jit(inline=True)
def _compute_alpha(g: Vector, p: Vector, pHp: Scalar) -> Scalar:
    alpha: Scalar = -jnp.vdot(g, p) / pHp
    alpha = jnp.nan_to_num(alpha, nan=0.0)
    return alpha


@jarp.jit(inline=True)
def _compute_beta(g: Vector, g_prev: Vector, p_prev: Vector, P: Vector) -> Scalar:
    y: Vector = g - g_prev
    yTp: Scalar = jnp.vdot(y, p_prev)
    Py: Scalar = P * y
    beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
        jnp.vdot(p_prev, g) / yTp
    )
    return beta
