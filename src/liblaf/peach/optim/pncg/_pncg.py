# ruff: noqa: N803, N806

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, PyTree

from liblaf.peach import compile_utils, tree
from liblaf.peach.constraints import Constraints
from liblaf.peach.functools import Objective
from liblaf.peach.optim.abc import Optimizer, OptimizeSolution, Problem, Result
from liblaf.peach.transforms import LinearTransform

from ._state import PNCGState
from ._stats import PNCGStats

type Params = PyTree
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

logger: logging.Logger = logging.getLogger(__name__)


@tree.define
class PNCG(Optimizer[PNCGState, PNCGStats]):
    from ._state import PNCGState as State
    from ._stats import PNCGStats as Stats

    Solution = OptimizeSolution[State, Stats]

    atol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=1e-5, converter=tree.converters.asarray, kw_only=True
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
    def init[T](
        self,
        objective: Objective,
        params: T,
        *,
        constraints: Constraints | None,
        transform: LinearTransform[Vector, T] | None,
    ) -> tuple[Problem, State, Stats]:
        self._warn_unsupported_constraints(constraints)
        if constraints is None:
            constraints = Constraints()
        params_flat: Vector
        params_flat, transform = self._transform_params(params, transform)
        problem = Problem(
            objective=objective, constraints=constraints, transform=transform
        )
        state: PNCGState = self.State(
            params=params_flat, search_direction=jnp.zeros_like(params_flat)
        )
        stats: PNCGStats = self.Stats()
        return problem, state, stats

    @override
    def postprocess[T](
        self, problem: Problem, state: State, stats: Stats, result: Result
    ) -> OptimizeSolution[State, Stats]:
        state.params = state.best_params
        stats.relative_decrease = state.best_decrease / state.first_decrease
        return super().postprocess(problem, state, stats, result)

    @override
    def step(self, problem: Problem, state: State) -> State:
        objective: Objective = problem.objective
        constr: Constraints = problem.constraints
        transform: LinearTransform[Vector, Params] = problem.transform
        assert objective.grad is not None
        assert objective.hess_diag is not None
        assert objective.hess_quad is not None

        params_tree: Params
        lin_fn: Callable[[Vector], Params]
        params_tree, lin_fn = transform.linearize(state.params)
        g_tree: Params = objective.grad(params_tree)
        g_tree = constr.project_grads(params_tree, g_tree)
        g: Vector = transform.linear_transpose(g_tree)
        H_diag_tree: Params = objective.hess_diag(params_tree)
        H_diag: Vector = transform.backward_hess_diag(H_diag_tree)
        P: Vector = _compute_preconditioner(H_diag)

        beta: Scalar
        if state.first_decrease is None:
            beta = jnp.zeros(())
        elif state.stagnation_counter >= self.stagnation_patience:
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
            state.stagnation_restarts += 1
            beta = jnp.zeros(())
        else:
            beta = self._compute_beta(
                g_prev=state.grad, g=g, p=state.search_direction, P=P
            )

        p: Vector = -P * g + beta * state.search_direction
        p_tree: T = lin_fn(p)
        p_tree = self._project_grad(p_tree, constraints)
        pHp: Scalar = objective.hess_quad(params, p_tree)
        alpha: Scalar = self._compute_alpha(g, p, pHp)
        delta_x: Vector = alpha * p
        delta_x = jnp.clip(delta_x, -self.max_delta, self.max_delta)
        state.params += delta_x
        DeltaE: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * alpha**2 * pHp
        if state.first_decrease is None:
            state.first_decrease = DeltaE
        if DeltaE > state.best_decrease:
            state.stagnation_counter += 1
        else:
            state.best_decrease = DeltaE
            state.best_params = state.params
            state.stagnation_counter = jnp.zeros_like(state.stagnation_counter)
        state.alpha = alpha
        state.beta = beta
        state.decrease = DeltaE
        state.grad = g
        state.hess_diag = H_diag
        state.hess_quad = pHp
        state.preconditioner = P
        state.search_direction = p
        return state

    @override
    def terminate[T](
        self,
        objective: Objective,
        state: State[T],
        stats: Stats,
        *,
        constraints: Constraints | None,
    ) -> tuple[bool, Result]:
        assert state.first_decrease is not None
        stats.relative_decrease = state.decrease / state.first_decrease
        done: bool = False
        result: Result = Result.UNKNOWN_ERROR
        if (
            not jnp.isfinite(state.decrease)
            or (state.alpha is not None and not jnp.isfinite(state.alpha))
            or (state.beta is not None and not jnp.isfinite(state.beta))
        ):
            done, result = False, Result.NAN
        elif state.decrease < self.atol + self.rtol * state.first_decrease:
            done, result = True, Result.SUCCESS
        elif stats.n_steps >= self.max_steps:
            done = True
            result = (
                Result.SUCCESS
                if self._check_success(state)
                else Result.MAX_STEPS_REACHED
            )
        elif state.stagnation_restarts >= self.stagnation_max_restarts:
            done = True
            result = Result.SUCCESS if self._check_success(state) else Result.STAGNATION
        else:
            done = False
            result = Result.UNKNOWN_ERROR
        return done, result

    def _check_success(self, state: State) -> Bool[Array, ""]:
        return state.best_decrease < self.atol + self.rtol * state.first_decrease

    def _compute_beta(
        self, state: State, g_prev: Vector, g: Vector, p: Vector, P: Vector
    ) -> Scalar:
        beta: Scalar = _compute_beta(g_prev, g, p, P)
        if self.beta_non_negative:
            beta = jnp.maximum(beta, 0.0)
        if beta > self.beta_restart_threshold:
            beta = jnp.zeros_like(beta)
        return beta

    def _project_grad[T](self, g: T, constraints: Constraints | None = None) -> T:
        if constraints is None:
            return g
        raise NotImplementedError


@compile_utils.jit(inline=True)
def _compute_alpha(g: Vector, p: Vector, pHp: Scalar) -> Scalar:
    alpha: Scalar = -jnp.vdot(g, p) / pHp
    alpha = jnp.nan_to_num(alpha, nan=1.0)
    return alpha


@compile_utils.jit(inline=True)
def _compute_beta(g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
    y: Vector = g - g_prev
    yTp: Scalar = jnp.vdot(y, p)
    Py: Scalar = P * y
    beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
        jnp.vdot(p, g) / yTp
    )
    beta = jnp.nan_to_num(beta, nan=0.0)
    return beta


@compile_utils.jit(inline=True)
def _compute_preconditioner(hess_diag: Vector) -> Vector:
    hess_diag = jnp.where(hess_diag <= 0.0, 1.0, hess_diag)
    preconditioner: Vector = jnp.reciprocal(hess_diag)
    return preconditioner
