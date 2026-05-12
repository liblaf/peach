# ruff: noqa: N806
from typing import cast, override

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from liblaf import jarp
from liblaf.peach.optim.base import (
    BaseProblem,
    Optimizer,
    Problem,
    Result,
    Solution,
)

from ._direction import DirectionUpdate
from ._hess_damping import HessianDamping, HessianDampingState
from ._terminate import ConvergenceCriteria, ConvergenceState
from ._types import PncgState, PncgStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define(kw_only=True)
class Pncg(Optimizer[PncgState, PncgStats]):
    from ._line_search import LineSearch, LineSearchState
    from ._types import PncgState as State
    from ._types import PncgStats as Stats

    type Solution = Optimizer.Solution[State, Stats]

    criteria: ConvergenceCriteria = jarp.field(factory=ConvergenceCriteria)
    direction: DirectionUpdate = jarp.field(factory=DirectionUpdate)
    hess_damping: HessianDamping = jarp.field(factory=HessianDamping)
    line_search: LineSearch = jarp.field(factory=LineSearch)

    @override
    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> State:
        return self.State(
            params=params,
            grad=jnp.full_like(params, jnp.nan),
            direction=jnp.full_like(params, jnp.nan),
            convergence_state=self.criteria.init(params),
            hess_damping_state=self.hess_damping.init(),
            line_search_state=self.line_search.init(),
        )

    @override
    def step[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State
    ) -> tuple[X, State]:
        problem: Problem[X] = cast("Problem[X]", problem)
        params: Vector = opt_state.params
        convergence_state: ConvergenceState = opt_state.convergence_state
        hess_damping_state: HessianDampingState = opt_state.hess_damping_state

        model_state: X = problem.before_step(model_state, params)
        g: Vector = problem.grad(model_state)
        H_diag: Vector = problem.hess_diag(model_state)
        H_diag_damp, hess_damping_state = self.hess_damping.hess_diag(
            state=hess_damping_state, H_diag=H_diag
        )
        P: Vector = jnp.reciprocal(H_diag_damp)

        p: Vector = self.direction(
            g=g,
            g_prev=opt_state.grad,
            P=P,
            p_prev=opt_state.direction,
            restart=opt_state.n_steps == 0,
        )

        pHp: Scalar = problem.hess_quad(model_state, p)
        pHp_damp: Scalar = self.hess_damping.hess_quad(
            state=hess_damping_state, p=p, pHp=pHp
        )
        line_search_state, model_state = self.line_search(
            problem=problem,
            model_state=model_state,
            params=params,
            p=p,
            g=g,
            pHp=pHp_damp,
        )
        alpha: Scalar = line_search_state.alpha

        actual_decrease: Scalar = line_search_state.f0 - line_search_state.f_alpha
        predicted_decrease: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * alpha**2 * pHp_damp
        hess_damping_state: HessianDampingState = self.hess_damping.update(
            hess_damping_state,
            actual_decrease=actual_decrease,
            line_search_steps=line_search_state.step,
            predicted_decrease=predicted_decrease,
        )

        convergence_state: ConvergenceState = self.criteria.update(
            state=convergence_state, g=g
        )

        x: Vector = params + alpha * p
        opt_state: Pncg.State = attrs.evolve(
            opt_state,
            params=x,
            grad=g,
            direction=p,
            convergence_state=convergence_state,
            hess_damping_state=hess_damping_state,
            line_search_state=line_search_state,
            n_steps=opt_state.n_steps + 1,
        )
        return model_state, opt_state

    @override
    def terminate[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State
    ) -> tuple[Bool[Array, ""], Result]:
        criteria_state: ConvergenceState = opt_state.convergence_state
        return self.criteria.terminate(criteria_state)

    @override
    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State, result: Result
    ) -> Solution:
        stats: PncgStats = self.Stats()
        return Solution(result=result, state=opt_state, stats=stats)
