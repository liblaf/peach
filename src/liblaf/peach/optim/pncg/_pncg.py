# ruff: noqa: N806
from typing import cast, override

import attrs
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.peach.optim.base import BaseProblem, Optimizer, Problem, Result, Solution

from ._types import PncgState, PncgStats

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define(kw_only=True)
class Pncg(Optimizer[PncgState, PncgStats]):
    """Preconditioned nonlinear conjugate-gradient optimizer.

    `PNCG` builds a diagonal preconditioner from a damped Hessian diagonal,
    computes a Dai-Kou conjugate-gradient direction, and accepts steps with
    Armijo backtracking. The accepted line-search trial state is returned as the
    next model state, so each call to [`step`][liblaf.peach.optim.pncg.PNCG.step]
    expects `model_state` to already match `opt_state.params`.
    """

    from ._direction import DirectionUpdate
    from ._hess_damping import HessianDamping, HessianDampingState
    from ._line_search import LineSearch, LineSearchState
    from ._terminate import ConvergenceCriteria, ConvergenceState
    from ._types import PncgState as State
    from ._types import PncgStats as Stats

    type Solution = Optimizer.Solution[State, Stats]

    criteria: ConvergenceCriteria = attrs.field(factory=ConvergenceCriteria)
    direction: DirectionUpdate = attrs.field(factory=DirectionUpdate)
    hess_damping: HessianDamping = attrs.field(factory=HessianDamping)
    line_search: LineSearch = attrs.field(factory=LineSearch)

    @override
    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> State:
        """Initialize PNCG state."""
        problem: Problem[X] = cast("Problem[X]", problem)
        fun: Scalar = problem.fun(model_state)
        return self.State(
            fun=fun,
            params=params.clone(),
            convergence_state=self.criteria.init(),
            hess_damping_state=self.hess_damping.init(),
            line_search_state=self.line_search.init(fun=fun),
        )

    @override
    def step[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State
    ) -> tuple[X, State]:
        """Run one PNCG step."""
        problem: Problem[X] = cast("Problem[X]", problem)

        g: Vector = problem.grad(model_state)
        H_diag: Vector = problem.hess_diag(model_state)
        H_diag_damp: Vector = self.hess_damping.hess_diag(
            state=opt_state.hess_damping_state, H_diag=H_diag
        )
        P: Vector = torch.reciprocal(H_diag_damp)

        p: Vector = self.direction(
            g=g,
            g_prev=opt_state.grad,
            P=P,
            p_prev=opt_state.direction,
            restart=opt_state.step == 0 or not opt_state.line_search_state.ok,
        )
        opt_state.direction = p
        opt_state.grad = g
        opt_state.hess_diag = H_diag
        slope: Scalar = torch.dot(g, p)
        opt_state.slope = slope

        pHp: Scalar = problem.hess_quad(model_state, p)
        opt_state.hess_quad = pHp
        pHp_damp: Scalar = self.hess_damping.hess_quad(
            state=opt_state.hess_damping_state, p=p, pHp=pHp
        )
        model_state: X = self.line_search(
            opt_state.line_search_state,
            problem=problem,
            model_state=model_state,
            m=slope,
            p=p,
            params=opt_state.params,
            pHp=pHp_damp,
        )
        ls_state: Pncg.LineSearchState = opt_state.line_search_state
        opt_state.fun = ls_state.f_alpha

        alpha: Scalar = ls_state.alpha
        actual_decrease: Scalar = ls_state.f0 - ls_state.f_alpha
        predicted_decrease: Scalar = -alpha * slope - 0.5 * alpha**2 * pHp_damp
        self.hess_damping.update(
            opt_state.hess_damping_state,
            actual_decrease=actual_decrease,
            line_search_step=ls_state.step,
            predicted_decrease=predicted_decrease,
        )

        self.criteria.update(
            opt_state.convergence_state, g=g, line_search_ok=ls_state.ok
        )

        opt_state.params += alpha * p
        return model_state, opt_state

    @override
    def terminate[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State
    ) -> tuple[bool, Result]:
        """Delegate stopping to the configured convergence criteria."""
        return self.criteria.terminate(opt_state.convergence_state)

    @override
    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: State, result: Result
    ) -> Solution:
        """Build a PNCG solution object."""
        stats: Pncg.Stats = self.Stats()
        return Solution(result=result, state=opt_state, stats=stats)
