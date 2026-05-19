from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.peach.optim.base import Problem, State
from liblaf.peach.optim.pncg import (
    ConvergenceCriteria,
    HessianDamping,
    LineSearch,
    Pncg,
    PncgState,
)

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


class QuadraticProblem(Problem[Vector]):
    def __init__(self, target: Vector) -> None:
        self.target = target

    @override
    def update(self, state: Vector, x: Vector, /) -> None:
        state.copy_(x)

    @override
    def fun(self, state: Vector, /) -> Scalar:
        residual: Vector = state - self.target
        return 0.5 * torch.dot(residual, residual)

    @override
    def grad(self, state: Vector, /) -> Vector:
        return state - self.target

    @override
    def hess_diag(self, state: Vector, /) -> Vector:
        return torch.ones_like(state)

    @override
    def hess_quad(self, state: Vector, p: Vector, /) -> Scalar:
        del state
        return torch.dot(p, p)


class FractionalStepProblem(QuadraticProblem):
    def __init__(self, target: Vector, step_fraction: Scalar) -> None:
        super().__init__(target)
        self.step_fraction = step_fraction

    @override
    def max_step_size(self, state: Vector, p: Vector, /) -> Scalar:
        del state, p
        return self.step_fraction


class CallbackProblem(QuadraticProblem):
    def __init__(self, target: Vector) -> None:
        super().__init__(target)
        self.callbacks: list[tuple[Vector, Vector, int]] = []

    @override
    def callback(self, model_state: Vector, opt_state: State, /) -> None:
        opt_state = cast("PncgState", opt_state)
        self.callbacks.append(
            (model_state.clone(), opt_state.params.clone(), opt_state.step)
        )


def test_hessian_damping_scales_by_mean_absolute_diagonal() -> None:
    damping = HessianDamping(initial=2.0)
    state = damping.init()

    hess_diag = damping.hess_diag(state, torch.tensor([-2.0, 0.0, 6.0]))

    torch.testing.assert_close(
        hess_diag, torch.tensor([22.0 / 3.0, 16.0 / 3.0, 34.0 / 3.0])
    )
    torch.testing.assert_close(state.hess_diag_mean, torch.tensor(8.0 / 3.0))
    torch.testing.assert_close(
        damping.hess_quad(state, torch.tensor([1.0, 2.0]), torch.tensor(3.0)),
        torch.tensor(89.0 / 3.0),
    )


def test_hessian_damping_update_halves_grows_and_caps_factor() -> None:
    damping = HessianDamping(factor_max=10.0, initial=4.0)
    state = damping.init()

    damping.update(
        state,
        actual_decrease=torch.tensor(5.0),
        line_search_step=0,
        predicted_decrease=torch.tensor(4.0),
    )
    assert state.factor == 2.0

    damping.update(
        state,
        actual_decrease=torch.tensor(0.0),
        line_search_step=3,
        predicted_decrease=torch.tensor(1.0),
    )
    assert state.factor == 10.0

    damping.update(
        state,
        actual_decrease=torch.tensor(5.0),
        line_search_step=0,
        predicted_decrease=torch.tensor(4.0),
    )
    assert state.factor == 5.0


def test_line_search_newton_and_upper_bounds() -> None:
    line_search = LineSearch()

    torch.testing.assert_close(
        line_search.line_search_newton(torch.tensor(-4.0), torch.tensor(2.0)),
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        line_search.line_search_newton(torch.tensor(-4.0), torch.tensor(0.0)),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        line_search.line_search_newton(torch.tensor(4.0), torch.tensor(2.0)),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        line_search.line_search_upper(torch.tensor([2.0, -4.0]), 1.0),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        line_search.line_search_upper(torch.zeros(2), 1.0),
        torch.tensor(0.0),
    )


def test_line_search_clamps_to_max_step_norm_before_trials() -> None:
    problem = QuadraticProblem(target=torch.zeros(1))
    params: Vector = torch.tensor([2.0])
    model_state: Vector = params.clone()
    direction: Vector = torch.tensor([-2.0])
    line_search = LineSearch(max_step_norm=1.0)
    state = line_search.init(fun=problem.fun(params))

    assert (
        line_search(
            state,
            problem=problem,
            model_state=model_state,
            params=params,
            m=torch.dot(params, direction),
            p=direction,
            pHp=torch.dot(direction, direction),
        )
        is None
    )

    torch.testing.assert_close(state.alpha, torch.tensor(0.5))
    torch.testing.assert_close(state.f0, torch.tensor(2.0))
    torch.testing.assert_close(state.f_alpha, torch.tensor(0.5))
    torch.testing.assert_close(model_state, torch.tensor([1.0]))
    torch.testing.assert_close(params, torch.tensor([2.0]))
    assert state.ok
    assert state.step == 0


def test_line_search_scales_initial_alpha_by_max_step_fraction() -> None:
    problem = FractionalStepProblem(
        target=torch.zeros(1), step_fraction=torch.tensor(0.25)
    )
    params: Vector = torch.tensor([2.0])
    model_state: Vector = params.clone()
    direction: Vector = torch.tensor([-2.0])
    line_search = LineSearch(max_step_norm=1.0)
    state = line_search.init(fun=problem.fun(params))

    line_search(
        state,
        problem=problem,
        model_state=model_state,
        params=params,
        m=torch.dot(params, direction),
        p=direction,
        pHp=torch.dot(direction, direction),
    )

    torch.testing.assert_close(state.alpha, torch.tensor(0.125))
    torch.testing.assert_close(model_state, torch.tensor([1.75]))
    torch.testing.assert_close(params, torch.tensor([2.0]))
    assert state.ok
    assert state.step == 0


def test_line_search_keeps_initial_alpha_for_full_max_step_fraction() -> None:
    problem = FractionalStepProblem(
        target=torch.zeros(1), step_fraction=torch.tensor(1.0)
    )
    params: Vector = torch.tensor([2.0])
    model_state: Vector = params.clone()
    direction: Vector = torch.tensor([-2.0])
    line_search = LineSearch(max_step_norm=1.0)
    state = line_search.init(fun=problem.fun(params))

    line_search(
        state,
        problem=problem,
        model_state=model_state,
        params=params,
        m=torch.dot(params, direction),
        p=direction,
        pHp=torch.dot(direction, direction),
    )

    torch.testing.assert_close(state.alpha, torch.tensor(0.5))
    torch.testing.assert_close(model_state, torch.tensor([1.0]))
    torch.testing.assert_close(params, torch.tensor([2.0]))
    assert state.ok
    assert state.step == 0


def test_pncg_step_updates_params_and_then_next_step_damping_factor() -> None:
    problem = QuadraticProblem(target=torch.tensor([3.0]))
    params: Vector = torch.tensor([0.0])
    model_state: Vector = params.clone()
    optimizer = Pncg(criteria=ConvergenceCriteria(max_steps=10))

    opt_state = optimizer.init(problem, model_state, params)
    assert optimizer.step(problem, model_state, opt_state) is None

    torch.testing.assert_close(opt_state.params, torch.tensor([3.0]))
    torch.testing.assert_close(model_state, torch.tensor([3.0]))
    torch.testing.assert_close(params, torch.tensor([0.0]))
    torch.testing.assert_close(opt_state.grad, torch.tensor([-3.0]))
    torch.testing.assert_close(opt_state.direction, torch.tensor([3.0]))
    torch.testing.assert_close(opt_state.line_search_state.alpha, torch.tensor(1.0))
    assert opt_state.line_search_state.step == 0
    assert opt_state.hess_damping_state.factor == 0.0


def test_pncg_minimize_keeps_mutated_optimizer_state_for_callbacks() -> None:
    problem = CallbackProblem(target=torch.tensor([3.0]))
    params: Vector = torch.tensor([0.0])
    model_state: Vector = params.clone()
    optimizer = Pncg(criteria=ConvergenceCriteria(max_steps=10))

    solution = optimizer.minimize(problem, model_state, params)

    assert solution.success
    torch.testing.assert_close(solution.params, torch.tensor([3.0]))
    torch.testing.assert_close(model_state, torch.tensor([3.0]))
    torch.testing.assert_close(params, torch.tensor([0.0]))
    assert len(problem.callbacks) == 2
    for callback_model_state, callback_params, _step in problem.callbacks:
        torch.testing.assert_close(callback_model_state, callback_params)
    assert problem.callbacks[-1][2] == solution.state.step
