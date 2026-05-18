from typing import override

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach.optim.base import Problem
from liblaf.peach.optim.pncg import (
    ConvergenceCriteria,
    HessianDamping,
    LineSearch,
    Pncg,
)

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class QuadraticProblem(Problem[Vector]):
    def __init__(self, target: Vector) -> None:
        self.target = target

    @override
    def before_trial(self, state: Vector, x: Vector, /) -> Vector:
        del state
        return x

    @override
    def fun(self, state: Vector, /) -> Scalar:
        residual: Vector = state - self.target
        return 0.5 * jnp.vdot(residual, residual)

    @override
    def grad(self, state: Vector, /) -> Vector:
        return state - self.target

    @override
    def hess_diag(self, state: Vector, /) -> Vector:
        return jnp.ones_like(state)

    @override
    def hess_quad(self, state: Vector, p: Vector, /) -> Scalar:
        del state
        return jnp.vdot(p, p)


class FractionalStepProblem(QuadraticProblem):
    def __init__(self, target: Vector, step_fraction: Scalar) -> None:
        super().__init__(target)
        self.step_fraction = step_fraction

    @override
    def max_step_size(self, state: Vector, p: Vector, /) -> Scalar:
        del state, p
        return self.step_fraction


def test_hessian_damping_scales_by_mean_positive_abs_diagonal() -> None:
    damping = HessianDamping(initial=jnp.asarray(2.0))
    state = damping.init()

    hess_diag, state = damping.hess_diag(state, jnp.asarray([-2.0, 0.0, 6.0]))

    np.testing.assert_allclose(np.asarray(hess_diag), np.asarray([10.0, 8.0, 14.0]))
    np.testing.assert_allclose(np.asarray(state.hess_diag_mean), 4.0)
    np.testing.assert_allclose(
        np.asarray(damping.hess_quad(state, jnp.asarray([1.0, 2.0]), jnp.asarray(3.0))),
        43.0,
    )


def test_hessian_damping_update_halves_grows_and_caps_factor() -> None:
    damping = HessianDamping(factor_max=jnp.asarray(10.0), initial=jnp.asarray(4.0))
    state = damping.init()

    state = damping.update(
        state,
        actual_decrease=jnp.asarray(5.0),
        line_search_step=jnp.asarray(0),
        predicted_decrease=jnp.asarray(4.0),
    )
    np.testing.assert_allclose(np.asarray(state.factor), 2.0)

    state = damping.update(
        state,
        actual_decrease=jnp.asarray(0.0),
        line_search_step=jnp.asarray(3),
        predicted_decrease=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(np.asarray(state.factor), 6.0)

    state = damping.update(
        state,
        actual_decrease=jnp.asarray(0.0),
        line_search_step=jnp.asarray(3),
        predicted_decrease=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(np.asarray(state.factor), 10.0)


def test_line_search_newton_and_upper_bounds() -> None:
    line_search = LineSearch()

    np.testing.assert_allclose(
        np.asarray(
            line_search.line_search_newton(
                jnp.asarray([1.0, 0.0]), jnp.asarray([-4.0, 0.0]), jnp.asarray(2.0)
            )
        ),
        2.0,
    )
    np.testing.assert_allclose(
        np.asarray(
            line_search.line_search_newton(
                jnp.asarray([1.0, 0.0]), jnp.asarray([-4.0, 0.0]), jnp.asarray(0.0)
            )
        ),
        1.0,
    )
    np.testing.assert_allclose(
        np.asarray(
            line_search.line_search_newton(
                jnp.asarray([1.0, 0.0]), jnp.asarray([4.0, 0.0]), jnp.asarray(2.0)
            )
        ),
        1.0,
    )
    np.testing.assert_allclose(
        np.asarray(
            line_search.line_search_upper(jnp.asarray([2.0, -4.0]), jnp.asarray(1.0))
        ),
        0.25,
    )
    np.testing.assert_allclose(
        np.asarray(line_search.line_search_upper(jnp.zeros(2), jnp.asarray(1.0))),
        0.0,
    )


def test_line_search_clamps_to_max_step_norm_before_trials() -> None:
    problem = QuadraticProblem(target=jnp.zeros(1))
    params: Vector = jnp.asarray([2.0])
    direction: Vector = jnp.asarray([-2.0])
    line_search = LineSearch(max_step_norm=jnp.asarray(1.0))

    state, model_state = line_search(
        problem=problem,
        model_state=params,
        params=params,
        p=direction,
        g=params,
        pHp=jnp.vdot(direction, direction),
    )

    np.testing.assert_allclose(np.asarray(state.alpha), 0.5)
    np.testing.assert_allclose(np.asarray(state.f0), 2.0)
    np.testing.assert_allclose(np.asarray(state.f_alpha), 0.5)
    np.testing.assert_allclose(np.asarray(model_state), np.asarray([1.0]))
    assert bool(state.ok)
    assert int(state.step) == 0


def test_line_search_scales_initial_alpha_by_max_step_fraction() -> None:
    problem = FractionalStepProblem(
        target=jnp.zeros(1), step_fraction=jnp.asarray(0.25)
    )
    params: Vector = jnp.asarray([2.0])
    direction: Vector = jnp.asarray([-2.0])
    line_search = LineSearch(max_step_norm=jnp.asarray(1.0))

    state, model_state = line_search(
        problem=problem,
        model_state=params,
        params=params,
        p=direction,
        g=params,
        pHp=jnp.vdot(direction, direction),
    )

    np.testing.assert_allclose(np.asarray(state.alpha), 0.125)
    np.testing.assert_allclose(np.asarray(model_state), np.asarray([1.75]))
    assert bool(state.ok)
    assert int(state.step) == 0


def test_line_search_keeps_initial_alpha_for_full_max_step_fraction() -> None:
    problem = FractionalStepProblem(target=jnp.zeros(1), step_fraction=jnp.asarray(1.0))
    params: Vector = jnp.asarray([2.0])
    direction: Vector = jnp.asarray([-2.0])
    line_search = LineSearch(max_step_norm=jnp.asarray(1.0))

    state, model_state = line_search(
        problem=problem,
        model_state=params,
        params=params,
        p=direction,
        g=params,
        pHp=jnp.vdot(direction, direction),
    )

    np.testing.assert_allclose(np.asarray(state.alpha), 0.5)
    np.testing.assert_allclose(np.asarray(model_state), np.asarray([1.0]))
    assert bool(state.ok)
    assert int(state.step) == 0


def test_pncg_step_updates_params_and_then_next_step_damping_factor() -> None:
    problem = QuadraticProblem(target=jnp.asarray([3.0]))
    params: Vector = jnp.asarray([0.0])
    optimizer = Pncg(criteria=ConvergenceCriteria(max_steps=jnp.asarray(10)))

    opt_state = optimizer.init(problem, params, params)
    _model_state, opt_state = optimizer.step(problem, params, opt_state)

    np.testing.assert_allclose(np.asarray(opt_state.params), np.asarray([1.5]))
    np.testing.assert_allclose(np.asarray(opt_state.grad), np.asarray([-3.0]))
    np.testing.assert_allclose(np.asarray(opt_state.direction), np.asarray([1.5]))
    np.testing.assert_allclose(np.asarray(opt_state.line_search_state.alpha), 1.0)
    assert int(opt_state.line_search_state.step) == 0
    np.testing.assert_allclose(np.asarray(opt_state.hess_damping_state.factor), 0.5)
