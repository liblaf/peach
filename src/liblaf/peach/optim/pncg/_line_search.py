# ruff: noqa: N803, N806
import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from liblaf import jarp
from liblaf.peach.optim.base import Problem
from liblaf.peach.utils import implemented

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class LineSearchState:
    """State returned by Armijo backtracking."""

    alpha: Scalar
    f0: Scalar
    m: Scalar
    f_alpha: Scalar = jarp.array(default=jnp.asarray(jnp.nan))
    ok: Bool[Array, ""] = jarp.array(default=jnp.zeros((), jnp.bool))
    step: Integer[Array, ""] = jarp.array(default=jnp.zeros((), jnp.int32))


@jarp.frozen
class LineSearch:
    """Armijo backtracking with Newton, norm, and problem-specific step bounds."""

    armijo: Scalar = jarp.array(default=jnp.asarray(1e-4))
    max_step_norm: Scalar = jarp.array(default=jnp.asarray(jnp.inf))
    max_steps: Integer[Array, ""] = jarp.array(default=jnp.asarray(20, jnp.int32))

    @jarp.fallback_jit(inline=True)
    def __call__[X](
        self,
        problem: Problem[X],
        model_state: X,
        params: Vector,
        p: Vector,
        g: Vector,
        pHp: Scalar,
    ) -> tuple[LineSearchState, X]:
        """Run line search along direction `p` from `params`.

        The initial step length is the smaller of the Newton proposal and the
        configured infinity-norm bound. If the problem implements
        [`Problem.max_step_size`][liblaf.peach.optim.base.Problem.max_step_size],
        that hook receives the proposed displacement `alpha * p` and returns a
        safe fraction of it. Every accepted or rejected trial is materialized
        through [`Problem.before_trial`][liblaf.peach.optim.base.Problem.before_trial].
        """
        alpha_upper: Scalar = self.line_search_upper(p, self.max_step_norm)
        alpha_newton: Scalar = self.line_search_newton(p, g, pHp)
        alpha: Scalar = jnp.nanmin(jnp.asarray([alpha_newton, alpha_upper]))
        if implemented(problem, Problem.max_step_size):
            step_fraction: Scalar = problem.max_step_size(model_state, alpha * p)
            alpha *= jnp.clip(step_fraction, 0.0, 1.0)

        def cond_fun(carry: tuple[LineSearchState, X]) -> Bool[Array, ""]:
            state, _model_state = carry
            return (state.step < self.max_steps) & ~state.ok

        def body_fun(carry: tuple[LineSearchState, X]) -> tuple[LineSearchState, X]:
            state, model_state = carry
            alpha: Scalar = state.alpha
            alpha: Scalar = jnp.where(state.step == 0, alpha, 0.5 * alpha)
            model_state: X = problem.before_trial(model_state, params + alpha * p)
            f_alpha: Scalar = problem.fun(model_state)
            ok: Bool[Array, ""] = self.armijo_condition(
                f_alpha, state.f0, alpha, state.m
            )
            step: Integer[Array, ""] = jnp.where(ok, state.step, state.step + 1)
            return attrs.evolve(
                state, alpha=alpha, f_alpha=f_alpha, ok=ok, step=step
            ), model_state

        f0: Scalar = problem.fun(model_state)
        m: Scalar = jnp.vdot(g, p)
        state: LineSearchState = LineSearchState(
            alpha=alpha,
            f0=f0,
            m=m,
            f_alpha=jnp.where(alpha == 0.0, f0, jnp.nan),
            ok=alpha == 0.0,
        )
        state, model_state = jarp.while_loop(cond_fun, body_fun, (state, model_state))
        return state, model_state

    def init(self) -> LineSearchState:
        """Create an empty line-search state."""
        return LineSearchState(
            alpha=jnp.asarray(jnp.nan), f0=jnp.asarray(jnp.nan), m=jnp.asarray(jnp.nan)
        )

    @staticmethod
    @jax.jit(inline=True)
    def line_search_newton(p: Vector, g: Vector, pHp: Scalar) -> Scalar:
        """Return the Newton step length or `1.0` when curvature is unsuitable."""
        gTp: Scalar = jnp.vdot(g, p)
        alpha: Scalar = -gTp / pHp
        return jnp.where((pHp <= 0.0) | (gTp >= 0.0), 1.0, alpha)

    @staticmethod
    @jax.jit(inline=True)
    def line_search_upper(p: Vector, max_step_norm: Scalar) -> Scalar:
        """Return the largest step satisfying the infinity-norm bound."""
        p_norm: Scalar = jnp.linalg.norm(p, ord=jnp.inf)
        return jnp.where(p_norm == 0.0, 0.0, max_step_norm / p_norm)

    def armijo_condition(
        self, f_alpha: Scalar, f0: Scalar, alpha: Scalar, m: Scalar
    ) -> Bool[Array, ""]:
        """Evaluate the Armijo sufficient-decrease condition."""
        return f_alpha <= f0 + alpha * self.armijo * m
