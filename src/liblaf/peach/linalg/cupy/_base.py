from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, override

import attrs
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.peach.linalg import utils
from liblaf.peach.linalg.base import (
    LinearSolution,
    LinearSolver,
    LinearSystem,
    Result,
)

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import linalg


type Free = Float[Array, " free"]
type FreeCp = Float[cp.ndarray, " free"]
type Scalar = Float[Array, ""]


@jarp.define
class CupySolver(LinearSolver):
    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    Solution = LinearSolution[State, Stats]

    max_steps: Integer[Array, ""] | None = jarp.field(default=None, kw_only=True)
    rtol: Scalar = jarp.array(default=1e-3, kw_only=True)
    atol: Scalar = jarp.array(default=0.0, kw_only=True)

    def _default_rtol_primary(self) -> Scalar:
        return 1e-2 * self.rtol

    rtol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_primary, takes_self=True),
        kw_only=True,
    )

    def _default_atol_primary(self) -> Scalar:
        return 1e-2 * self.atol

    atol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_primary, takes_self=True),
        kw_only=True,
    )

    @override
    def init[Params](
        self, system: LinearSystem[Params], params: Params
    ) -> tuple[State, Stats]:
        params_flat: Free = system.transform.backward_tangents(params)
        state: CupySolver.State = self.State(params=params_flat)
        stats: CupySolver.Stats = self.Stats()
        return state, stats

    @override
    def compute[Params](
        self, system: LinearSystem[Params], state: State, stats: Stats
    ) -> tuple[State, Stats, Result]:
        import cupy as cp

        # cb_wrapper: Callable = self._make_callback(callback, state, stats)
        lop: linalg.LinearOperator = _as_lop(system)
        options: dict[str, Any] = self._options(system)
        x: FreeCp
        info: int
        x, info = self._wrapped(
            lop,
            cp.from_dlpack(system.b_flat),
            cp.from_dlpack(state.params),
            # callback=cb_wrapper,
            **options,
        )
        state.params = jnp.from_dlpack(x)
        # stats.n_steps = len(grapes.get_timer(cb_wrapper))
        result: Result
        stats, result = self._finalize(system, state, stats, info)
        return state, stats, result

    # def _make_callback(
    #     self, callback: Callback[State, Stats] | None, state: State, stats: Stats
    # ) -> Callable:
    #     @grapes.timer(label=f"{self.name}.callback()")
    #     def wrapper(xk: FreeCp) -> None:
    #         if callback is None:
    #             return
    #         state.params_flat = jnp.from_dlpack(xk)
    #         stats.n_steps = len(grapes.get_timer(wrapper)) + 1
    #         callback(state, stats)

    #     return wrapper

    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = {"tol": self.rtol_primary, "atol": self.atol_primary}
        if self.max_steps is not None:
            options["maxiter"] = self.max_steps
        if system.preconditioner is not None:
            options["M"] = _preconditioner(system)
        return options

    def _finalize(
        self, system: LinearSystem, state: State, stats: Stats, info: int
    ) -> tuple[Stats, Result]:
        assert system.matvec is not None
        abs_residual: Float[Array, ""] = utils.absolute_residual(
            system.matvec, state.params, system.b_flat
        )
        b_norm: Float[Array, ""] = jnp.linalg.norm(system.b_flat)
        stats.info = info
        stats.relative_residual = utils.safe_divide(abs_residual, b_norm)
        if info == 0:
            # TODO: info from CuPy may not be reliable, we should do something better here
            return stats, Result.SUCCESS
        if info < 0:
            return stats, Result.BREAKDOWN
        stats.n_steps = info
        return stats, Result.MAX_STEPS_REACHED

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        raise NotImplementedError


def _as_lop[Params](system: LinearSystem[Params]) -> linalg.LinearOperator:
    import cupy as cp
    from cupyx.scipy.sparse import linalg

    assert system.matvec is not None

    def matvec(x: FreeCp) -> FreeCp:
        assert system.matvec is not None
        x_jax: Free = jnp.from_dlpack(x)
        y_jax: Free = system.matvec(x_jax)
        return cp.from_dlpack(y_jax)

    def rmatvec(x: FreeCp) -> FreeCp:
        assert system.rmatvec is not None
        x_jax: Free = jnp.from_dlpack(x)
        y_jax: Free = system.rmatvec(x_jax)
        return cp.from_dlpack(y_jax)

    dim: int
    (dim,) = system.b_flat.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        rmatvec=None if system.rmatvec is None else rmatvec,
        dtype=system.b_flat.dtype,
    )


def _preconditioner[Params](
    system: LinearSystem[Params],
) -> linalg.LinearOperator | None:
    import cupy as cp
    from cupyx.scipy.sparse import linalg

    if system.preconditioner is None:
        return None

    def matvec(x: FreeCp) -> FreeCp:
        assert system.preconditioner is not None
        x_jax: Free = jnp.from_dlpack(x)
        y_jax: Free = system.preconditioner(x_jax)
        return cp.from_dlpack(y_jax)

    def rmatvec(x: FreeCp) -> FreeCp:
        assert system.rpreconditioner is not None
        x_jax: Free = jnp.from_dlpack(x)
        y_jax: Free = system.rpreconditioner(x_jax)
        return cp.from_dlpack(y_jax)

    dim: int
    (dim,) = system.b_flat.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        rmatvec=None if system.rpreconditioner is None else rmatvec,
        dtype=system.b_flat.dtype,
    )
