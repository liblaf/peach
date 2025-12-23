from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, override

import attrs
import cupy
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf import grapes
from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg import utils
from liblaf.peach.linalg.abc import (
    Callback,
    LinearSolution,
    LinearSolver,
    Params,
    Result,
    SetupResult,
)
from liblaf.peach.linalg.system import LinearSystem

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import linalg


type Free = Float[Array, " free"]
type FreeCp = Float[cp.ndarray, " free"]
type Scalar = Float[Array, ""]


@tree.define
class CupySolver(LinearSolver):
    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    Solution = LinearSolution[State, Stats]

    max_steps: Integer[Array, ""] | None = tree.field(
        default=None, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=1e-5, converter=tree.converters.asarray, kw_only=True
    )
    atol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )

    def _default_real_rtol(self) -> Scalar:
        return self.rtol / 10.0

    real_rtol: Scalar = tree.array(
        default=attrs.Factory(_default_real_rtol, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

    def _default_real_atol(self) -> Scalar:
        return self.atol / 10.0

    real_atol: Scalar = tree.array(
        default=attrs.Factory(_default_real_atol, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

    @override
    def setup(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[State, Stats]:
        params_flat: Free
        system, params_flat, constraints = system.flatten(
            params, constraints=constraints
        )
        state: CupySolver.State = self.State(
            params_flat=params_flat, structure=system.structure
        )
        if self.jit:
            system = system.jit()
        if self.timer:
            system = system.timer()
        return SetupResult(system, constraints, state, self.Stats())

    @override
    def _solve(
        self,
        system: LinearSystem,
        state: State,
        stats: Stats,
        *,
        callback: Callback[State, Stats] | None = None,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[State, Stats, Result]:
        if constraints:
            raise NotImplementedError
        cb_wrapper: Callable = self._make_callback(callback, state, stats)
        lop: linalg.LinearOperator = _as_lop(system)
        x: FreeCp
        info: int
        x, info = self._wrapped(
            lop,
            cupy.from_dlpack(system.b_flat),
            cupy.from_dlpack(state.params_flat),
            callback=cb_wrapper,
            **self._options(system),
        )
        state.params_flat = jnp.from_dlpack(x)
        stats.n_steps = len(grapes.get_timer(cb_wrapper))
        result: Result
        stats, result = self._finalize(system, state, stats, info)
        return state, stats, result

    def _make_callback(
        self, callback: Callback[State, Stats] | None, state: State, stats: Stats
    ) -> Callable:
        @grapes.timer(label=f"{self.name}.callback()")
        def wrapper(xk: FreeCp) -> None:
            if callback is None:
                return
            state.params_flat = jnp.from_dlpack(xk)
            stats.n_steps = len(grapes.get_timer(wrapper)) + 1
            callback(state, stats)

        return wrapper

    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = {"tol": self.real_rtol, "atol": self.real_atol}
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
            system.matvec, state.params_flat, system.b_flat
        )
        b_norm: Float[Array, ""] = jnp.linalg.norm(system.b_flat)
        stats.info = info
        stats.relative_residual = abs_residual / b_norm
        if info == 0:
            # info from CuPy is not reliable, so we double check convergence here
            result: Result = (
                Result.SUCCESS
                if abs_residual <= self.atol + self.rtol * b_norm
                else Result.UNKNOWN_ERROR
            )
            return stats, result
        if info < 0:
            return stats, Result.BREAKDOWN
        stats.n_steps = info
        return stats, Result.MAX_STEPS_REACHED

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        raise NotImplementedError


def _as_lop(system: LinearSystem) -> linalg.LinearOperator:
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


def _preconditioner(system: LinearSystem) -> linalg.LinearOperator | None:
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
