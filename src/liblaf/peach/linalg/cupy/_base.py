from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import grapes
from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
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


@tree.define
class CupySolver(LinearSolver):
    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    Solution = LinearSolution[State, Stats]

    rtol: float = tree.field(default=1e-5, kw_only=True)
    atol: float = tree.field(default=0.0, kw_only=True)
    max_steps: int | None = tree.field(default=None, kw_only=True)

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
            params_flat=params_flat, flat_def=system.flat_def
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
            system.b_flat,
            state.params_flat,
            callback=cb_wrapper,
            **self._options(system),
        )
        state.params_flat = jnp.from_dlpack(x)
        stats.n_steps = len(grapes.get_timer(cb_wrapper))
        result: Result
        stats, result = self._finalize(info, stats)
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
        options: dict[str, Any] = {"tol": self.rtol, "atol": self.atol}
        if self.max_steps is not None:
            options["maxiter"] = self.max_steps
        if system.preconditioner is not None:
            options["M"] = _preconditioner(system)
        return options

    def _finalize(self, info: int, stats: Stats) -> tuple[Stats, Result]:
        stats.info = info
        if info == 0:
            return stats, Result.SUCCESS
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
