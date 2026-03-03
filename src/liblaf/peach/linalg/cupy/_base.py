from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach.linalg import utils
from liblaf.peach.linalg.base import (
    LinearSolution,
    LinearSolver,
    Result,
    SupportsPreconditioner,
    SupportsRmatvec,
    SupportsRpreconditioner,
)

from ._types import CupyLinearSystem, CupyState, CupyStats

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import linalg


type Vector = Float[Array, " N"]
type VectorCupy = Float[cp.ndarray, " N"]


@jarp.define
class CupySolver(LinearSolver[CupyLinearSystem, CupyState, CupyStats]):
    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    type Solution = LinearSolution[State, Stats]

    maxiter: int | None = None

    @override
    def init(self, system: CupyLinearSystem, params: Vector) -> tuple[State, Stats]:
        state: CupySolver.State = self.State(params=params)
        stats: CupySolver.Stats = self.Stats()
        return state, stats

    @override
    def compute(
        self, system: CupyLinearSystem, state: State, stats: Stats
    ) -> tuple[State, Stats, Result]:
        import cupy as cp

        lop: linalg.LinearOperator = _as_lop(system)
        options: dict[str, Any] = self._options(system)
        x: VectorCupy
        info: int
        x, info = self._wrapped(
            lop,
            cp.from_dlpack(system.b),
            cp.from_dlpack(state.params),
            **options,
        )
        state.params = jnp.from_dlpack(x)
        result: Result
        stats, result = self._finalize(system, state, stats, info)
        return state, stats, result

    def _options(self, system: CupyLinearSystem) -> dict[str, Any]:
        return {"maxiter": self.maxiter, "M": _preconditioner(system)}

    def _finalize(
        self, system: CupyLinearSystem, state: State, stats: Stats, info: int
    ) -> tuple[Stats, Result]:
        stats.info = info
        stats.relative_residual = utils.relative_residual(
            system.matvec, state.params, system.b
        )
        if info == 0:
            # TODO: info from CuPy may not be reliable, we should do something better here
            return stats, Result.SUCCESS
        if info < 0:
            return stats, Result.BREAKDOWN
        stats.n_steps = info
        return stats, Result.MAX_STEPS_REACHED

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        raise NotImplementedError


def _as_lop(system: CupyLinearSystem) -> linalg.LinearOperator:
    import cupy as cp
    from cupyx.scipy.sparse import linalg

    def matvec(x: VectorCupy) -> VectorCupy:
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = system.matvec(x_jax)
        return cp.from_dlpack(y_jax)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        if TYPE_CHECKING:
            assert isinstance(system, SupportsRmatvec)
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = system.rmatvec(x_jax)
        return cp.from_dlpack(y_jax)

    dim: int
    (dim,) = system.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # pyright: ignore[reportCallIssue]
        rmatvec=None if getattr(system, "rmatvec", None) is None else rmatvec,  # pyright: ignore[reportCallIssue]
        dtype=system.b.dtype,
    )


def _preconditioner(system: CupyLinearSystem) -> linalg.LinearOperator | None:
    import cupy as cp
    from cupyx.scipy.sparse import linalg

    if getattr(system, "preconditioner", None) is None:
        return None

    def matvec(x: VectorCupy) -> VectorCupy:
        if TYPE_CHECKING:
            assert isinstance(system, SupportsPreconditioner)
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = system.preconditioner(x_jax)
        return cp.from_dlpack(y_jax)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        if TYPE_CHECKING:
            assert isinstance(system, SupportsRpreconditioner)
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = system.rpreconditioner(x_jax)
        return cp.from_dlpack(y_jax)

    dim: int
    (dim,) = system.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # pyright: ignore[reportCallIssue]
        rmatvec=None if getattr(system, "rpreconditioner", None) is None else rmatvec,  # pyright: ignore[reportCallIssue]
        dtype=system.b.dtype,
    )
