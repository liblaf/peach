from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, cast, override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import BaseProblem, LinearSolver, Problem, Result
from liblaf.peach.utils import implemented

from ._types import CupyState, CupyStats

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import linalg


type Vector = Float[Array, " N"]
type VectorCupy = Float[cp.ndarray, " N"]


@jarp.define(kw_only=True)
class CupySolver(LinearSolver[CupyState, CupyStats]):
    """Base class for linear solvers backed by CuPy sparse routines."""

    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    type Solution = LinearSolver.Solution[State, Stats]

    maxiter: int | None = None

    @override
    def init(self, problem: BaseProblem, params: Vector) -> State:
        """Initialize a CuPy solver state."""
        return self.State(params=params)

    @override
    def compute(self, problem: BaseProblem, state: State) -> tuple[State, Result]:
        """Run the wrapped CuPy solver through DLPack array transfers."""
        import cupy as cp

        problem: Problem = cast("Problem", problem)
        lop: linalg.LinearOperator = _as_lop(problem)
        options: dict[str, Any] = self._options(problem)
        x, info = self._wrapped(
            lop, cp.from_dlpack(problem.b), cp.from_dlpack(state.params), **options
        )
        state.params = jnp.from_dlpack(x)
        state, result = self._finalize(problem, state, info)
        return state, result

    def _options(self, problem: Problem) -> dict[str, Any]:
        """Build keyword options for the wrapped CuPy solver."""
        return {"maxiter": self.maxiter, "M": _preconditioner(problem)}

    def _finalize(
        self, problem: Problem, state: State, info: int
    ) -> tuple[State, Result]:
        """Translate CuPy solver status and residuals into Peach state."""
        state.info = info
        state.absolute_residual = jnp.linalg.norm(
            problem.matvec(state.params) - problem.b
        )
        state.relative_residual = state.absolute_residual / jnp.linalg.norm(problem.b)
        if info == 0:
            return state, Result.SUCCESS
        if info < 0:
            return state, Result.BREAKDOWN
        state.n_steps = info
        return state, Result.MAX_STEPS_REACHED

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        """Call the concrete CuPy solver implementation."""
        raise NotImplementedError


def _as_lop(problem: Problem) -> linalg.LinearOperator:
    """Build a CuPy `LinearOperator` from a Peach linear problem."""
    import cupy as cp
    from cupyx.scipy.sparse import linalg

    def matvec(x: VectorCupy) -> VectorCupy:
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = problem.matvec(x_jax)
        return cp.from_dlpack(y_jax, copy=True)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = problem.rmatvec(x_jax)
        return cp.from_dlpack(y_jax, copy=True)

    dim: int
    (dim,) = problem.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # ty:ignore[unknown-argument]
        rmatvec=rmatvec if implemented(problem, Problem.rmatvec) else None,  # ty:ignore[unknown-argument]
        dtype=problem.b.dtype,
    )


def _preconditioner(problem: Problem) -> linalg.LinearOperator | None:
    """Build a CuPy preconditioner operator when the problem implements one."""
    if not implemented(problem, Problem.precondition):
        return None

    import cupy as cp
    from cupyx.scipy.sparse import linalg

    def matvec(x: VectorCupy) -> VectorCupy:
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = problem.precondition(x_jax)
        return cp.from_dlpack(y_jax, copy=True)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        x_jax: Vector = jnp.from_dlpack(x)
        y_jax: Vector = problem.rprecondition(x_jax)
        return cp.from_dlpack(y_jax, copy=True)

    dim: int
    (dim,) = problem.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # ty:ignore[unknown-argument]
        rmatvec=rmatvec if implemented(problem, Problem.rprecondition) else None,  # ty:ignore[unknown-argument]
        dtype=problem.b.dtype,
    )
