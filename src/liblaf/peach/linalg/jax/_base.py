import abc
from typing import Any, override

import attrs
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from liblaf.peach.linalg import utils
from liblaf.peach.linalg.base import LinearSolution, LinearSolver, Result

from ._types import JaxLinearSystem, JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " free"]


@jarp.define
class JaxSolver(LinearSolver[JaxLinearSystem, JaxState, JaxStats]):
    from ._types import JaxState as State
    from ._types import JaxStats as Stats

    type Solution = LinearSolution[JaxState, JaxStats]

    maxiter: int | None = None

    atol: Scalar = jarp.array(default=0.0, kw_only=True)
    rtol: Scalar = jarp.array(default=1e-4, kw_only=True)

    def _default_atol_primary(self) -> Scalar:
        return jnp.asarray(1e-1 * self.atol)

    atol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_primary, takes_self=True),
        kw_only=True,
    )

    def _default_rtol_primary(self) -> Scalar:
        return jnp.asarray(1e-1 * self.rtol)

    rtol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_primary, takes_self=True), kw_only=True
    )

    @override
    def init(
        self, system: JaxLinearSystem, params: PyTree
    ) -> tuple[JaxState, JaxStats]:
        state: JaxState = self.State(params=params)
        stats: JaxStats = self.Stats()
        return state, stats

    @override
    def compute(
        self, system: JaxLinearSystem, state: JaxState, stats: JaxStats
    ) -> tuple[JaxState, JaxStats, Result]:
        state.params, stats.info = self._wrapped(
            system.matvec, system.b, state.params, **self._options(system)
        )
        residual: Vector = system.matvec(state.params) - system.b
        residual_norm: Scalar = jnp.linalg.norm(residual)
        b_norm: Scalar = jnp.linalg.norm(system.b)
        stats.residual_relative = utils.safe_divide(residual_norm, b_norm)
        result: Result
        if residual_norm <= self.atol + self.rtol * b_norm:
            result = Result.SUCCESS
        else:
            result = Result.UNKNOWN_ERROR
        return state, stats, result

    def _options(self, system: JaxLinearSystem) -> dict[str, Any]:
        maxiter: int = system.b.size if self.maxiter is None else self.maxiter
        return {
            "tol": self.rtol_primary,
            "atol": self.atol_primary,
            "maxiter": maxiter,
            "M": getattr(system, "preconditioner", None),
        }

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Any]:
        raise NotImplementedError
