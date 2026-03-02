import abc
from typing import Any, override

import attrs
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from liblaf.peach.linalg import utils
from liblaf.peach.linalg.base import LinearSolution, LinearSolver, LinearSystem, Result

from ._types import JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " free"]


@jarp.define
class JaxSolver(LinearSolver[JaxState, JaxStats]):
    from ._types import JaxState as State
    from ._types import JaxStats as Stats

    Solution = LinearSolution[JaxState, JaxStats]

    max_steps: int | None = None

    atol: Scalar = jarp.array(default=0.0, kw_only=True)
    rtol: Scalar = jarp.array(default=1e-3, kw_only=True)

    def _default_atol_primary(self) -> Scalar:
        return jnp.asarray(1e-2 * self.atol)

    atol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_atol_primary, takes_self=True),
        kw_only=True,
    )

    def _default_rtol_primary(self) -> Scalar:
        return jnp.asarray(1e-2 * self.rtol)

    rtol_primary: Scalar = jarp.field(
        default=attrs.Factory(_default_rtol_primary, takes_self=True), kw_only=True
    )

    @override
    def init[Params](
        self, system: LinearSystem[Params], params: PyTree
    ) -> tuple[JaxState, JaxStats]:
        params_flat: Vector = system.transform.backward_primals(params)
        state: JaxState = self.State(params=params_flat)
        stats: JaxStats = self.Stats()
        return state, stats

    @override
    def compute[Params](
        self, system: LinearSystem[Params], state: State, stats: Stats
    ) -> tuple[State, Stats, Result]:
        assert system.matvec is not None
        state.params, stats.info = self._wrapped(
            system.matvec, system.b_flat, state.params, **self._options(system)
        )
        residual: Vector = system.matvec(state.params) - system.b_flat
        residual_norm: Scalar = jnp.linalg.norm(residual)
        b_norm: Scalar = jnp.linalg.norm(system.b_flat)
        stats.residual_relative = utils.safe_divide(residual_norm, b_norm)
        result: Result
        if residual_norm <= self.atol + self.rtol * b_norm:
            result = Result.SUCCESS
        else:
            result = Result.UNKNOWN_ERROR
        return state, stats, result

    def _options(self, system: LinearSystem) -> dict[str, Any]:
        max_steps: int = (
            system.b_flat.size if self.max_steps is None else self.max_steps
        )
        return {
            "tol": self.rtol_primary,
            "atol": self.atol_primary,
            "maxiter": max_steps,
            "M": system.preconditioner,
        }

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Any]:
        raise NotImplementedError
