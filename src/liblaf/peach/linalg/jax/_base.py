import abc
from collections.abc import Iterable
from typing import Any, override

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg import utils
from liblaf.peach.linalg.abc import Callback, LinearSolution, LinearSolver, Result
from liblaf.peach.linalg.system import LinearSystem

from ._types import JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " free"]


@tree.define
class JaxSolver(LinearSolver[JaxState, JaxStats]):
    from ._types import JaxState as State
    from ._types import JaxStats as Stats

    Solution = LinearSolution[JaxState, JaxStats]

    max_steps: int | None = None

    atol: Scalar = tree.array(
        default=0.0, converter=tree.converters.asarray, kw_only=True
    )
    rtol: Scalar = tree.array(
        default=1e-3, converter=tree.converters.asarray, kw_only=True
    )

    def _default_atol_primary(self) -> Scalar:
        return 1e-2 * self.atol

    atol_primary: Scalar = tree.array(
        default=attrs.Factory(_default_atol_primary, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

    def _default_rtol_primary(self) -> Scalar:
        return 1e-2 * self.rtol

    rtol_primary: Scalar = tree.array(
        default=attrs.Factory(_default_rtol_primary, takes_self=True),
        converter=tree.converters.asarray,
        kw_only=True,
    )

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
        if callback is not None:
            raise NotImplementedError
        assert system.matvec is not None
        state.params_flat, stats.info = self._wrapped(
            system.matvec, system.b_flat, state.params_flat, **self._options(system)
        )
        residual: Vector = system.matvec(state.params_flat) - system.b_flat
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
