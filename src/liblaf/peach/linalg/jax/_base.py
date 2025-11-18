import abc
from collections.abc import Iterable
from typing import Any, override

import jax.numpy as jnp
from jaxtyping import Array, Float

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

from ._types import JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " free"]


@tree.define
class JaxSolver(LinearSolver[JaxState, JaxStats]):
    from ._types import JaxState as State
    from ._types import JaxStats as Stats

    atol: float = 0.0
    max_steps: int | None = None
    rtol: float = 1e-5

    @override
    def setup(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[JaxState, JaxStats]:
        params_flat: Vector
        system, params_flat, constraints = system.flatten(
            params, constraints=constraints
        )
        state = JaxState(params_flat=params_flat, unflatten=system.unflatten)
        if self.jit:
            system = system.jit()
        if self.timer:
            system = system.timer()
        return SetupResult(system, constraints, state, JaxStats())

    @override
    def solve(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback[JaxState, JaxStats] | None = None,
    ) -> LinearSolution[JaxState, JaxStats]:
        state: JaxState
        stats: JaxStats
        system, constraints, state, stats = self.setup(
            system, params, constraints=constraints
        )
        if constraints:
            raise NotImplementedError
        if callback is not None:
            raise NotImplementedError
        assert system.matvec is not None
        state.params_flat, _info = self._wrapped(
            system.matvec, system.b_flat, state.params_flat, **self._options(system)
        )
        residual: Vector = system.matvec(state.params_flat) - system.b_flat
        residual_norm: Scalar = jnp.linalg.norm(residual)
        b_norm: Scalar = jnp.linalg.norm(system.b_flat)
        stats.residual_relative = residual_norm / b_norm
        result: Result
        if residual_norm <= self.atol + self.rtol * b_norm:
            result = Result.SUCCESS
        else:
            result = Result.UNKNOWN_ERROR
        return self.finalize(system, state, stats, result)

    def _options(self, system: LinearSystem) -> dict[str, Any]:
        max_steps: int = (
            system.b_flat.size if self.max_steps is None else self.max_steps
        )
        return {
            "tol": self.rtol,
            "atol": self.atol,
            "maxiter": max_steps,
            "M": system.preconditioner,
        }

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Any]:
        raise NotImplementedError
