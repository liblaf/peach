import time
from collections.abc import Iterable
from typing import override

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

from ._base import JaxSolver
from ._cg import JaxCG
from ._gmres import JaxGMRES
from ._types import JaxCompositeStats, JaxState, JaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class JaxCompositeSolver(LinearSolver[JaxState, JaxCompositeStats]):
    from ._types import JaxCompositeStats as Stats
    from ._types import JaxState as State

    solvers: list[JaxSolver] = tree.field(factory=lambda: [JaxCG(), JaxGMRES()])

    @override
    def setup(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[JaxState, JaxCompositeStats]:
        state: JaxState
        system, constraints, state, _ = self.solvers[0].setup(
            system, params, constraints=constraints
        )
        return SetupResult(system, constraints, state, JaxCompositeStats())

    @override
    def solve(
        self,
        system: LinearSystem,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback[JaxState, JaxCompositeStats] | None = None,
    ) -> LinearSolution[JaxState, JaxCompositeStats]:
        state: JaxState
        stats: JaxCompositeStats
        system, constraints, state, stats = self.setup(
            system, params, constraints=constraints
        )
        if constraints:
            raise NotImplementedError
        if callback is not None:
            raise NotImplementedError
        assert system.matvec is not None
        params_flat: Vector = None  # pyright: ignore[reportAssignmentType]
        result: Result = Result.UNKNOWN_ERROR
        for solver in self.solvers:
            sub_stats = JaxStats()
            params_flat, _info = solver._wrapped(  # noqa: SLF001
                system.matvec,
                system.b_flat,
                state.params_flat,
                **solver._options(system),  # noqa: SLF001
            )
            residual: Vector = system.matvec(params_flat) - system.b_flat
            residual_norm: Scalar = jnp.linalg.norm(residual)
            b_norm: Scalar = jnp.linalg.norm(system.b_flat)
            sub_stats.residual_relative = residual_norm / b_norm
            sub_stats.end_time = time.perf_counter()
            stats.stats.append(sub_stats)
            if residual_norm <= solver.atol + solver.rtol * b_norm:
                result = Result.SUCCESS
                break
        state.params_flat = params_flat
        return self.finalize(system, state, stats, result)
