from collections.abc import Iterable
from typing import override

from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg.abc import Callback, LinearSolution, LinearSolver, Result
from liblaf.peach.linalg.system import LinearSystem

type Vector = Float[Array, " free"]


def _default_solvers() -> list[LinearSolver]:
    from liblaf.peach import cuda

    if cuda.is_available():
        from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

        return [CupyCG(), CupyMinRes()]

    from liblaf.peach.linalg.scipy import ScipyCG, ScipyMinRes

    return [ScipyCG(), ScipyMinRes()]


@tree.define
class CompositeSolver(LinearSolver):
    from ._composite_types import CompositeState as State
    from ._composite_types import CompositeStats as Stats

    Solution = LinearSolution[State, Stats]

    solvers: list[LinearSolver] = tree.field(factory=_default_solvers)

    @override
    def _solve(
        self,
        system: LinearSystem,
        state: State,
        stats: Stats,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback[State, Stats] | None = None,
    ) -> tuple[State, Stats, Result]:
        solution: LinearSolution = None  # pyright: ignore[reportAssignmentType]
        for solver in self.solvers:
            solution = solver.solve(
                system, state.params, callback=callback, constraints=constraints
            )
            state.state.append(solution.state)
            stats.stats.append(solution.stats)
            if solution.success:
                break
        state.params_flat = solution.params_flat
        return state, stats, solution.result
