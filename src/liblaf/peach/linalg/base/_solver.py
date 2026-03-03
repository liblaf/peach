import time

import jarp
from jaxtyping import Array, Float

from ._system import LinearSystem
from ._types import LinearSolution, Result, State, Stats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class LinearSolver[P: LinearSystem, S: State, T: Stats]:
    from ._types import LinearSolution as Solution
    from ._types import State, Stats

    def init(self, system: P, params: Vector) -> tuple[S, T]:
        raise NotImplementedError

    def compute(self, system: P, state: S, stats: T) -> tuple[S, T, Result]:
        raise NotImplementedError

    def postprocess(
        self,
        system: P,  # noqa: ARG002
        state: S,
        stats: T,
        result: Result,
    ) -> Solution[S, T]:
        stats._end_time = time.perf_counter()  # noqa: SLF001
        return LinearSolution(result=result, state=state, stats=stats)

    def solve(self, system: P, params: Vector) -> Solution[S, T]:
        state: S
        stats: T
        state, stats = self.init(system, params)
        result: Result
        state, stats, result = self.compute(system, state, stats)
        return self.postprocess(system, state, stats, result)
