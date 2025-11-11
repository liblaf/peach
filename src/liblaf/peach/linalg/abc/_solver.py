import abc
import time

from liblaf.peach import tree
from liblaf.peach.linalg.op import LinearOperator

from ._types import Callback, LinearSolution, Params, Result, State, Stats


@tree.define
class LinearSolver[StateT: State, StatsT: Stats](abc.ABC):
    jit: bool = False
    timer: bool = False

    @abc.abstractmethod
    def setup(
        self,
        op: LinearOperator,
        b: Params,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
    ) -> tuple[LinearOperator, StateT, StatsT]:
        raise NotImplementedError

    def finalize(
        self,
        op: LinearOperator,  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> LinearSolution[StateT, StatsT]:
        stats.end_time = time.perf_counter()
        return LinearSolution(state=state, stats=stats, result=result)

    @abc.abstractmethod
    def solve(
        self,
        op: LinearOperator,
        b: Params,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> LinearSolution[StateT, StatsT]:
        raise NotImplementedError
