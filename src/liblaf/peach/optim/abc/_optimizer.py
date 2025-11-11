import abc
import time

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._types import Callback, OptimizeSolution, Params, Result, State, Stats


@tree.define
class Optimizer[StateT: State, StatsT: Stats](abc.ABC):
    max_steps: int = 256
    jit: bool = tree.field(default=False, kw_only=True)
    timer: bool = tree.field(default=False, kw_only=True)

    @abc.abstractmethod
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
    ) -> tuple[Objective, StateT, StatsT]: ...

    @abc.abstractmethod
    def step(self, objective: Objective, state: StateT) -> StateT: ...

    def update_stats(
        self,
        objective: Objective,  # noqa: ARG002
        state: StateT,  # noqa: ARG002
        stats: StatsT,
    ) -> StatsT:
        return stats

    @abc.abstractmethod
    def terminate(
        self, objective: Objective, state: StateT, stats: StatsT
    ) -> tuple[bool, Result]: ...

    def postprocess(
        self,
        objective: Objective,  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> OptimizeSolution[StateT, StatsT]:
        stats.end_time = time.perf_counter()
        solution: OptimizeSolution[StateT, StatsT] = OptimizeSolution(
            result=result, state=state, stats=stats
        )
        return solution

    def minimize(
        self,
        objective: Objective,
        params: Params,
        *,
        fixed_mask: Params | None = None,
        n_fixed: int | None = None,
        lower_bound: Params | None = None,
        upper_bound: Params | None = None,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> OptimizeSolution[StateT, StatsT]:
        state: StateT
        stats: StatsT
        objective, state, stats = self.init(
            objective,
            params,
            fixed_mask=fixed_mask,
            n_fixed=n_fixed,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        done: bool = False
        n_steps: int = 0
        result: Result = Result.UNKNOWN_ERROR
        while n_steps < self.max_steps and not done:
            state = self.step(objective, state)
            n_steps += 1
            stats.n_steps = n_steps
            stats = self.update_stats(objective, state, stats)
            if callback is not None:
                callback(state, stats)
            done, result = self.terminate(objective, state, stats)
        if not done:
            result = Result.MAX_STEPS_REACHED
        solution: OptimizeSolution[StateT, StatsT] = self.postprocess(
            objective, state, stats, result
        )
        return solution
