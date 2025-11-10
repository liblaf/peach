import abc

from liblaf import grapes
from liblaf.peach import tree_utils
from liblaf.peach.optim.objective import Objective

from ._types import Callback, OptimizeSolution, Params, Result, State, Stats


@tree_utils.tree
class Optimizer[StateT: State, StatsT: Stats](abc.ABC):
    jit: bool = False
    max_steps: int = 256
    timer: bool = False

    @abc.abstractmethod
    def init(
        self, objective: Objective, params: Params
    ) -> tuple[Objective, StateT, StatsT]: ...

    @abc.abstractmethod
    def step(
        self, objective: Objective, params: Params, state: StateT
    ) -> tuple[Params, StateT]: ...

    @abc.abstractmethod
    def update_stats(
        self, objective: Objective, params: Params, state: StateT, stats: StatsT
    ) -> StatsT: ...

    @abc.abstractmethod
    def terminate(
        self, objective: Objective, params: Params, state: StateT, stats: StatsT
    ) -> tuple[bool, Result]: ...

    @abc.abstractmethod
    def postprocess(
        self,
        objective: Objective,
        params: Params,
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> OptimizeSolution:
        solution: OptimizeSolution = OptimizeSolution(
            result=result, params=params, state=state, stats=stats
        )
        return solution

    def minimize(
        self,
        objective: Objective,
        params: Params,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> OptimizeSolution[StateT, StatsT]:
        with grapes.timer(label=str(self)) as timer:
            state: StateT
            stats: StatsT
            objective, state, stats = self.init(objective, params)
            done: bool = False
            n_steps: int = 0
            result: Result = Result.UNKNOWN_ERROR
            while n_steps < self.max_steps and not done:
                params, state = self.step(objective, params, state)
                n_steps += 1
                stats.n_steps = n_steps
                stats.time = timer.elapsed()
                stats = self.update_stats(objective, params, state, stats)
                if callback is not None:
                    callback(state, stats)
                done, result = self.terminate(objective, params, state, stats)
            if not done:
                result = Result.MAX_STEPS_REACHED
            solution: OptimizeSolution[StateT, StatsT] = self.postprocess(
                objective, params, state, stats, result
            )
        solution.stats.time = timer.elapsed()
        return solution
