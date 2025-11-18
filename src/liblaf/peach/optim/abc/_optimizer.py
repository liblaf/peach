import abc
import time
from collections.abc import Iterable
from typing import NamedTuple

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim.objective import Objective

from ._types import Callback, OptimizeSolution, Params, Result, State, Stats


class SetupResult[StateT: State, StatsT: Stats](NamedTuple):
    objective: Objective
    constraints: list[Constraint]
    state: StateT
    stats: StatsT


@tree.define
class Optimizer[StateT: State, StatsT: Stats](abc.ABC):
    from ._types import State, Stats

    max_steps: int = tree.field(default=256, kw_only=True)
    jit: bool = tree.field(default=False, kw_only=True)
    timer: bool = tree.field(default=False, kw_only=True)

    @abc.abstractmethod
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[StateT, StatsT]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        objective: Objective,
        state: StateT,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> StateT:
        raise NotImplementedError

    def update_stats(
        self,
        objective: Objective,  # noqa: ARG002
        state: StateT,  # noqa: ARG002
        stats: StatsT,
        *,
        constraints: Iterable[Constraint] = (),  # noqa: ARG002
    ) -> StatsT:
        return stats

    @abc.abstractmethod
    def terminate(
        self,
        objective: Objective,
        state: StateT,
        stats: StatsT,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[bool, Result]:
        raise NotImplementedError

    def postprocess(
        self,
        objective: Objective,  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
        *,
        constraints: Iterable[Constraint] = (),  # noqa: ARG002
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
        constraints: Iterable[Constraint] = (),
        callback: Callback[StateT, StatsT] | None = None,
    ) -> OptimizeSolution[StateT, StatsT]:
        state: StateT
        stats: StatsT
        objective, constraints, state, stats = self.init(
            objective, params, constraints=constraints
        )
        done: bool = False
        n_steps: int = 0
        result: Result = Result.UNKNOWN_ERROR
        while n_steps < self.max_steps and not done:
            state = self.step(objective, state, constraints=constraints)
            n_steps += 1
            stats.n_steps = n_steps
            stats = self.update_stats(objective, state, stats, constraints=constraints)
            if callback is not None:
                callback(state, stats)
            done, result = self.terminate(
                objective, state, stats, constraints=constraints
            )
        if not done:
            result = Result.MAX_STEPS_REACHED
        solution: OptimizeSolution[StateT, StatsT] = self.postprocess(
            objective, state, stats, result, constraints=constraints
        )
        return solution
