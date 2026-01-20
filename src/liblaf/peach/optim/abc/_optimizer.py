import abc
import functools
import time
from collections.abc import Iterable

from jaxtyping import Array, Float
from liblaf.grapes.logging import autolog

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.functools import Objective, ObjectiveProtocol

from ._types import Callback, OptimizeSolution, Result, State, Stats

type Vector = Float[Array, " N"]


@tree.define
class Optimizer[StateT: State, StatsT: Stats](abc.ABC):
    from ._types import OptimizeSolution as Solution
    from ._types import State, Stats

    max_steps: int = tree.field(default=None, kw_only=True)

    @functools.cached_property
    def name(self) -> str:
        cls: type = type(self)
        return getattr(cls, "__qualname__", None) or getattr(
            cls, "__name__", "Optimizer"
        )

    def init(
        self,
        objective: Objective,  # noqa: ARG002
        params: Vector,
        *,
        constraints: Iterable[Constraint] = (),  # noqa: ARG002
    ) -> tuple[StateT, StatsT]:
        state = self.State(params=params)
        stats = self.Stats()
        return state, stats  # pyright: ignore[reportReturnType]

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
        objective: ObjectiveProtocol,  # noqa: ARG002
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
        objective: ObjectiveProtocol,  # noqa: ARG002
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
        params: Vector,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback[StateT, StatsT] | None = None,
    ) -> OptimizeSolution[StateT, StatsT]:
        max_steps: int = self.max_steps or self._default_max_steps(
            objective, params, constraints=constraints
        )
        state: StateT
        stats: StatsT
        state, stats = self.init(objective, params, constraints=constraints)
        done: bool = False
        n_steps: int = 0
        result: Result = Result.UNKNOWN_ERROR
        while n_steps < max_steps and not done:
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

    def _default_max_steps(
        self,
        objective: Objective,  # noqa: ARG002
        params: Vector,
        *,
        constraints: Iterable[Constraint] = (),  # noqa: ARG002
    ) -> int:
        return params.size

    def _warn_unsupported_constraints(self, constraints: Iterable[Constraint]) -> None:
        _logging_hide = True
        if constraints:
            autolog.warning(
                "'%s' does not support the following constraints: %r",
                self.name,
                constraints,
            )
