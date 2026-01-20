import abc
import functools
import time
from collections.abc import Iterable
from typing import ClassVar

from jaxtyping import Array, Float
from liblaf.grapes.logging import autolog

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint, Constraints
from liblaf.peach.functools import Objective
from liblaf.peach.transforms import FlattenTransform, LinearTransform

from ._types import Callback, OptimizeSolution, Problem, Result, State, Stats

type Vector = Float[Array, " N"]


@tree.define
class Optimizer[StateT: State, StatsT: Stats](abc.ABC):
    from ._types import State, Stats

    Solution = OptimizeSolution[State, Stats]

    supported_constraints: ClassVar[tuple[type[Constraint], ...]] = ()

    max_steps: int | None = tree.field(default=None, kw_only=True)

    @functools.cached_property
    def name(self) -> str:
        cls: type = type(self)
        return cls.__qualname__

    @abc.abstractmethod
    def init[T](
        self,
        objective: Objective,
        params: T,
        *,
        constraints: Constraints | None,
        transform: LinearTransform[Vector, T] | None,
    ) -> tuple[Problem, StateT, StatsT]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, problem: Problem, state: StateT) -> StateT:
        raise NotImplementedError

    def update_stats(self, problem: Problem, state: StateT, stats: StatsT) -> StatsT:  # noqa: ARG002
        return stats

    @abc.abstractmethod
    def terminate(
        self, problem: Problem, state: StateT, stats: StatsT
    ) -> tuple[bool, Result]:
        raise NotImplementedError

    def postprocess[T](
        self,
        problem: Problem,  # noqa: ARG002
        state: StateT,
        stats: StatsT,
        result: Result,
    ) -> OptimizeSolution[StateT, StatsT]:
        stats.end_time = time.perf_counter()
        solution: OptimizeSolution[StateT, StatsT] = OptimizeSolution(
            result=result, state=state, stats=stats
        )
        return solution

    def minimize[T](
        self,
        objective: Objective,
        params: T,
        *,
        callback: Callback[StateT, StatsT] | None = None,
        constraints: Constraints | None = None,
        transform: LinearTransform[Vector, T] | None = None,
    ) -> OptimizeSolution[StateT, StatsT]:
        problem: Problem
        state: StateT
        stats: StatsT
        problem, state, stats = self.init(
            objective, params, constraints=constraints, transform=transform
        )
        max_steps: int = self.max_steps or self._default_max_steps(state.params)
        done: bool = False
        n_steps: int = 0
        result: Result = Result.UNKNOWN_ERROR
        while n_steps < max_steps and not done:
            state = self.step(problem, state)
            n_steps += 1
            stats.n_steps = n_steps
            stats = self.update_stats(problem, state, stats)
            if callback is not None:
                callback(problem, state, stats)
            done, result = self.terminate(problem, state, stats)
        if not done:
            result = Result.MAX_STEPS_REACHED
        solution: OptimizeSolution[StateT, StatsT] = self.postprocess(
            problem, state, stats, result
        )
        return solution

    def _default_max_steps(self, params_flat: Vector) -> int:
        return params_flat.size

    def _transform_params[T](
        self, params: T, transform: LinearTransform[Vector, T] | None = None
    ) -> tuple[Vector, LinearTransform[Vector, T]]:
        params_flat: Vector
        if transform is None:
            structure: tree.Structure[T]
            params_flat, structure = tree.flatten(params)
            transform = FlattenTransform(structure=structure)
        else:
            params_flat = transform.backward_params(params)
        return params_flat, transform

    def _warn_unsupported_constraints(
        self, constraints: Iterable[Constraint] | None
    ) -> None:
        _logging_hide = True
        if not constraints:
            return
        for constr in constraints:
            if not isinstance(constr, self.supported_constraints):
                autolog.warning("%s does not support %r.", self.name, constr)
