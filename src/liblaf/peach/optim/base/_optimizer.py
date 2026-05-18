from typing import cast

from jaxtyping import Float
from torch import Tensor

from liblaf.peach.utils import is_implemented

from ._protocols import BaseProblem, Problem
from ._types import Solution, State, Stats

type Vector = Float[Tensor, " N"]


class Optimizer[S: State, T: Stats]:
    """Base class for iterative optimizers."""

    from ._types import Result, Solution, State, Stats

    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> S:
        """Create optimizer state from a model state and parameter vector."""
        raise NotImplementedError

    def step[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S
    ) -> tuple[X, S]:
        """Advance the optimizer by one step."""
        raise NotImplementedError

    def terminate[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S
    ) -> tuple[bool, Result]:
        """Return whether optimization should stop and why."""
        raise NotImplementedError

    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S, result: Result
    ) -> Solution[S, T]:
        """Build the final solution object."""
        del problem, model_state
        stats: T = cast("T", {})
        return Optimizer.Solution(result=result, state=opt_state, stats=stats)

    def minimize[X](
        self, problem: BaseProblem[X], model_state: X, params: Vector
    ) -> tuple[Solution[S, T], X]:
        """Run optimization until [`terminate`][liblaf.peach.optim.base.Optimizer.terminate] succeeds."""
        problem: Problem[X] = cast("Problem[X]", problem)
        opt_state: S = self.init(problem, model_state, params)
        while True:
            model_state, opt_state = self.step(problem, model_state, opt_state)
            if is_implemented(problem, Problem.callback):
                problem.callback(model_state, opt_state)
            ok, result = self.terminate(problem, model_state, opt_state)
            if ok:
                break
        solution: Solution[S, T] = self.postprocess(
            problem, model_state, opt_state, result
        )
        return solution, model_state
