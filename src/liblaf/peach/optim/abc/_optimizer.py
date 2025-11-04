import abc

from liblaf import grapes
from liblaf.peach import tree_utils
from liblaf.peach.optim.objective import Objective

from ._solution import OptimizeSolution, Result
from ._typing import Aux, Callback, X


@tree_utils.tree
class Optimizer[State](abc.ABC):
    max_steps: int = 256

    @abc.abstractmethod
    def init(self, objective: Objective, x: X) -> State: ...

    @abc.abstractmethod
    def step(
        self, objective: Objective, x: X, state: State
    ) -> tuple[X, State, Aux]: ...

    @abc.abstractmethod
    def terminate(
        self, objective: Objective, x: X, state: State
    ) -> tuple[bool, Result]: ...

    @abc.abstractmethod
    def postprocess(
        self, objective: Objective, x: X, aux: Aux, state: State, result: Result
    ) -> OptimizeSolution:
        solution: OptimizeSolution = OptimizeSolution(
            aux=aux, state=state, stats={}, x=x, result=result
        )
        return solution

    def minimize(
        self,
        objective: Objective,
        x: X,
        callback: Callback[State] | None = None,
    ) -> OptimizeSolution:
        with grapes.timer(name=str(self)) as timer:
            state: State = self.init(objective, x)
            aux: Aux = None
            done: bool = False
            n_steps: int = 0
            result: Result = Result.FAILURE
            while n_steps < self.max_steps and not done:
                x, state, aux = self.step(objective, x, state)
                n_steps += 1
                if callback is not None:
                    callback(state, n_steps)
                done, result = self.terminate(objective, x, state)
            if not done:
                result = Result.MAX_STEPS_REACHED
            solution: OptimizeSolution = self.postprocess(
                objective, x, aux, state, result
            )
        solution.stats["n_steps"] = n_steps
        solution.stats["time"] = timer.elapsed()
        return solution
