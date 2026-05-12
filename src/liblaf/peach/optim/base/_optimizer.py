from typing import cast

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from liblaf import jarp
from liblaf.peach.utils import implemented

from ._protocols import BaseProblem, Problem
from ._types import Result, Solution, State, Stats

type Vector = Float[Array, " N"]


@jarp.define
class Optimizer[S: State, T: Stats]:
    from ._types import Result, Solution, State, Stats

    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> S:
        raise NotImplementedError

    def step[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S
    ) -> tuple[X, S]:
        raise NotImplementedError

    def terminate[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S
    ) -> tuple[Bool[Array, ""], Result]:
        raise NotImplementedError

    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S, result: Result
    ) -> Solution[S, T]:
        del problem, model_state
        stats: T = cast("T", {})
        return Optimizer.Solution(result=result, state=opt_state, stats=stats)

    def minimize[X](
        self, problem: BaseProblem[X], model_state: X, params: Vector
    ) -> tuple[Solution[S, T], X]:
        opt_state: S = self.init(problem, model_state, params)
        model_state, opt_state, result = self._while_loop(
            problem, model_state, opt_state
        )
        solution: Solution[S, T] = self.postprocess(
            problem, model_state, opt_state, result
        )
        return solution, model_state

    @jarp.fallback_jit(inline=True)
    def _while_loop[X](
        self, problem: BaseProblem, model_state: X, opt_state: S
    ) -> tuple[X, S, Result]:
        type Carry = tuple[X, S, Bool[Array, ""], Result]
        problem: Problem[X] = cast("Problem[X]", problem)

        def cond_fun(carry: Carry) -> Bool[Array, ""]:
            _model_state, _opt_state, ok, _result = carry
            return ~ok

        def body_fun(carry: Carry) -> Carry:
            model_state, opt_state, ok, result = carry
            model_state, opt_state = self.step(problem, model_state, opt_state)
            if implemented(problem, Problem.callback):
                problem.callback(model_state, opt_state)
            ok, result = self.terminate(problem, model_state, opt_state)
            return model_state, opt_state, ok, result

        model_state, opt_state, _, result = jarp.while_loop(
            cond_fun,
            body_fun,
            (model_state, opt_state, jnp.asarray(False), Result.UNKNOWN_ERROR),  # noqa: FBT003
        )
        return model_state, opt_state, result
