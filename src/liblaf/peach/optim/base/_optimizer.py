import time

import jarp
import jax
from jaxtyping import Array, Bool, Float

from ._objective import Objective
from ._types import Solution, State, Stats

type BooleanNumeric = Bool[Array, ""]
type Vector = Float[Array, " N"]


@jarp.define
class Optimizer[P: Objective, S: State, T: Stats]:
    from ._types import Callback, Result, Solution, State, Stats

    jit: bool = jarp.static(default=True, kw_only=True)

    def init[X](
        self,
        objective: P,
        model_state: X,  # pyright: ignore[reportInvalidTypeVarUse]
        params: Vector,
    ) -> tuple[S, T]:
        raise NotImplementedError

    def step[X](
        self,
        objective: P,
        model_state: X,
        opt_state: S,
    ) -> tuple[X, S]:
        raise NotImplementedError

    def update_stats[X](
        self,
        objective: P,  # noqa: ARG002
        model_state: X,  # pyright: ignore[reportInvalidTypeVarUse]  # noqa: ARG002
        opt_state: S,  # noqa: ARG002
        opt_stats: T,
    ) -> T:
        return opt_stats

    def terminate[X](
        self,
        objective: P,
        model_state: X,  # pyright: ignore[reportInvalidTypeVarUse]
        opt_state: S,
        opt_stats: T,
    ) -> BooleanNumeric:
        raise NotImplementedError

    def postprocess[X](
        self,
        objective: P,  # noqa: ARG002
        model_state: X,  # pyright: ignore[reportInvalidTypeVarUse]  # noqa: ARG002
        opt_state: S,
        opt_stats: T,
    ) -> Solution[S, T]:
        opt_stats._end_time = time.perf_counter()  # noqa: SLF001
        return Optimizer.Solution(
            result=Optimizer.Result.SUCCESS, state=opt_state, stats=opt_stats
        )

    def minimize[X](
        self,
        objective: P,
        model_state: X,
        params: Vector,
        callback: Callback[X, S, T] | None = None,
    ) -> tuple[Solution[S, T], X]:
        opt_state: S
        opt_stats: T
        opt_state, opt_stats = self.init(objective, model_state, params)
        if self.jit:
            model_state, opt_state, opt_stats = self._while_loop_jit(
                objective, model_state, opt_state, opt_stats, callback
            )
        else:
            model_state, opt_state, opt_stats = self._while_loop(
                objective, model_state, opt_state, opt_stats, callback
            )
        solution: Solution[S, T] = self.postprocess(
            objective, model_state, opt_state, opt_stats
        )
        return solution, model_state

    def _while_loop[X](
        self,
        objective: P,
        model_state: X,
        opt_state: S,
        opt_stats: T,
        callback: Callback[X, S, T] | None = None,
    ) -> tuple[X, S, T]:
        def cond_fun(carry: tuple[X, S, T]) -> BooleanNumeric:
            model_state: X
            opt_state: S
            opt_stats: T
            model_state, opt_state, opt_stats = carry
            return ~self.terminate(objective, model_state, opt_state, opt_stats)

        def body_fun(
            carry: tuple[X, S, T],
        ) -> tuple[X, S, T]:
            model_state: X
            opt_state: S
            opt_stats: T
            model_state, opt_state, opt_stats = carry
            model_state, opt_state = self.step(objective, model_state, opt_state)
            opt_stats = self.update_stats(objective, model_state, opt_state, opt_stats)
            if callback is not None:
                if self.jit:
                    jax.debug.callback(
                        callback, objective, model_state, opt_state, opt_stats
                    )
                else:
                    callback(objective, model_state, opt_state, opt_stats)
            return model_state, opt_state, opt_stats

        return jarp.while_loop(
            cond_fun, body_fun, (model_state, opt_state, opt_stats), jit=self.jit
        )

    @jarp.jit(inline=True, filter=True)
    def _while_loop_jit[X](
        self,
        objective: P,
        model_state: X,
        opt_state: S,
        opt_stats: T,
        callback: Callback[X, S, T] | None = None,
    ) -> tuple[X, S, T]:
        return self._while_loop(objective, model_state, opt_state, opt_stats, callback)
