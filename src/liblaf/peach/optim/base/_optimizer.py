import jarp
import jax
from jaxtyping import Array, Bool

from ._objective import Objective
from ._types import Solution, State, Stats

type BooleanNumeric = Bool[Array, ""]


@jarp.define
class Optimizer[StateT: State, StatsT: Stats]:
    from ._types import Callback, Result, Solution, State, Stats

    jit: bool = jarp.static(default=True, kw_only=True)

    def init[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
    ) -> tuple[StateT, StatsT]:
        raise NotImplementedError

    def step[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: StateT,
    ) -> tuple[ModelState, StateT]:
        raise NotImplementedError

    def update_stats[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],  # noqa: ARG002
        model_state: ModelState,  # noqa: ARG002
        opt_state: StateT,  # noqa: ARG002
        stats: StatsT,
    ) -> StatsT:
        return stats

    def terminate[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: StateT,
        opt_stats: StatsT,
    ) -> BooleanNumeric:
        raise NotImplementedError

    def postprocess[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: StateT,
        opt_stats: StatsT,
    ) -> Solution[StateT, StatsT]:
        raise NotImplementedError

    def minimize[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        params: Params,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> tuple[Solution[StateT, StatsT], ModelState]:
        opt_state: StateT
        opt_stats: StatsT
        opt_state, opt_stats = self.init(objective, model_state, params)
        if self.jit:
            model_state, opt_state, opt_stats = self._while_loop_jit(
                objective, model_state, opt_state, opt_stats, callback
            )
        else:
            model_state, opt_state, opt_stats = self._while_loop(
                objective, model_state, opt_state, opt_stats, callback
            )
        solution: Solution[StateT, StatsT] = self.postprocess(
            objective, model_state, opt_state, opt_stats
        )
        return solution, model_state

    def _while_loop[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: StateT,
        opt_stats: StatsT,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> tuple[ModelState, StateT, StatsT]:
        def cond_fun(carry: tuple[ModelState, StateT, StatsT]) -> BooleanNumeric:
            model_state: ModelState
            opt_state: StateT
            opt_stats: StatsT
            model_state, opt_state, opt_stats = carry
            return ~self.terminate(objective, model_state, opt_state, opt_stats)

        def body_fun(
            carry: tuple[ModelState, StateT, StatsT],
        ) -> tuple[ModelState, StateT, StatsT]:
            model_state: ModelState
            opt_state: StateT
            opt_stats: StatsT
            model_state, opt_state, opt_stats = carry
            model_state, opt_state = self.step(objective, model_state, opt_state)
            opt_stats = self.update_stats(objective, model_state, opt_state, opt_stats)
            if callback is not None:
                jax.debug.callback(callback, model_state, opt_state, opt_stats)
            return model_state, opt_state, opt_stats

        return jarp.while_loop(
            cond_fun, body_fun, (model_state, opt_state, opt_stats), jit=self.jit
        )

    @jarp.jit(inline=True, filter=True)
    def _while_loop_jit[ModelState, Params](
        self,
        objective: Objective[ModelState, Params],
        model_state: ModelState,
        opt_state: StateT,
        opt_stats: StatsT,
        callback: Callback[StateT, StatsT] | None = None,
    ) -> tuple[ModelState, StateT, StatsT]:
        return self._while_loop(objective, model_state, opt_state, opt_stats, callback)
