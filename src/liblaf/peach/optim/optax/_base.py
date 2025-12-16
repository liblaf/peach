import warnings
from collections.abc import Iterable
from typing import override

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim.abc import (
    Callback,
    Optimizer,
    OptimizeSolution,
    Params,
    Result,
    SetupResult,
)
from liblaf.peach.optim.objective import Objective

from ._types import OptaxState, OptaxStats

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class Optax(Optimizer[OptaxState, OptaxStats]):
    from ._types import OptaxState as State
    from ._types import OptaxStats as Stats

    Solution = OptimizeSolution[State, Stats]
    Callback = Callback[State, Stats]

    wrapped: optax.GradientTransformation
    gtol: Scalar = tree.array(
        default=1e-5, converter=tree.converters.asarray, kw_only=True
    )

    @override
    def init(
        self,
        objective: Objective,
        params: Params,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> SetupResult[State, Stats]:
        state: OptaxState
        stats: OptaxStats
        objective, constraints, state, stats = super().init(
            objective, params, constraints=constraints
        )
        state.wrapped = self.wrapped.init(state.params_flat)
        return SetupResult(objective, constraints, state, stats)

    @override
    def step(
        self,
        objective: Objective,
        state: State,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> State:
        if constraints:
            warnings.warn(
                "Optax optimizer does not support constraints. Ignoring them.",
                RuntimeWarning,
                stacklevel=3,
            )
        assert objective.grad is not None
        state.grad_flat = objective.grad(state.params_flat)
        state.updates_flat, state.wrapped = self.wrapped.update(  # pyright: ignore[reportAttributeAccessIssue]
            state.grad_flat, state.wrapped, state.params_flat
        )
        state.params_flat = optax.apply_updates(state.params_flat, state.updates_flat)  # pyright: ignore[reportAttributeAccessIssue]
        return state

    @override
    def terminate(
        self,
        objective: Objective,
        state: State,
        stats: Stats,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> tuple[bool, Result]:
        g_norm: Scalar = jnp.linalg.norm(state.grad_flat, ord=jnp.inf)
        if state.first_grad_norm is None:
            state.first_grad_norm = g_norm
            return False, Result.UNKNOWN_ERROR
        if g_norm <= self.gtol * state.first_grad_norm:
            return True, Result.SUCCESS
        return False, Result.UNKNOWN_ERROR
