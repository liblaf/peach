import logging
from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._abc import LineSearch

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


logger: logging.Logger = logging.getLogger(__name__)


@tree.define
class LineSearchNaive(LineSearch):
    initial: LineSearch = tree.field()

    decay: Scalar = tree.array(
        default=0.5, converter=tree.converters.asarray, kw_only=True
    )
    max_steps: Integer[Array, ""] = tree.array(
        default=20, converter=tree.converters.asarray, kw_only=True
    )

    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
    ) -> Scalar:
        assert objective.fun is not None
        step_size: Scalar = self.initial.search(
            objective, params, grad, search_direction
        )
        f0: Scalar = objective.fun(params)
        for i in range(self.max_steps):
            if i > 0:
                step_size *= self.decay
            params_next: Vector = params + step_size * search_direction
            f_next: Scalar = objective.fun(params_next)
            if jnp.isfinite(f_next) and f_next < f0:
                break
        return step_size
