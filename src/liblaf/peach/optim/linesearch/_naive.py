from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._abc import LineSearch

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearchNaive(LineSearch):
    initial: LineSearch = tree.field()

    decay: Scalar = tree.array(
        default=0.5, converter=tree.converters.asarray, kw_only=True
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
        while True:
            params_next: Vector = params + step_size * search_direction
            f_next: Scalar = objective.fun(params_next)
            if jnp.isfinite(f_next) and f_next < f0:
                break
            step_size *= self.decay
        return step_size
