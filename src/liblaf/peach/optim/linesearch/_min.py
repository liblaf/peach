from collections.abc import Iterable
from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._abc import LineSearch

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearchMin(LineSearch):
    methods: Iterable[LineSearch] = tree.field()

    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
    ) -> Scalar:
        return jnp.min(
            jnp.stack(
                [
                    method.search(objective, params, grad, search_direction)
                    for method in self.methods
                ]
            )
        )
