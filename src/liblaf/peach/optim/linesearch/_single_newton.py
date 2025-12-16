from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._abc import LineSearch

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearchSingleNewton(LineSearch):
    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
    ) -> Scalar:
        assert objective.hess_quad is not None
        hess_quad: Scalar = objective.hess_quad(params, search_direction)
        if not jnp.isfinite(hess_quad) or hess_quad <= 0.0:
            hess_quad = jnp.sum(jnp.square(search_direction))
        return -jnp.vdot(grad, search_direction) / hess_quad
