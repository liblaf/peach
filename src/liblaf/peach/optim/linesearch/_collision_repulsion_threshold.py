from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

from ._abc import LineSearch

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearchCollisionRepulsionThreshold(LineSearch):
    collision_repulsion_threshold: Scalar = tree.array(
        default=jnp.inf, converter=tree.converters.asarray
    )

    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
    ) -> Scalar:
        p_norm: Scalar = jnp.linalg.norm(search_direction, ord=jnp.inf)
        return self.collision_repulsion_threshold / (2.0 * p_norm)  # pyright: ignore[reportAssignmentType]
