from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.linalg.abc import State, Stats

type Scalar = Float[Array, ""]


@tree.define
class JaxState(State):
    pass


@tree.define
class JaxStats(Stats):
    residual_relative: Scalar = tree.array(default=None)


@tree.define
class JaxCompositeStats(Stats):
    stats: list[JaxStats] = tree.field(factory=list)
