from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.linalg.abc import State, Stats

type Scalar = Float[Array, ""]


@tree.define
class CompositeState(State):
    state: list[State] = tree.field(factory=list)


@tree.define
class CompositeStats(Stats):
    stats: list[Stats] = tree.field(factory=list)
    relative_residual: Scalar = tree.array(default=None, kw_only=True)
