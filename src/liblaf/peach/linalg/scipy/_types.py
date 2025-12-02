from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.linalg.abc import State, Stats

type Scalar = Float[Array, ""]


@tree.define
class ScipyState(State):
    pass


@tree.define
class ScipyStats(Stats):
    info: int = -1
    n_steps: int | None = None
