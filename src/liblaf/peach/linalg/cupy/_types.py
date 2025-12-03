from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.linalg.abc import State, Stats

type Scalar = Float[Array, ""]


@tree.define
class CupyState(State):
    pass


@tree.define
class CupyStats(Stats):
    info: int = -1
    n_steps: int | None = None
