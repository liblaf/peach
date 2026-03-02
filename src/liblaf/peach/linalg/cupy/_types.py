import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import State, Stats

type Scalar = Float[Array, ""]


@jarp.define
class CupyState(State):
    pass


@jarp.define
class CupyStats(Stats):
    info: int = -1
    n_steps: int | None = jarp.field(default=None, kw_only=True)
    relative_residual: Scalar = jarp.array(default=None, kw_only=True)
