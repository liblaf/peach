import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import State, Stats

type Scalar = Float[Array, ""]


@jarp.define
class JaxState(State):
    pass


@jarp.define
class JaxStats(Stats):
    info: int | None = None
    residual_relative: Scalar = jarp.array(default=None)
