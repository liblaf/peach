import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import State, Stats

type Scalar = Float[Array, ""]


@jarp.define
class FallbackState(State):
    state: list[State] = jarp.field(factory=list)


@jarp.define
class FallbackStats(Stats):
    stats: list[Stats] = jarp.field(factory=list)
    absolute_residual: Scalar = jarp.array(default=None, kw_only=True)
    absolute_residuals: Float[Array, " N"] = jarp.field(default=None, kw_only=True)
