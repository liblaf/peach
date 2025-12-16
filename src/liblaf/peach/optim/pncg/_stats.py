from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.abc import Stats

type Scalar = Float[Array, ""]


@tree.define
class PNCGStats(Stats):
    relative_decrease: Scalar = tree.array(default=None, kw_only=True)
