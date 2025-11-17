import time

from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.linalg.abc import Params
from liblaf.peach.tree import TreeView, Unflatten

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

_clock = time.perf_counter


@tree.define
class JaxState:
    params = TreeView[Params]()
    """x"""
    params_flat: Vector = tree.array(default=None)

    unflatten: Unflatten[Params] | None = None


@tree.define
class JaxStats:
    start_time: float = tree.field(factory=_clock, init=False)
    end_time: float | None = None
    residual_relative: Scalar | None = None

    @property
    def time(self) -> float:
        if self.end_time is None:
            return _clock() - self.start_time
        return self.end_time - self.start_time
