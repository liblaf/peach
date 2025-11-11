import time

from jaxtyping import Array, Float

from liblaf.peach import tree_utils
from liblaf.peach.linalg.abc import Params
from liblaf.peach.tree_utils import TreeView, Unflatten

type Vector = Float[Array, " N"]


_clock = time.perf_counter


@tree_utils.define
class JaxState:
    params = TreeView[Params]()
    """x"""
    params_flat: Vector = tree_utils.array(default=None)

    b = TreeView[Params]()
    b_flat: Vector = tree_utils.array(default=None)

    unflatten: Unflatten[Params] | None = None


@tree_utils.define
class JaxStats:
    start_time: float = tree_utils.field(factory=_clock, init=False)
    end_time: float | None = None
    residual_relative: float | None = None

    @property
    def time(self) -> float:
        if self.end_time is None:
            return _clock() - self.start_time
        return self.end_time - self.start_time
