from typing import Any, override

import cupy as cp
from cupyx.scipy.sparse import linalg
from jaxtyping import Float

from liblaf.peach import tree
from liblaf.peach.linalg.system import LinearSystem

from ._base import CupySolver

type FreeCp = Float[cp.ndarray, " free"]


@tree.define
class CupyMinRes(CupySolver):
    shift: float = tree.field(default=0.0, kw_only=True)

    @override
    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(system)
        options.update({"shift": self.shift})
        options.pop("atol", None)
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        return linalg.minres(*args, **kwargs)
