from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import LinearSystem

from ._base import CupySolver

if TYPE_CHECKING:
    import cupy as cp


type FreeCp = Float[cp.ndarray, " free"]
type Free = Float[Array, " free"]


@jarp.define
class CupyMinRes(CupySolver):
    shift: float = jarp.field(default=0.0, kw_only=True)

    @override
    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(system)
        options.update({"shift": self.shift})
        options.pop("atol", None)
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        from cupyx.scipy.sparse import linalg

        return linalg.minres(*args, **kwargs)
