from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import jarp
from jaxtyping import Array, Float

from ._base import CupySolver
from ._types import CupyLinearSystem

if TYPE_CHECKING:
    import cupy as cp


type Vector = Float[Array, " N"]
type VectorCupy = Float[cp.ndarray, " N"]


@jarp.define
class CupyMinRes(CupySolver):
    shift: float = jarp.field(default=0.0, kw_only=True)
    tol: float = jarp.field(default=1e-5, kw_only=True)

    @override
    def _options(self, system: CupyLinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(system)
        options.update({"shift": self.shift, "tol": self.tol})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        from cupyx.scipy.sparse import linalg

        return linalg.minres(*args, **kwargs)
