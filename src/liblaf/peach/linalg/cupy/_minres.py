from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from jaxtyping import Float

from liblaf.peach import tree
from liblaf.peach.linalg.system import LinearSystem

from ._base import CupySolver

if TYPE_CHECKING:
    import cupy as cp


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
        from cupyx.scipy.sparse import linalg

        return linalg.minres(*args, **kwargs)
