from __future__ import annotations

from typing import TYPE_CHECKING, override

from jaxtyping import Float

from liblaf.peach import tree

from ._base import CupySolver

if TYPE_CHECKING:
    import cupy as cp


type FreeCp = Float[cp.ndarray, " free"]


@tree.define
class CupyCG(CupySolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        from cupyx.scipy.sparse import linalg

        return linalg.cg(*args, **kwargs)
