from typing import override

import cupy as cp
from cupyx.scipy.sparse import linalg
from jaxtyping import Float

from liblaf.peach import tree

from ._base import CupySolver

type FreeCp = Float[cp.ndarray, " free"]


@tree.define
class CupyCG(CupySolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeCp, int]:
        return linalg.cg(*args, **kwargs)
