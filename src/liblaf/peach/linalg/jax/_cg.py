from typing import override

import jarp
import jax
from jaxtyping import Array, Shaped

from ._base import JaxSolver

type Vector = Shaped[Array, " free"]


@jarp.define
class JaxCG(JaxSolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, None]:
        return jax.scipy.sparse.linalg.cg(*args, **kwargs)
