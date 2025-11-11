from typing import override

import jax
from jaxtyping import Array, Shaped

from liblaf.peach import tree

from ._base import JaxSolver

type Vector = Shaped[Array, " free"]


@tree.define
class JaxCG(JaxSolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, None]:
        return jax.scipy.sparse.linalg.cg(*args, **kwargs)
