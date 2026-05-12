from typing import override

import jax
from jaxtyping import Array, Shaped

from liblaf import jarp

from ._base import JaxSolver

type Vector = Shaped[Array, " free"]


@jarp.define(kw_only=True)
class JaxCG(JaxSolver):
    """Conjugate-gradient solver backed by `jax.scipy.sparse.linalg.cg`."""

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, None]:
        """Call `jax.scipy.sparse.linalg.cg`."""
        return jax.scipy.sparse.linalg.cg(*args, **kwargs)
