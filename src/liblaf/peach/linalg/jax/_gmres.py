from typing import Any, Literal, override

import jax
from jaxtyping import Array, ArrayLike, Integer, Shaped

from liblaf.peach import tree
from liblaf.peach.linalg.system import LinearSystem

from ._base import JaxSolver

type Vector = Shaped[Array, " free"]


@tree.define
class JaxGMRES(JaxSolver):
    restart: int = 20
    solve_method: Literal["incremental", "batched"] = "batched"

    @override
    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(system)
        options.update({"restart": self.restart, "solve_method": self.solve_method})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Integer[ArrayLike, ""]]:
        return jax.scipy.sparse.linalg.gmres(*args, **kwargs)
