from typing import Any, Literal, override

import jax
from jaxtyping import Array, ArrayLike, Integer, Shaped

from liblaf.peach import tree_utils
from liblaf.peach.linalg.op import LinearOperator

from ._base import JaxSolver
from ._types import JaxState

type Vector = Shaped[Array, " free"]


@tree_utils.define
class JaxGMRES(JaxSolver):
    restart: int = 20
    solve_method: Literal["incremental", "batched"] = "batched"

    @override
    def _options(self, op: LinearOperator, state: JaxState) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(op, state)
        options.update({"restart": self.restart, "solve_method": self.solve_method})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[Vector, Integer[ArrayLike, ""]]:
        return jax.scipy.sparse.linalg.gmres(*args, **kwargs)
