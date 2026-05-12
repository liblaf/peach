from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import Problem

from ._base import CupySolver

if TYPE_CHECKING:
    import cupy as cp

type Vector = Float[Array, " N"]
type VectorCupy = Float[cp.ndarray, " N"]


@jarp.define(kw_only=True)
class CupyMinRes(CupySolver):
    """MINRES solver backed by `cupyx.scipy.sparse.linalg.minres`."""

    shift: float = jarp.field(default=0.0, kw_only=True)
    tol: float = jarp.field(default=1e-5, kw_only=True)

    @override
    def _options(self, problem: Problem) -> dict[str, Any]:
        """Build MINRES options, including `shift` and `tol`."""
        options: dict[str, Any] = super()._options(problem)
        options.update({"shift": self.shift, "tol": self.tol})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        """Call `cupyx.scipy.sparse.linalg.minres`."""
        from cupyx.scipy.sparse import linalg

        return linalg.minres(*args, **kwargs)
