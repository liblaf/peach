from typing import override

import jarp
from jaxtyping import Array, Float

from ._base import Transform

type Vector = Float[Array, " N"]


@jarp.define
class UnravelTransform[T](Transform[Vector, T]):
    structure: jarp.Structure[T]

    @override
    def forward_primals(self, primals_in: Vector) -> T:
        return self.structure.unravel(primals_in)

    @override
    def forward_tangents(self, tangents_in: Vector) -> T:
        return self.structure.unravel(tangents_in)

    @override
    def backward_primals(self, primals_out: T) -> Vector:
        return self.structure.ravel(primals_out)

    @override
    def backward_tangents(self, tangents_out: T) -> Vector:
        return self.structure.ravel(tangents_out)

    @override
    def backward_hess_diag(self, hess_diag_out: T) -> Vector:
        return self.structure.ravel(hess_diag_out)
