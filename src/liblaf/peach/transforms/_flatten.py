from collections.abc import Callable
from typing import override

from jaxtyping import Array, Float

from liblaf.peach import tree

from ._abc import LinearTransform

type Vector = Float[Array, " N"]


@tree.define
class FlattenTransform[T](LinearTransform[Vector, T]):
    structure: tree.Structure = tree.static(repr=False, kw_only=True)

    @override
    def forward_primals(self, primals: Vector) -> T:
        return self.structure.unflatten(primals)

    @override
    def linearize(self, primals: Vector) -> tuple[T, Callable[[Vector], T]]:
        params_out: T = self.structure.unflatten(primals)
        return params_out, self.structure.unflatten

    @override
    def linear_transpose(self, tangents_out: T) -> Vector:
        return self.structure.flatten(tangents_out)

    @override
    def forward_tangents(self, primals: Vector, tangents: Vector) -> T:
        return self.structure.unflatten(tangents)

    @override
    def forward_hess_diag(self, hess_diag: Vector) -> T:
        return self.structure.unflatten(hess_diag)

    @override
    def backward_params(self, primals_out: T) -> Vector:
        return self.structure.flatten(primals_out)

    @override
    def backward_hess_diag(self, hess_diag_out: T) -> Vector:
        return self.structure.flatten(hess_diag_out)
