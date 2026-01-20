from typing import override

from jaxtyping import Array, Float

from liblaf.peach import tree

from ._abc import LinearTransform

type Vector = Float[Array, " N"]


@tree.define
class FlattenTransform[T](LinearTransform[T, Vector]):
    structure: tree.Structure = tree.static(default=None, repr=False, kw_only=True)

    @override
    def forward_primals(self, params: T) -> Vector:
        return self._flatten(params)

    @override
    def forward_tangents(self, params: T, grad: T) -> Vector:
        return self._flatten(grad)

    @override
    def forward_hess_diag(self, params: T, hess_diag: T) -> Vector:
        return self._flatten(hess_diag)

    @override
    def backward_params(self, params_out: Vector) -> T:
        return self.structure.unflatten(params_out)

    @override
    def backward_grad(self, params_out: Vector, grad_out: Vector) -> T:
        return self.structure.unflatten(grad_out)

    @override
    def backward_hess_diag(self, params_out: Vector, hess_diag_out: Vector) -> T:
        return self.structure.unflatten(hess_diag_out)

    def _flatten(self, obj: T) -> Vector:
        flat: Vector
        flat, self.structure = tree.flatten(obj)
        return flat
