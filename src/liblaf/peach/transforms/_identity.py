from collections.abc import Callable
from typing import override

from liblaf.peach import tree
from liblaf.peach.transforms import LinearTransform


def _identity[T](x: T) -> T:
    return x


@tree.define
class IdentityTransform[T](LinearTransform[T, T]):
    @override
    def wraps[C: Callable](self, method: str, wrapped: C) -> C:
        return wrapped

    @override
    def forward_primals(self, primals: T) -> T:
        return primals

    @override
    def linearize(self, primals: T) -> tuple[T, Callable[[T], T]]:
        return primals, _identity

    @override
    def linear_transpose(self, tangents_out: T) -> T:
        return tangents_out

    @override
    def forward_tangents(self, primals: T, tangents: T) -> T:
        return tangents

    @override
    def forward_hess_diag(self, hess_diag: T) -> T:
        return hess_diag

    @override
    def backward_params(self, primals_out: T) -> T:
        return primals_out

    @override
    def backward_hess_diag(self, hess_diag_out: T) -> T:
        return hess_diag_out
