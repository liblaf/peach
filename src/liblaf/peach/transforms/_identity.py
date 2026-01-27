from typing import override

import jarp

from ._base import Transform


@jarp.define
class IdentityTransform[T](Transform[T, T]):
    @override
    def forward_primals(self, primals_in: T) -> T:
        return primals_in

    @override
    def forward_tangents(self, tangents_in: T) -> T:
        return tangents_in

    @override
    def backward_primals(self, primals_out: T) -> T:
        return primals_out

    @override
    def backward_tangents(self, tangents_out: T) -> T:
        return tangents_out

    @override
    def backward_hess_diag(self, hess_diag_out: T) -> T:
        return hess_diag_out
