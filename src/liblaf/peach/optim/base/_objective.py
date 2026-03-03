from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@runtime_checkable
class Objective[X](Protocol):
    def update(self, state: X, params: Vector, /) -> X: ...


@runtime_checkable
class SupportsFun[X](Protocol):
    def fun(self, state: X, /) -> Scalar: ...


@runtime_checkable
class SupportsGrad[X](Protocol):
    def grad(self, state: X, /) -> Vector: ...


@runtime_checkable
class SupportsValueAndGrad[X](Protocol):
    def value_and_grad(self, state: X, /) -> tuple[Scalar, Vector]: ...


@runtime_checkable
class SupportsHessProd[X](Protocol):
    def hess_prod(self, state: X, p: Vector, /) -> Vector: ...


@runtime_checkable
class SupportsHessDiag[X](Protocol):
    def hess_diag(self, state: X, /) -> Vector: ...


@runtime_checkable
class SupportsHessQuad[X](Protocol):
    def hess_quad(self, state: X, p: Vector, /) -> Scalar: ...
