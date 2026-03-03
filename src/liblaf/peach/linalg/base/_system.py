from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

type Vector = Float[Array, " N"]


@runtime_checkable
class LinearSystem(Protocol):
    @property
    def b(self) -> Vector: ...


@runtime_checkable
class SupportsMatvec(Protocol):
    def matvec(self, x: Vector) -> Vector: ...


@runtime_checkable
class SupportsRmatvec(Protocol):
    def rmatvec(self, x: Vector) -> Vector: ...


@runtime_checkable
class SupportsPreconditioner(Protocol):
    def preconditioner(self, x: Vector) -> Vector: ...


@runtime_checkable
class SupportsRpreconditioner(Protocol):
    def rpreconditioner(self, x: Vector) -> Vector: ...
