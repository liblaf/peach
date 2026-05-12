from typing import Protocol

from jaxtyping import Array, Float

from liblaf.peach.utils import not_implemented

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class BaseProblem[X](Protocol): ...


class State(Protocol):
    @property
    def params(self) -> Vector: ...


class Stats(Protocol): ...


class Problem[X](Protocol):
    @not_implemented
    def fun(self, state: X, /) -> Scalar: ...
    @not_implemented
    def grad(self, state: X, /) -> Vector: ...
    @not_implemented
    def hess_diag(self, state: X, /) -> Vector: ...
    @not_implemented
    def hess_prod(self, state: X, p: Vector, /) -> Vector: ...
    @not_implemented
    def hess_quad(self, state: X, p: Vector, /) -> Scalar: ...
    @not_implemented
    def value_and_grad(self, state: X, /) -> tuple[Scalar, Vector]: ...

    @not_implemented
    def before_step(self, state: X, x: Vector, /) -> X: ...
    @not_implemented
    def before_trial(self, state: X, x: Vector, /) -> X: ...
    @not_implemented
    def max_step_size(self, state: X, p: Vector, /) -> Scalar: ...

    @not_implemented
    def callback(self, model_state: X, opt_state: State, /) -> None: ...
