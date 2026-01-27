from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, overload

import jarp
import jax.tree_util as jtu
from jaxtyping import Array, Float

from liblaf.peach.transforms import IdentityTransform, Transform

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class UpdateProtocol[State, Params](Protocol):
    def __call__(self, state: State, params: Params, /) -> State: ...


class FunProtocol[State, Params](Protocol):
    def __call__(self, state: State, /) -> Scalar: ...


class GradProtocol[State, Params](Protocol):
    def __call__(self, state: State, /) -> Params: ...


class HessDiagProtocol[State, Params](Protocol):
    def __call__(self, state: State, /) -> Params: ...


class HessProdProtocol[State, Params](Protocol):
    def __call__(self, state: State, vector: Params, /) -> Params: ...


class HessQuadProtocol[State, Params](Protocol):
    def __call__(self, state: State, vector: Params, /) -> Scalar: ...


class ValueAndGradProtocol[State, Params](Protocol):
    def __call__(self, state: State, /) -> tuple[Scalar, Params]: ...


@jarp.define
class Objective[State, Params]:
    _update: Callable = jarp.field(kw_only=True, alias="update")
    _fun: Callable | None = jarp.field(default=None, kw_only=True, alias="fun")
    _grad: Callable | None = jarp.field(default=None, kw_only=True, alias="grad")
    _hess_diag: Callable | None = jarp.field(
        default=None, kw_only=True, alias="hess_diag"
    )
    _hess_prod: Callable | None = jarp.field(
        default=None, kw_only=True, alias="hess_prod"
    )
    _hess_quad: Callable | None = jarp.field(
        default=None, kw_only=True, alias="hess_quad"
    )
    _value_and_grad: Callable | None = jarp.field(
        default=None, kw_only=True, alias="value_and_grad"
    )

    args: Sequence[Any] = jarp.field(default=(), kw_only=True)
    kwargs: Mapping[str, Any] = jarp.field(factory=dict, kw_only=True)
    transform: Transform[Vector, Params] = jarp.field(
        factory=IdentityTransform, kw_only=True
    )

    @property
    def update(self) -> UpdateProtocol[State, Params]:
        return self._wraps(self._update)

    @property
    def fun(self) -> FunProtocol[State, Params] | None:
        return self._wraps(self._fun)

    @property
    def grad(self) -> GradProtocol[State, Params] | None:
        return self._wraps(self._grad)

    @property
    def hess_diag(self) -> HessDiagProtocol[State, Params] | None:
        return self._wraps(self._hess_diag)

    @property
    def hess_prod(self) -> HessProdProtocol[State, Params] | None:
        return self._wraps(self._hess_prod)

    @property
    def hess_quad(self) -> HessQuadProtocol[State, Params] | None:
        return self._wraps(self._hess_quad)

    @property
    def value_and_grad(self) -> ValueAndGradProtocol[State, Params] | None:
        return self._wraps(self._value_and_grad)

    @overload
    def _wraps(self, func: None) -> None: ...
    @overload
    def _wraps[T](self, func: Callable[..., T]) -> Callable[..., T]: ...
    def _wraps[T](self, func: Callable[..., T] | None) -> Callable[..., T] | None:
        if func is None:
            return None
        if self.args or self.kwargs:
            func = jtu.Partial(func, *self.args, **self.kwargs)
        return func
