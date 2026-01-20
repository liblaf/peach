import functools
from collections.abc import Callable
from typing import Protocol

from liblaf.peach import tree

from ._context import FunctionContext


class ObjectiveProtocol(Protocol):
    @property
    def fun(self) -> Callable | None: ...
    @property
    def grad(self) -> Callable | None: ...
    @property
    def hess_prod(self) -> Callable | None: ...
    @property
    def hess_diag(self) -> Callable | None: ...
    @property
    def hess_quad(self) -> Callable | None: ...
    @property
    def value_and_grad(self) -> Callable | None: ...


@tree.define
class Objective(FunctionContext, ObjectiveProtocol):
    _fun_wrapped: Callable | None = tree.field(default=None, alias="fun")
    _grad_wrapped: Callable | None = tree.field(default=None, alias="grad")
    _hess_prod_wrapped: Callable | None = tree.field(default=None, alias="hess_prod")
    _hess_diag_wrapped: Callable | None = tree.field(default=None, alias="hess_diag")
    _hess_quad_wrapped: Callable | None = tree.field(default=None, alias="hess_quad")
    _value_and_grad_wrapped: Callable | None = tree.field(
        default=None, alias="value_and_grad"
    )

    @functools.cached_property
    def fun(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._fun_wrapped, input_params=(0,))

    @functools.cached_property
    def grad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._grad_wrapped, input_params=(0,), output_grad=(0,))

    @functools.cached_property
    def hess_prod(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(
            self._hess_prod_wrapped,
            input_params=(0,),
            input_grad=(1,),
            output_grad=(0,),
        )

    @functools.cached_property
    def hess_diag(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(
            self._hess_diag_wrapped, input_params=(0,), output_hess_diag=(0,)
        )

    @functools.cached_property
    def hess_quad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._hess_quad_wrapped, input_params=(0,), input_grad=(1,))

    @functools.cached_property
    def value_and_grad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(
            self._value_and_grad_wrapped, input_params=(0,), output_grad=(1,)
        )
