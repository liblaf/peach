import functools
from collections.abc import Callable

from liblaf.peach import tree

from ._context import FunctionContext


@tree.define
class Objective(FunctionContext):
    _prepare: Callable | None = tree.field(default=None, alias="prepare")
    _fun_wrapped: Callable | None = tree.field(default=None, alias="fun")
    _grad_wrapped: Callable | None = tree.field(default=None, alias="grad")
    _hess_prod_wrapped: Callable | None = tree.field(default=None, alias="hess_prod")
    _hess_diag_wrapped: Callable | None = tree.field(default=None, alias="hess_diag")
    _hess_quad_wrapped: Callable | None = tree.field(default=None, alias="hess_quad")
    _value_and_grad_wrapped: Callable | None = tree.field(
        default=None, alias="value_and_grad"
    )

    @functools.cached_property
    def prepare(self) -> Callable | None:
        return self._wraps(self._prepare, method="prepare")

    @functools.cached_property
    def fun(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._fun_wrapped, method="fun")

    @functools.cached_property
    def grad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._grad_wrapped, method="grad")

    @functools.cached_property
    def hess_prod(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._hess_prod_wrapped, method="hess_prod")

    @functools.cached_property
    def hess_diag(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._hess_diag_wrapped, method="hess_diag")

    @functools.cached_property
    def hess_quad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._hess_quad_wrapped, method="hess_quad")

    @functools.cached_property
    def value_and_grad(self) -> Callable | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._wraps(self._value_and_grad_wrapped, method="value_and_grad")
