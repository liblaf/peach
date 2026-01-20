import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, overload

import jax
import wrapt
from jaxtyping import PyTree

from liblaf.peach import compile_utils, tree, utils

from ._registry import Registry, WrappedMethod

type Params = PyTree


@tree.define
class LinearTransform[I, O]:
    registry: ClassVar[Registry] = Registry()

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        for parent in inspect.getmro(cls):
            if parent is cls:
                continue
            parent_registry: Registry | None = getattr(parent, "registry", None)
            if parent_registry is None:
                continue
            if cls.registry is parent_registry:
                cls.registry = parent_registry.copy()
            cls.registry.inherit(parent_registry)

    @overload
    @classmethod
    def register[C: WrappedMethod](cls, func: C, *, name: str | None = None) -> C: ...
    @overload
    @classmethod
    def register[C: WrappedMethod](
        cls, *, name: str | None = None
    ) -> Callable[[C], C]: ...
    @classmethod
    def register(
        cls, func: WrappedMethod | None = None, *, name: str | None = None
    ) -> Callable[..., Any]:
        return cls.registry.func(func, name=name)

    def wraps(self, method: str, wrapped: Callable[..., Any]) -> Callable[..., Any]:
        wrapper_or_name: str | Callable = self.registry[method]
        wrapper: Callable = (
            getattr(self, wrapper_or_name)
            if isinstance(wrapper_or_name, str)
            else wrapper_or_name
        )
        return wrapt.decorator(wrapper)(wrapped)

    def forward_primals(self, primals: I) -> O:
        raise NotImplementedError

    @compile_utils.jit(inline=True)
    def linearize(self, primals: I) -> tuple[O, Callable[[I], O]]:
        primals_out: O
        lin_fun: Callable[[I], O]
        primals_out, lin_fun = jax.linearize(self.forward_primals, primals)
        return primals_out, lin_fun

    @compile_utils.jit(inline=True)
    def linear_transpose(self, tangents_out: O) -> I:
        primals_shape: I = jax.eval_shape(self.backward_params, tangents_out)
        f_transpose: Callable[[O], I] = jax.linear_transpose(
            self.forward_primals, primals_shape
        )
        return f_transpose(tangents_out)

    @compile_utils.jit(inline=True)
    def forward_tangents(self, primals: I, tangents: I) -> O:
        lin_fun: Callable[[I], O]
        _primals_out, lin_fun = self.linearize(primals)
        return lin_fun(tangents)

    def forward_hess_diag(self, hess_diag: I) -> O:
        raise NotImplementedError

    def backward_params(self, primals_out: O) -> I:
        raise NotImplementedError

    def backward_hess_diag(self, hess_diag_out: O) -> I:
        raise NotImplementedError

    @registry.method
    def _call_prepare(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        args[0] = self.forward_primals(primals)
        return wrapped(*args, **kwargs)

    @registry.method
    def _call_fun(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        args[0] = self.forward_primals(primals)
        return wrapped(*args, **kwargs)

    @registry.method
    def _call_grad(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        args[0] = self.forward_primals(primals)
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[0] = self.linear_transpose(outputs[0])
        return utils.unpack(outputs)

    @registry.method
    def _call_hess_prod(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        lin_fun: Callable[[I], O]
        args[0], lin_fun = self.linearize(primals)
        args[1] = lin_fun(args[1])
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[0] = self.linear_transpose(outputs[0])
        return utils.unpack(outputs)

    @registry.method
    def _call_hess_diag(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        args[0] = self.forward_primals(primals)
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[0] = self.backward_hess_diag(outputs[0])
        return utils.unpack(outputs)

    @registry.method
    def _call_hess_quad(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        lin_fun: Callable[[I], O]
        args[0], lin_fun = self.linearize(primals)
        args[1] = lin_fun(args[1])
        return wrapped(*args, **kwargs)

    @registry.method
    def _call_value_and_grad(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        args: list[Any] = list(args)
        primals: I = args[0]
        args[0] = self.forward_primals(primals)
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[1] = self.linear_transpose(outputs[1])
        return utils.unpack(outputs)
