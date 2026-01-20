from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import jax
import wrapt
from jaxtyping import PyTree

from liblaf.peach import compile_utils, tree, utils

from ._registry import Registry

type Params = PyTree


@tree.define
class LinearTransform[I, O]:
    registry: ClassVar[Registry] = Registry(prefix="_")

    def wraps(self, method: str, wrapped: Callable[..., Any]) -> Callable[..., Any]:
        func_or_name: str | Callable = self.registry[method]
        func: Callable = (
            getattr(self, func_or_name)
            if isinstance(func_or_name, str)
            else func_or_name
        )
        return wrapt.decorator(func)(wrapped)

    def forward_primals(self, primals: I) -> O:
        raise NotImplementedError

    @compile_utils.jit(inline=True)
    def linearize(self, primals: I) -> tuple[O, Callable[[I], O]]:
        primals_out: O
        lin_fun: Callable[[I], O]
        primals_out, lin_fun = jax.linearize(self.forward_primals, primals)
        return primals_out, lin_fun

    @compile_utils.jit(inline=True)
    def linear_transpose(self, primals: I) -> Callable[[O], I]:
        return jax.linear_transpose(self.forward_primals, primals)

    def forward_hess_diag(self, hess_diag: I) -> O:
        raise NotImplementedError

    def backward_params(self, primals_out: O) -> I:
        raise NotImplementedError

    def backward_hess_diag(self, hess_diag_out: O) -> I:
        raise NotImplementedError

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
        f_transpose: Callable[[O], I] = self.linear_transpose(primals)
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[0] = f_transpose(outputs[0])
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
        f_transpose: Callable[[O], I] = self.linear_transpose(primals)
        args[1] = lin_fun(args[1])
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[0] = f_transpose(outputs[0])
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
        f_transpose: Callable[[O], I] = self.linear_transpose(primals)
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))
        outputs[1] = f_transpose(outputs[1])
        return utils.unpack(outputs)
