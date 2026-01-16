from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any, overload

import attrs
import jax
import wrapt

from liblaf.peach.tree import _utils
from liblaf.peach.tree._utils import AuxData
from liblaf.peach.tree.wrappers import BaseObjectProxy, BoundMethodWrapper

type Params = tuple[tuple[Any, ...], dict[str, Any]]
type InnerCallable[T] = Callable[
    [BaseObjectProxy[Params], Iterable[Any]], BaseObjectProxy[T]
]


def jit_dynamic[**P, T](fun: Callable[P, T], **kwargs) -> Callable[P, T]:
    _warnings_hide = True
    fun_dynamic: Iterable[Any]
    fun_static: AuxData
    fun_dynamic, fun_static = _utils.partition(fun)
    inner = _inner_wrapper(fun_static)
    jit_wrapped = jax.jit(inner, **kwargs)
    functools.update_wrapper(jit_wrapped, fun)
    outer = _DynamicOuterWrapper(jit_wrapped, fun_dynamic)
    # intentionally don't `update_wrapper(outer, fun)` here, as it would overwrite `__wrapped__`
    return outer


@attrs.frozen
class _DynamicInnerWrapper[T]:
    fun_static: AuxData

    def __call__(
        self, inputs: BaseObjectProxy[Params], fun_dynamic: Iterable[Any]
    ) -> BaseObjectProxy[T]:
        __tracebackhide__ = True
        fun: Callable[..., T] = _utils.combine(fun_dynamic, self.fun_static)
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        args, kwargs = inputs.__wrapped__
        output: T = fun(*args, **kwargs)
        return BaseObjectProxy(output)


class _DynamicOuterWrapper[**P, T](wrapt.CallableObjectProxy):
    _self_fun_dynamic: Iterable[Any]

    def __init__(self, wrapped: InnerCallable[T], fun_dynamic: Iterable[Any]) -> None:
        super().__init__(wrapped)
        self._self_fun_dynamic = fun_dynamic

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        __tracebackhide__ = True
        inputs: BaseObjectProxy[Params] = BaseObjectProxy((args, kwargs))
        outputs: BaseObjectProxy[T] = self.__wrapped__(inputs, self._self_fun_dynamic)
        return outputs.__wrapped__

    @overload
    def __get__(
        self, instance: None, owner: type[Any]
    ) -> _DynamicOuterWrapper[P, T]: ...
    @overload
    def __get__(self, instance: Any, owner: type[Any]) -> Callable[..., T]: ...
    def __get__(self, instance: object | None, owner: type | None = None) -> Callable:
        if instance is None:
            return self
        return BoundMethodWrapper(self, instance)  # pyright: ignore[reportArgumentType]


# Cache inner wrappers to preserve JAX's compiled function cache. JAX caches
# compiled functions internally, but discards them when inner wrappers are
# garbage collected. By caching wrappers with lru_cache, we keep them alive
# across calls, avoiding redundant recompilation.
#
# We use lru_cache instead of WeakKeyDictionary because the inner wrapper holds
# a reference to the static function part (used as cache key), creating a
# circular reference that prevents garbage collection.
@functools.lru_cache
def _inner_wrapper(fun_static: AuxData) -> _DynamicInnerWrapper:
    return _DynamicInnerWrapper(fun_static)
