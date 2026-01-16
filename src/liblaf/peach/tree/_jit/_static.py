import functools
from collections.abc import Callable
from typing import Any

import jax
import wrapt

from liblaf.peach.tree.wrappers import BaseObjectProxy

from ._utils import update_wrapper

type StaticInnerParams = tuple[tuple[Any, ...], dict[str, Any]]
type JitWrapped[T] = Callable[[BaseObjectProxy[StaticInnerParams]], BaseObjectProxy[T]]


def jit_static[**P, T](fun: Callable[P, T], **kwargs) -> Callable[P, T]:
    inner: _StaticInnerWrapper[T] = _inner_wrapper(fun)
    jit_wrapped: JitWrapped[T] = jax.jit(inner, **kwargs)
    outer: _StaticOuterWrapper[P, T] = _StaticOuterWrapper(jit_wrapped)
    update_wrapper(outer, fun)
    return outer


class _StaticInnerWrapper[T](wrapt.CallableObjectProxy):
    def __call__(
        self, inputs: BaseObjectProxy[StaticInnerParams]
    ) -> BaseObjectProxy[T]:
        __tracebackhide__ = True
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        args, kwargs = inputs.__wrapped__
        output: T = self.__wrapped__(*args, **kwargs)
        return BaseObjectProxy(output)


class _StaticOuterWrapper[**P, T](wrapt.CallableObjectProxy):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        __tracebackhide__ = True
        inputs: BaseObjectProxy[StaticInnerParams] = BaseObjectProxy((args, kwargs))
        outputs: BaseObjectProxy[T] = self.__wrapped__(inputs)
        return outputs.__wrapped__


# JAX caches compiled functions internally. When an inner wrapper is garbage
# collected, JAX discards its cached compilation, forcing recompilation on the
# next call. Using functools.lru_cache keeps inner wrappers alive across calls,
# preserving JAX's compilation cache and avoiding redundant compilations.
#
# We avoid WeakKeyDictionary because the inner wrapper references the original
# function, which we use as a cache key. This creates a circular reference that
# prevents garbage collection and defeats the purpose of weak references.
#
# The lru_cache approach balances three goals:
# 1. Preserve JAX's compilation cache by keeping wrappers alive
# 2. Control memory usage with a bounded cache size
# 3. Enable garbage collection of old wrappers when the cache reaches capacity
@functools.lru_cache
def _inner_wrapper[**P, T](fun: Callable[P, T]) -> _StaticInnerWrapper[T]:
    return _StaticInnerWrapper(fun)
