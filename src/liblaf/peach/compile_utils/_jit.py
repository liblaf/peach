from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any, TypedDict, Unpack, overload

import jax
import wrapt

from liblaf.peach import tree
from liblaf.peach.tree import PyTreeProxy

from ._jit_able import is_jit_able

type Params = tuple[tuple[Any, ...], dict[str, Any]]
type InnerCallable[T] = Callable[[PyTreeProxy[Params], Iterable[Any]], PyTreeProxy[T]]


class JitKwargs(TypedDict, total=False):
    inline: bool


@overload
def jit[**P, T](fun: Callable[P, T], **kwargs: Unpack[JitKwargs]) -> Callable[P, T]: ...
@overload
def jit[**P, T](
    **kwargs: Unpack[JitKwargs],
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def jit[**P, T](fun: Callable[P, T] | None = None, **kwargs) -> Callable:
    _warnings_hide = True
    if fun is None:
        return functools.partial(jit, **kwargs)
    if not is_jit_able(fun):
        return fun
    fun_dynamic: Iterable[Any]
    fun_static: tree.AuxData
    fun_dynamic, fun_static = tree.partition(fun)
    inner: _DynamicInnerWrapper = _inner_wrapper(fun_static)
    jit_wrapped = jax.jit(inner, **kwargs)
    functools.update_wrapper(jit_wrapped, fun)
    outer = _DynamicOuterWrapper(fun_dynamic)(jit_wrapped)
    # intentionally don't `functools.update_wrapper(outer, fun)` here, as it would overwrite `__wrapped__`
    tree.update_wrapper(outer, fun)
    return outer


@tree.frozen
class _DynamicInnerWrapper[T]:
    fun_static: tree.AuxData

    def __call__(self, *args: Any) -> PyTreeProxy[T]:
        __tracebackhide__ = True
        instance: Any = None
        inputs: PyTreeProxy[Params]
        fun_dynamic: Iterable[Any]
        if len(args) == 2:
            inputs, fun_dynamic = args
        elif len(args) == 3:
            # I don't know why this happens, but this works around it.
            # Maybe something wrong with method binding in JAX or wrapt?
            instance, inputs, fun_dynamic = args
        else:
            raise TypeError(args)
        fun: Callable[..., T] = tree.combine(fun_dynamic, self.fun_static)
        args: tuple[Any, ...]
        kwargs: dict[str, Any]
        args, kwargs = inputs.__wrapped__
        if instance is not None:
            args = (instance, *args)
        output: T = fun(*args, **kwargs)
        return PyTreeProxy(output)


@tree.define
class _DynamicOuterWrapper:
    fun_dynamic: Iterable[Any]

    @wrapt.decorator
    def __call__(
        self,
        wrapped: Callable,
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        __tracebackhide__ = True
        inputs: PyTreeProxy[Params] = PyTreeProxy((args, kwargs))
        outputs: PyTreeProxy[Any] = wrapped(inputs, self.fun_dynamic)
        return outputs.__wrapped__


# Cache inner wrappers to preserve JAX's compiled function cache. JAX caches
# compiled functions internally, but discards them when inner wrappers are
# garbage collected. By caching wrappers with lru_cache, we keep them alive
# across calls, avoiding redundant recompilation.
#
# We use lru_cache instead of WeakKeyDictionary because the inner wrapper holds
# a reference to the static function part (used as cache key), creating a
# circular reference that prevents garbage collection.
@functools.lru_cache
def _inner_wrapper(fun_static: tree.AuxData) -> _DynamicInnerWrapper:
    return _DynamicInnerWrapper(fun_static)
