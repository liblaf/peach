from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any, TypedDict, Unpack, overload

import jax

from liblaf.peach import tree
from liblaf.peach.tree import register_pytree_prelude

from ._wrapper import Inner, Outer, wraps_inner


class JitKwargs(TypedDict, total=False):
    inline: bool
    proxy: bool


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
    proxy: bool = kwargs.pop("proxy", False)
    if not proxy:
        return jax.jit(fun, **kwargs)
    fun_dynamic: Iterable[Any]
    fun_static: tree.AuxData
    fun_dynamic, fun_static = tree.partition(fun)
    inner: Inner = wraps_inner(fun_static)
    jit_wrapped = jax.jit(inner, **kwargs)
    functools.update_wrapper(jit_wrapped, fun)
    outer = Outer(fun_dynamic)(jit_wrapped)
    # intentionally don't `functools.update_wrapper(outer, fun)` here, as it would overwrite `__wrapped__`
    tree.update_wrapper(outer, fun)
    return outer


register_pytree_prelude()
