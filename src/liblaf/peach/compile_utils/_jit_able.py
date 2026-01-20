import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import wrapt

_JIT_ABLE_ATTRS: Sequence[str] = ("_jit_able", "_self_jit_able")


@functools.singledispatch
def is_jit_able(fun: Callable[..., Any]) -> bool:
    result: bool | None = _get_jit_able_attr(fun)
    if result is not None:
        return result
    wrapped: Callable[..., Any] = inspect.unwrap(fun)
    if wrapped is not fun:
        return is_jit_able(wrapped)
    return True


@is_jit_able.register(wrapt.FunctionWrapper)
def _(fun: wrapt.FunctionWrapper) -> bool:
    return (
        getattr(fun, "_self_jit_able", True)
        and is_jit_able(fun._self_wrapper)  # noqa: SLF001
        and is_jit_able(fun.__wrapped__)
    )


@is_jit_able.register(wrapt.BoundFunctionWrapper)
def _(fun: wrapt.BoundFunctionWrapper) -> bool:
    return (
        getattr(fun, "_self_jit_able", True)
        and is_jit_able(fun.__wrapped__)
        and is_jit_able(fun._self_parent)  # noqa: SLF001
        and is_jit_able(fun._self_wrapper)  # noqa: SLF001
    )


def _get_jit_able_attr(fun: Callable[..., Any]) -> bool | None:
    for name in _JIT_ABLE_ATTRS:
        result: bool | None = getattr(fun, name, None)
        if result is not None:
            return result
    return None


def not_jit_able[C: Callable](fun: C) -> C:
    fun._jit_able = False  # noqa: SLF001 # pyright: ignore[reportFunctionMemberAccess]
    return fun
