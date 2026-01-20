# ruff: noqa: SLF001

from collections.abc import Callable, Iterable
from typing import Any

import jax.tree_util as jtu
import wrapt

type AuxData = None
type Children = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]


def _flatten_wrapt_partial_callable_object_proxy(
    obj: wrapt.PartialCallableObjectProxy,
) -> tuple[Children, AuxData]:
    children: Children = (obj.__wrapped__, obj._self_args, obj._self_kwargs)  # pyright: ignore[reportAttributeAccessIssue]
    return children, None


def _flatten_wrapt_partial_callable_object_proxy_with_keys(
    obj: wrapt.PartialCallableObjectProxy,
) -> tuple[list[tuple[Any, Any]], AuxData]:
    children_with_keys: list[tuple[Any, Any]] = [
        (jtu.GetAttrKey("__wrapped__"), obj.__wrapped__),  # pyright: ignore[reportAttributeAccessIssue]
        (jtu.GetAttrKey("_self_args"), obj._self_args),  # pyright: ignore[reportAttributeAccessIssue]
        (jtu.GetAttrKey("_self_kwargs"), obj._self_kwargs),  # pyright: ignore[reportAttributeAccessIssue]
    ]
    return children_with_keys, None


def _unflatten_wrapt_partial_callable_object_proxy(
    _aux: AuxData, children: Children
) -> wrapt.PartialCallableObjectProxy:
    wrapped: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    wrapped, args, kwargs = children
    return wrapt.PartialCallableObjectProxy(wrapped, *args, **kwargs)


def register_pytree_wrapt_partial() -> None:
    jtu.register_pytree_with_keys(
        wrapt.PartialCallableObjectProxy,
        flatten_with_keys=_flatten_wrapt_partial_callable_object_proxy_with_keys,
        unflatten_func=_unflatten_wrapt_partial_callable_object_proxy,  # pyright: ignore[reportArgumentType]
        flatten_func=_flatten_wrapt_partial_callable_object_proxy,
    )
