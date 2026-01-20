import types
from collections.abc import Callable, Iterable
from typing import Any

import jax.tree_util as jtu

type AuxData = None
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]
type Children = tuple[Callable[..., Any], object]
type PyTreeDef = Any


def _flatten_method(obj: types.MethodType) -> tuple[Children, AuxData]:
    children: Children = (obj.__func__, obj.__self__)
    return children, None


def _flatten_method_with_keys(obj: types.MethodType) -> tuple[KeyLeafPairs, AuxData]:
    children: KeyLeafPairs = [
        (jtu.GetAttrKey("__func__"), obj.__func__),
        (jtu.GetAttrKey("__self__"), obj.__self__),
    ]
    return children, None


def _unflatten_method(_aux: AuxData, children: Children) -> types.MethodType:
    func: Callable[..., Any]
    instance: object
    func, instance = children
    return func.__get__(instance, type(instance))


def register_pytree_method() -> None:
    jtu.register_pytree_with_keys(
        types.MethodType,
        flatten_with_keys=_flatten_method_with_keys,
        unflatten_func=_unflatten_method,  # pyright: ignore[reportArgumentType]
        flatten_func=_flatten_method,
    )
