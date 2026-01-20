# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Self

import jax.tree_util as jtu
import wrapt

from liblaf.peach.tree._define import frozen

type AuxData = None
type Children[**P, R] = tuple[
    wrapt.WrappedFunction[P, R], Any, wrapt.WrapperFunction[P, R], Any, Any
]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]
type PyTreeDef = Any


@frozen
class _FunctionWrapperAuxData:
    enabled: bool | wrapt.Boolean | Callable[[], bool] | None
    binding: str

    @classmethod
    def from_wrapper(
        cls, wrapper: wrapt.FunctionWrapper | wrapt.BoundFunctionWrapper
    ) -> Self:
        return cls(enabled=wrapper._self_enabled, binding=wrapper._self_binding)


def _flatten_function_wrapper[**P, R](
    obj: wrapt.FunctionWrapper[P, R] | wrapt.BoundFunctionWrapper[P, R],
) -> tuple[Children[P, R], _FunctionWrapperAuxData]:
    children: Children[P, R] = (
        obj.__wrapped__,
        obj._self_instance,
        obj._self_wrapper,
        obj._self_parent,
        obj._self_owner,
    )
    aux: _FunctionWrapperAuxData = _FunctionWrapperAuxData.from_wrapper(obj)
    return children, aux


def _flatten_function_wrapper_with_keys[**P, R](
    obj: wrapt.FunctionWrapper[P, R] | wrapt.BoundFunctionWrapper[P, R],
) -> tuple[KeyLeafPairs, _FunctionWrapperAuxData]:
    children_with_keys: KeyLeafPairs = [
        (jtu.GetAttrKey("__wrapped__"), obj.__wrapped__),
        (jtu.GetAttrKey("_self_instance"), obj._self_instance),
        (jtu.GetAttrKey("_self_wrapper"), obj._self_wrapper),
        (jtu.GetAttrKey("_self_parent"), obj._self_parent),
        (jtu.GetAttrKey("_self_owner"), obj._self_owner),
    ]
    aux: _FunctionWrapperAuxData = _FunctionWrapperAuxData.from_wrapper(obj)
    return children_with_keys, aux


def _unflatten_function_wrapper[**P, R](
    aux: _FunctionWrapperAuxData, children: Children[P, R]
) -> wrapt.FunctionWrapper[P, R]:
    wrapped: wrapt.WrappedFunction[P, R]
    wrapper: wrapt.WrapperFunction[P, R]
    (wrapped, _instance, wrapper, _parent, _owner) = children
    obj: wrapt.FunctionWrapper[P, R] = wrapt.FunctionWrapper(
        wrapped, wrapper, aux.enabled
    )
    return obj


def _unflatten_bound_function_wrapper[**P, R](
    aux: _FunctionWrapperAuxData, children: Children[P, R]
) -> wrapt.BoundFunctionWrapper[P, R]:
    wrapped: wrapt.WrappedFunction[P, R]
    instance: Any
    wrapper: wrapt.WrapperFunction[P, R]
    parent: Any
    owner: Any
    (wrapped, instance, wrapper, parent, owner) = children
    obj: wrapt.BoundFunctionWrapper[P, R] = wrapt.BoundFunctionWrapper(
        wrapped,
        instance,  # pyright: ignore[reportCallIssue]
        wrapper,
        aux.enabled,
        aux.binding,
        parent,
        owner,
    )
    return obj


def register_pytree_wrapt_function_wrapper() -> None:
    jtu.register_pytree_with_keys(
        wrapt.FunctionWrapper,
        flatten_with_keys=_flatten_function_wrapper_with_keys,
        unflatten_func=_unflatten_function_wrapper,  # pyright: ignore[reportArgumentType]
        flatten_func=_flatten_function_wrapper,
    )
    jtu.register_pytree_with_keys(
        wrapt.BoundFunctionWrapper,
        flatten_with_keys=_flatten_function_wrapper_with_keys,
        unflatten_func=_unflatten_bound_function_wrapper,  # pyright: ignore[reportArgumentType]
        flatten_func=_flatten_function_wrapper,
    )
