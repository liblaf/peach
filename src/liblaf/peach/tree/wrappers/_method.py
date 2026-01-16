import types
from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Self, overload

import jax.tree_util as jtu
import wadler_lindig as wl
import wrapt

from liblaf import grapes

type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]
type Leaves = Iterable[Any]
type PyTreeDef = Any


@jtu.register_pytree_with_keys_class
class BoundMethodWrapper[S, **P, T](wrapt.CallableObjectProxy):
    __wrapped__: types.MethodType

    @overload
    def __init__(self, func: Callable[P, T]) -> None: ...
    @overload
    def __init__(self, func: Callable[Concatenate[S, P], T], instance: S) -> None: ...
    def __init__(
        self,
        func: Callable[P, T] | Callable[Concatenate[S, P], T],
        instance: S | None = None,
    ) -> None:
        wrapped: Callable = (
            func if instance is None else types.MethodType(func, instance)
        )
        super().__init__(wrapped)

    def __repr__(self) -> str:
        return grapes.pformat(self)

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        return (
            wl.TextDoc("<")
            + wl.TextDoc(type(self).__name__)
            + wl.TextDoc(" ")
            + wl.TextDoc(self.__func__.__qualname__)
            + wl.TextDoc(" of ")
            + wl.pdoc(self.__self__, **kwargs)
            + wl.TextDoc(">")
        )

    @property
    def __func__(self) -> Callable[Concatenate[S, P], T]:
        return self.__wrapped__.__func__

    @property
    def __self__(self) -> object:
        return self.__wrapped__.__self__

    def tree_flatten(self) -> tuple[Leaves, None]:
        children: Leaves = [self.__func__, self.__self__]
        return children, None

    def tree_flatten_with_keys(
        self,
    ) -> tuple[KeyLeafPairs, None]:
        children: KeyLeafPairs = [
            (jtu.GetAttrKey("__func__"), self.__func__),
            (jtu.GetAttrKey("__self__"), self.__self__),
        ]
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux: None, children: Leaves) -> Self:
        func: Callable[Concatenate[S, P], T]
        instance: S
        func, instance = children
        return cls(func.__get__(instance, type(instance)))


@jtu.register_pytree_with_keys_class
class MethodDescriptor[S: object, **P, T](wrapt.CallableObjectProxy):
    __wrapped__: Callable[Concatenate[S, P], T]

    @overload
    def __get__(self, instance: None, owner: type | None = None, /) -> Self: ...
    @overload
    def __get__(
        self, instance: S, owner: type | None = None, /
    ) -> BoundMethodWrapper[S, P, T]: ...
    def __get__(self, instance: S | None, owner: type | None = None) -> Any:
        if instance is None:
            return self
        return BoundMethodWrapper(self.__wrapped__, instance)

    def tree_flatten(self) -> tuple[Leaves, None]:
        children: Leaves = [self.__wrapped__]
        return children, None

    def tree_flatten_with_keys(self) -> tuple[KeyLeafPairs, None]:
        children: KeyLeafPairs = [
            (jtu.GetAttrKey("__wrapped__"), self.__wrapped__),
        ]
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux: None, children: Leaves) -> Self:
        wrapped: Callable[Concatenate[object, P], T]
        (wrapped,) = children
        return cls(wrapped)


def method[C: Callable](func: C) -> C:
    return MethodDescriptor(func)  # pyright: ignore[reportReturnType]
