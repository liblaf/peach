from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack, overload

import attrs
import toolz
from jaxtyping import Array, ArrayLike
from liblaf.grapes import wraps

from liblaf.peach.tree import converters


class FieldKwargs[T](TypedDict, total=False):
    default: T
    validator: attrs._ValidatorArgType[T] | None
    repr: attrs._ReprArgType
    hash: bool | None
    init: bool
    metadata: Mapping[Any, Any] | None
    converter: (
        attrs._ConverterType
        | list[attrs._ConverterType]
        | tuple[attrs._ConverterType, ...]
        | None
    )
    factory: Callable[[], T] | None
    kw_only: bool | None
    eq: attrs._EqOrderType | None
    order: attrs._EqOrderType | None
    on_setattr: attrs._OnSetAttrArgType | None
    alias: str | None
    type: type | None


def array(**kwargs: Unpack[FieldKwargs[ArrayLike | None]]) -> Array:
    kwargs.setdefault("converter", converters.asarray)
    return field(**kwargs)  # pyright: ignore[reportReturnType]


@wraps(attrs.field)
def container(**kwargs) -> Any:
    if "converter" in kwargs and "factory" not in kwargs:
        kwargs["factory"] = kwargs["converter"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" in kwargs:
        kwargs["converter"] = kwargs["factory"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" not in kwargs:
        kwargs["converter"] = _dict_if_none
        kwargs["factory"] = dict
    return field(**kwargs)


@wraps(attrs.field)
def field(**kwargs) -> Any:
    if "default_factory" in kwargs:
        kwargs.setdefault("factory", kwargs.pop("default_factory"))
    if kwargs.pop("static", False):
        kwargs["metadata"] = toolz.assoc(kwargs.get("metadata") or {}, "static", True)  # noqa: FBT003
    return attrs.field(**kwargs)  # pyright: ignore[reportCallIssue]


@wraps(attrs.field)
def static(**kwargs) -> Any:
    kwargs.setdefault("static", True)
    return field(**kwargs)  # pyright: ignore[reportCallIssue]


@overload
def _dict_if_none(value: None) -> dict: ...
@overload
def _dict_if_none[T](value: T) -> T: ...
def _dict_if_none(value: Any) -> Any:
    if value is None:
        return {}
    return value
