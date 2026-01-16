from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypedDict, Unpack, dataclass_transform, overload

import attrs

from liblaf import grapes

from ._field import array, container, field, static
from ._register_fieldz import register_fieldz


class DefineKwargs(TypedDict, total=False):
    these: dict[str, Any] | None
    repr: bool
    unsafe_hash: bool | None
    hash: bool | None
    init: bool
    slots: bool
    frozen: bool
    weakref_slot: bool
    str: bool
    auto_attribs: bool
    kw_only: bool
    cache_hash: bool
    auto_exc: bool
    eq: bool | None
    order: bool | None
    auto_detect: bool
    getstate_setstate: bool | None
    on_setattr: attrs._OnSetAttrArgType | None
    field_transformer: attrs._FieldTransformer | None
    match_args: bool

    register_pytree: bool


@overload
@dataclass_transform(field_specifiers=(attrs.field, array, container, field, static))
def define[T: type](cls: T, /, **kwargs: Unpack[DefineKwargs]) -> T: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field, array, container, field, static))
def define[T: type](
    cls: None = None, /, **kwargs: Unpack[DefineKwargs]
) -> Callable[[T], T]: ...
def define(cls: type | None = None, /, **kwargs: Unpack[DefineKwargs]) -> Any:
    if cls is None:
        return functools.partial(define, **kwargs)
    register_pytree: bool = kwargs.pop("register_pytree", True)
    cls: type = grapes.attrs.define(cls, **kwargs)  # pyright: ignore[reportCallIssue]
    if register_pytree:
        cls = register_fieldz(cls)
    return cls
