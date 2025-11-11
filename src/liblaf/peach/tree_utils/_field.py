from typing import Any, overload

import attrs
import jax.numpy as jnp
import toolz
from jaxtyping import Array, ArrayLike
from liblaf.grapes import wraps


@wraps(attrs.field)
def array(**kwargs) -> Any:
    kwargs.setdefault("converter", _optional_as_array)
    return field(**kwargs)


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


@overload
def _optional_as_array(value: None) -> None: ...
@overload
def _optional_as_array(value: ArrayLike) -> Array: ...
def _optional_as_array(value: Any) -> Any:
    if value is None:
        return None
    return jnp.asarray(value)
