from __future__ import annotations

import collections
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem


type _Value = str | Callable


class Registry(collections.UserDict[str, _Value]):
    prefix: str = ""
    suffix: str = ""

    def __init__(
        self,
        arg: SupportsKeysAndGetItem[str, _Value] | None = None,
        /,
        *,
        prefix: str = "",
        suffix: str = "",
        **kwargs: _Value,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        super().__init__(arg, **kwargs)

    @overload
    def __call__[C: Callable](self, func: C, *, name: str | None = None) -> C: ...
    @overload
    def __call__[C: Callable](
        self, func: None = None, *, name: str | None = None
    ) -> Callable[[C], C]: ...
    def __call__(
        self, func: Callable[..., Any] | None = None, *, name: str | None = None
    ) -> Callable[..., Any]:
        if func is None:
            return functools.partial(self, name=name)
        if name is None:
            name = self._make_name(func.__name__)
        self.data[name] = func
        return func

    @overload
    def method[C: Callable](self, func: C, *, name: str | None = None) -> C: ...
    @overload
    def method[C: Callable](
        self, func: None = None, *, name: str | None = None
    ) -> Callable[[C], C]: ...
    def method(
        self, func: Callable[..., Any] | None = None, *, name: str | None = None
    ) -> Callable[..., Any]:
        if func is None:
            return functools.partial(self.method, name=name)
        if name is None:
            name = self._make_name(func.__name__)
        self.data[name] = func.__name__
        return func

    def inherit(self, other: Mapping[str, str | Callable]) -> None:
        for name, func in other.items():
            self.data.setdefault(name, func)

    def _make_name(self, name: str) -> str:
        if not name.startswith(self.prefix):
            msg: str = f'"{name}" name must start with "{self.prefix}"'
            raise ValueError(msg)
        if not name.endswith(self.suffix):
            msg: str = f'"{name}" name must end with "{self.suffix}"'
            raise ValueError(msg)
        name = name[len(self.prefix) :]
        name = name[: -len(self.suffix)]
        return name
