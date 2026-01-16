from collections.abc import Generator, Iterable
from typing import Any, Self

import attrs
import fieldz
import jax.tree_util as jtu

type AuxData = tuple[Any, ...]
type Children = Iterable[Any]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]


@attrs.define
class _Flattener[T]:
    cls: type[T]
    data_fields: tuple[str, ...] = ()
    meta_fields: tuple[str, ...] = ()

    @classmethod
    def from_cls(
        cls,
        nodetype: type[T],
        data_fields: Iterable[str] | None = None,
        meta_fields: Iterable[str] | None = None,
    ) -> Self:
        if data_fields is None:
            data_fields = _filter_fields(nodetype, static=False)
        if meta_fields is None:
            meta_fields = _filter_fields(nodetype, static=True)
        return cls(
            cls=nodetype, data_fields=tuple(data_fields), meta_fields=tuple(meta_fields)
        )

    def flatten_with_keys(self, obj: Any) -> tuple[KeyLeafPairs, AuxData]:
        children: KeyLeafPairs = tuple(
            (jtu.GetAttrKey(name), getattr(obj, name)) for name in self.data_fields
        )
        aux: AuxData = tuple(getattr(obj, name) for name in self.meta_fields)
        return children, aux

    def flatten(self, obj: Any) -> tuple[Children, AuxData]:
        children: Children = tuple(getattr(obj, name) for name in self.data_fields)
        aux: AuxData = tuple(getattr(obj, name) for name in self.meta_fields)
        return children, aux

    def unflatten(self, aux: AuxData, children: Children) -> T:
        obj: T = object.__new__(self.cls)
        for name, value in zip(self.meta_fields, aux, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in zip(self.data_fields, children, strict=True):
            object.__setattr__(obj, name, value)
        return obj


def register_fieldz[T: type](
    cls: T,
    data_fields: Iterable[str] | None = None,
    meta_fields: Iterable[str] | None = None,
) -> T:
    flattener: _Flattener[T] = _Flattener.from_cls(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )
    jtu.register_pytree_with_keys(
        cls,
        flatten_with_keys=flattener.flatten_with_keys,
        unflatten_func=flattener.unflatten,
        flatten_func=flattener.flatten,
    )
    return cls


def _filter_fields(cls: type, *, static: bool) -> Generator[str]:
    for field in fieldz.fields(cls):
        if field.metadata.get("static", False) == static:
            yield field.name
