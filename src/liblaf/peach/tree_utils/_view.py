import functools
from collections.abc import Callable
from typing import Any

from jaxtyping import Array

from liblaf.peach import tree_utils


class TreeView[T]:
    name: str
    unflatten_name: str

    def __init__(self, flat: str | None = None, unflatten: str = "unflatten") -> None:
        if flat is not None:
            self.flat_name = flat
        self.unflatten_name = unflatten

    def __get__(self, instance: Any, owner: Any, /) -> T:
        value: Array = getattr(instance, self.flat_name)
        unflatten: Callable[[Array], T] | None = getattr(
            instance, self.unflatten_name, None
        )
        if unflatten is None:
            return value  # pyright: ignore[reportReturnType]
        return unflatten(value)

    def __set__(self, instance: Any, value: T) -> None:
        flat: Array
        unflatten: Callable[[Array], T]
        flat, unflatten = tree_utils.flatten(value)
        setattr(instance, self.flat_name, flat)
        setattr(instance, self.unflatten_name, unflatten)

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name

    @functools.cached_property
    def flat_name(self) -> str:
        if self.name.endswith("_tree"):
            return self.name.removesuffix("_tree")
        return f"{self.name}_flat"


class FlatView[T]:
    name: str
    unflatten_name: str

    def __init__(self, tree: str | None = None, unflatten: str = "unflatten") -> None:
        if tree is not None:
            self.tree_name = tree
        self.unflatten_name = unflatten

    def __get__(self, instance: Any, owner: Any, /) -> Array:
        tree: T = getattr(instance, self.tree_name)
        flat: Array
        unflatten: Callable[[Array], T]
        flat, unflatten = tree_utils.flatten(tree)
        setattr(instance, self.unflatten_name, unflatten)
        return flat

    def __set__(self, instance: Any, value: Array) -> None:
        unflatten: Callable[[Array], T] | None = getattr(
            instance, self.unflatten_name, None
        )
        if unflatten is None:
            setattr(instance, self.tree_name, value)
            return
        tree: T = unflatten(value)
        setattr(instance, self.tree_name, tree)

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name

    @functools.cached_property
    def tree_name(self) -> str:
        if self.name.endswith("_flat"):
            return self.name.removesuffix("_flat")
        return f"{self.name}_tree"
