import functools
from collections.abc import Callable

from jaxtyping import Array

from ._flatten import Unflatten, flatten


class TreeView[T]:
    name: str
    unflatten_name: str

    def __init__(self, flat: str | None = None, unflatten: str = "unflatten") -> None:
        if flat is not None:
            self.flat_name = flat
        self.unflatten_name = unflatten

    def __get__(self, instance: object, owner: type) -> T:
        value: Array = getattr(instance, self.flat_name)
        unflatten: Unflatten[T] | None = getattr(instance, self.unflatten_name, None)
        if unflatten is None:
            return value  # pyright: ignore[reportReturnType]
        return unflatten(value)

    def __set__(self, instance: object, tree: T) -> None:
        unflatten: Unflatten[T] | None = getattr(instance, self.unflatten_name, None)
        flat: Array
        if unflatten is None:
            flat, unflatten = flatten(tree)
            setattr(instance, self.unflatten_name, unflatten)
        else:
            flat = unflatten.flatten(tree)
        setattr(instance, self.flat_name, flat)

    def __set_name__(self, owner: type, name: str) -> None:
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

    def __get__(self, instance: object, owner: type) -> Array:
        tree: T = getattr(instance, self.tree_name)
        unflatten: Unflatten[T] | None = getattr(instance, self.unflatten_name, None)
        flat: Array
        if unflatten is None:
            flat, unflatten = flatten(tree)
            setattr(instance, self.unflatten_name, unflatten)
        else:
            flat = unflatten.flatten(tree)
        return flat

    def __set__(self, instance: object, flat: Array) -> None:
        unflatten: Callable[[Array], T] | None = getattr(
            instance, self.unflatten_name, None
        )
        if unflatten is None:
            setattr(instance, self.tree_name, flat)
            return
        tree: T = unflatten(flat)
        setattr(instance, self.tree_name, tree)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @functools.cached_property
    def tree_name(self) -> str:
        if self.name.endswith("_flat"):
            return self.name.removesuffix("_flat")
        return f"{self.name}_tree"
