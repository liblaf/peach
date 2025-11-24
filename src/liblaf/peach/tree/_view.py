import functools

from jaxtyping import Array

from ._flatten import FlatDef, flatten


class TreeView[T]:
    name: str
    flat_def_name: str

    def __init__(self, flat: str | None = None, flat_def: str = "flat_def") -> None:
        if flat is not None:
            self.flat_name = flat
        self.flat_def_name = flat_def

    def __get__(self, instance: object, owner: type) -> T:
        value: Array = getattr(instance, self.flat_name)
        flat_def: FlatDef[T] = getattr(instance, self.flat_def_name)
        return flat_def.unflatten(value)

    def __set__(self, instance: object, tree: T) -> None:
        flat_def: FlatDef[T] | None = getattr(instance, self.flat_def_name, None)
        flat: Array
        if flat_def is None:
            flat, flat_def = flatten(tree)
            setattr(instance, self.flat_def_name, flat_def)
        else:
            flat = flat_def.flatten(tree)
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
    flat_def_name: str

    def __init__(self, tree: str | None = None, flat_def: str = "flat_def") -> None:
        if tree is not None:
            self.tree_name = tree
        self.flat_def_name = flat_def

    def __get__(self, instance: object, owner: type) -> Array:
        tree: T = getattr(instance, self.tree_name)
        flat_def: FlatDef[T] | None = getattr(instance, self.flat_def_name, None)
        flat: Array
        if flat_def is None:
            flat, flat_def = flatten(tree)
            setattr(instance, self.flat_def_name, flat_def)
        else:
            flat = flat_def.flatten(tree)
        return flat

    def __set__(self, instance: object, flat: Array) -> None:
        flat_def: FlatDef[T] = getattr(instance, self.flat_def_name)
        tree: T = flat_def.unflatten(flat)
        setattr(instance, self.tree_name, tree)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @functools.cached_property
    def tree_name(self) -> str:
        if self.name.endswith("_flat"):
            return self.name.removesuffix("_flat")
        return f"{self.name}_tree"
