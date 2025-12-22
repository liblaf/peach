import functools

from jaxtyping import Array

from ._flatten import Structure, flatten


class TreeView[T]:
    name: str
    structure_name: str

    def __init__(self, flat: str | None = None, structure: str = "structure") -> None:
        if flat is not None:
            self.flat_name = flat
        self.structure_name = structure

    def __get__(self, instance: object, owner: type) -> T:
        value: Array = getattr(instance, self.flat_name)
        structure: Structure[T] = getattr(instance, self.structure_name)
        return structure.unflatten(value)

    def __set__(self, instance: object, tree: T) -> None:
        structure: Structure[T] | None = getattr(instance, self.structure_name, None)
        flat: Array
        if structure is None:
            flat, structure = flatten(tree)
            setattr(instance, self.structure_name, structure)
        else:
            flat = structure.flatten(tree)
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
    structure_name: str

    def __init__(self, tree: str | None = None, structure: str = "structure") -> None:
        if tree is not None:
            self.tree_name = tree
        self.structure_name = structure

    def __get__(self, instance: object, owner: type) -> Array:
        tree: T = getattr(instance, self.tree_name)
        structure: Structure[T] | None = getattr(instance, self.structure_name, None)
        flat: Array
        if structure is None:
            flat, structure = flatten(tree)
            setattr(instance, self.structure_name, structure)
        else:
            flat = structure.flatten(tree)
        return flat

    def __set__(self, instance: object, flat: Array) -> None:
        structure: Structure[T] = getattr(instance, self.structure_name)
        tree: T = structure.unflatten(flat)
        setattr(instance, self.tree_name, tree)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @functools.cached_property
    def tree_name(self) -> str:
        if self.name.endswith("_flat"):
            return self.name.removesuffix("_flat")
        return f"{self.name}_tree"
