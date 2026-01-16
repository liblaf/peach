from collections.abc import Iterable
from typing import Any, Self

import jax
import jax.tree_util as jtu
import wrapt

from liblaf.peach.tree import _utils

type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]
type Leaves = Iterable[Any]
type PyTreeDef = Any


@jtu.register_pytree_with_keys_class
class BaseObjectProxy[T](wrapt.BaseObjectProxy):
    def tree_flatten(self) -> tuple[Leaves, _utils.AuxData]:
        _warnings_hide = True
        leaves: Leaves
        treedef: PyTreeDef
        leaves, treedef = jax.tree.flatten(self.__wrapped__)
        dynamic_leaves: list[Any]
        static_leaves: list[Any]
        dynamic_leaves, static_leaves = _utils.partition_leaves(leaves)
        aux = _utils.AuxData(static_leaves=static_leaves, treedef=treedef)
        return dynamic_leaves, aux

    def tree_flatten_with_keys(self) -> tuple[Iterable[KeyLeafPair], _utils.AuxData]:
        _warnings_hide = True
        leaves_with_path: Iterable[KeyLeafPair]
        treedef: PyTreeDef
        leaves_with_path, treedef = jax.tree.flatten_with_path(self.__wrapped__)
        dynamic_leaves_with_path: list[KeyLeafPair]
        static_leaves: list[Any]
        dynamic_leaves_with_path, static_leaves = _utils.partition_leaves_with_path(
            leaves_with_path
        )
        aux = _utils.AuxData(static_leaves=static_leaves, treedef=treedef)
        return dynamic_leaves_with_path, aux

    @classmethod
    def tree_unflatten(cls, aux: _utils.AuxData, children: Leaves) -> Self:
        leaves: list[Any] = _utils.combine_leaves(children, aux.static_leaves)
        wrapped: Any = jax.tree.unflatten(aux.treedef, leaves)
        return cls(wrapped)
