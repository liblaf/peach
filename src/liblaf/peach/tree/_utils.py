import inspect
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import attrs
import equinox as eqx
import jax
import jax.tree_util as jtu
from environs import env
from liblaf.grapes import warnings

type PyTreeDef = Any

_WARN_STATIC_LEAF: bool = env.bool("WARN_STATIC_LEAF", True)


@attrs.frozen
class AuxData:
    static_leaves: tuple[Any, ...] = attrs.field(converter=tuple)
    treedef: Any


def combine(dynamic_leaves: Iterable[Any | None], aux: AuxData) -> Any:
    leaves: list[Any] = combine_leaves(dynamic_leaves, aux.static_leaves)
    return jax.tree.unflatten(aux.treedef, leaves)


def combine_leaves[T](
    dynamic_leaves: Iterable[T | None], static_leaves: Iterable[T | None]
) -> list[T]:
    leaves: list[T] = [
        static_leaf if dynamic_leaf is None else dynamic_leaf
        for dynamic_leaf, static_leaf in zip(dynamic_leaves, static_leaves, strict=True)
    ]  # pyright: ignore[reportAssignmentType]
    return leaves


def partition(obj: Any) -> tuple[list[Any | None], AuxData]:
    _warnings_hide = True
    leaves: Iterable[Any]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(obj)
    dynamic_leaves: list[Any | None]
    static_leaves: list[Any | None]
    dynamic_leaves, static_leaves = partition_leaves(leaves)
    aux = AuxData(static_leaves=static_leaves, treedef=treedef)
    return dynamic_leaves, aux


def partition_leaves[T](
    leaves: Iterable[T], *, pred: Callable[[T], bool] = eqx.is_array
) -> tuple[list[T | None], list[T | None]]:
    _warnings_hide = True
    dynamic_leaves: list[T | None] = []
    static_leaves: list[T | None] = []
    for leaf in leaves:
        if pred(leaf):
            dynamic_leaves.append(leaf)
            static_leaves.append(None)
        else:
            if _WARN_STATIC_LEAF:
                _warn_static_leaf(leaf)
            dynamic_leaves.append(None)
            static_leaves.append(leaf)
    return dynamic_leaves, static_leaves


def partition_leaves_with_path[T](
    leaves_with_path: Iterable[tuple[jtu.KeyPath, T]],
    *,
    pred: Callable[[T], bool] = eqx.is_array,
) -> tuple[list[tuple[jtu.KeyPath, T | None]], list[T | None]]:
    dynamic_leaves_with_path: list[tuple[jtu.KeyPath, T | None]] = []
    static_leaves: list[T | None] = []
    for path, leaf in leaves_with_path:
        if pred(leaf):
            dynamic_leaves_with_path.append((path, leaf))
            static_leaves.append(None)
        else:
            if _WARN_STATIC_LEAF:
                _warn_static_leaf(leaf)
            dynamic_leaves_with_path.append((path, None))
            static_leaves.append(leaf)
    return dynamic_leaves_with_path, static_leaves


def _warn_static_leaf(leaf: Any) -> None:
    _warnings_hide = True
    if leaf is None:
        return
    if not isinstance(leaf, Hashable):
        warnings.warn(
            f"Static leaf of type {type(leaf)} is not hashable. "
            "JAX requires static auxiliary data to be hashable for caching. "
            "Consider registering this type as a pytree using jax.tree_util.register_pytree_node.",
        )
        return
    if isinstance(leaf, (int, float, complex, str, bytes)):
        return
    if inspect.isbuiltin(leaf) or inspect.isfunction(leaf):
        return
    if inspect.ismethod(leaf):
        warnings.warn(
            f"Static leaf is a bound method {leaf}. "
            "Bound methods have hashes based on the method itself, not the instance they're bound to. "
            "This means the hash won't change when the bound instance changes, which can cause caching issues. "
            "Use `@tree.method` decorator on the method definition to ensure proper behavior.",
        )
        return
    cls = type(leaf)
    if cls.__eq__ is object.__eq__ and cls.__hash__ is object.__hash__:
        warnings.warn(
            f"Static leaf of type {cls} uses default object.__hash__ based on id(). "
            "This means the hash won't change when the object is modified, which can lead to stale cache entries. "
            "Consider registering this type as a pytree using jax.tree_util.register_pytree_node to properly track its internal state.",
        )
        return
