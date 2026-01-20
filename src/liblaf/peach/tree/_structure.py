import functools
from collections.abc import Iterable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, DTypeLike

from . import _utils
from ._define import frozen
from ._field_specifiers import static as static_field

type PyTreeDef = Any
type Shape = tuple[int, ...]

_SINGLE_TREEDEF: PyTreeDef = jax.tree.structure(jnp.empty(()))


@frozen
class Structure[T]:
    dtype: DTypeLike = static_field()
    offsets: tuple[int, ...] = static_field()
    shapes: tuple[Shape | None, ...] = static_field()
    static_leaves: tuple[Any, ...] = static_field()
    treedef: PyTreeDef = static_field()

    @functools.cached_property
    def is_single_array(self) -> bool:
        return self.treedef == _SINGLE_TREEDEF

    def flatten(self, tree: T | Array) -> Array:
        if eqx.is_array(tree):
            tree: Array = cast("Array", tree)
            return jnp.ravel(tree)
        leaves: list[Any]
        leaves, _ = jax.tree.flatten(tree)
        dynamic_leaves: list[Array | None]
        dynamic_leaves, _ = _utils.partition_leaves(leaves)
        return _ravel(dynamic_leaves)

    def unflatten(self, flat: T | ArrayLike, dtype: DTypeLike | None = None) -> T:
        if not eqx.is_array(flat):
            return cast("T", flat)  # flat is already a PyTree
        flat: Array = jnp.asarray(flat, self.dtype if dtype is None else dtype)
        if self.is_single_array:
            return cast("T", jnp.reshape(flat, self.shapes[0]))
        dynamic_leaves: list[Array | None] = _unravel(flat, self.offsets, self.shapes)
        leaves: list[Any] = _utils.combine_leaves(dynamic_leaves, self.static_leaves)
        return jax.tree.unflatten(self.treedef, leaves)


def flatten[T](tree: T) -> tuple[Array, Structure[T]]:
    leaves: list[Any]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(tree)
    dynamic_leaves: list[Any | None]
    static_leaves: list[Any | None]
    dynamic_leaves, static_leaves = _utils.partition_leaves(leaves)
    structure: Structure[T] = Structure(
        offsets=_offsets_from_leaves(dynamic_leaves),
        shapes=_shapes_from_leaves(dynamic_leaves),
        static_leaves=tuple(static_leaves),
        treedef=treedef,
        dtype=_dtype_from_leaves(dynamic_leaves),
    )
    flat: Array = _ravel(dynamic_leaves)
    return flat, structure


def _dtype_from_leaves(leaves: Iterable[Any]) -> DTypeLike:
    leaves = (leaf for leaf in leaves if leaf is not None)
    return jnp.result_type(*leaves)


def _offsets_from_leaves(leaves: Iterable[Any | None]) -> tuple[int, ...]:
    offsets: list[int] = []
    i: int = 0
    for leaf in leaves:
        if leaf is not None:
            i += jnp.size(leaf)
        offsets.append(i)
    del offsets[-1]
    return tuple(offsets)


@jax.jit
def _ravel(dynamic_leaves: Iterable[Array | None]) -> Array:
    return jnp.concatenate(
        [leaf for leaf in dynamic_leaves if leaf is not None], axis=None
    )


def _shapes_from_leaves(leaves: Iterable[Any | None]) -> tuple[Shape | None, ...]:
    return tuple(None if leaf is None else jnp.shape(leaf) for leaf in leaves)


@jax.jit(static_argnums=(1, 2))
def _unravel(
    flat: Array, offsets: tuple[int, ...], shapes: tuple[Shape | None, ...]
) -> list[Array | None]:
    chunks: list[Array] = jnp.split(flat, offsets)
    leaves: list[Array | None] = [
        None if shape is None else jnp.reshape(chunk, shape)
        for chunk, shape in zip(chunks, shapes, strict=True)
    ]
    return leaves
