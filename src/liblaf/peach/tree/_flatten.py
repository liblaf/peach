from collections.abc import Callable, Iterable

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Integer, PyTree, Shaped

from ._define import define
from ._field import static


@define
class FlatDef[T]:
    full_flat: Shaped[Array, " full"]
    static: T = static()
    unravel: Callable[[Shaped[Array, " full"]], T]
    free_indices: Integer[Array, " free"] | None = None

    @eqx.filter_jit
    def flatten(self, tree: T) -> Shaped[Array, " free"]:
        data: T = eqx.filter(tree, eqx.is_array)
        full_flat: Shaped[Array, " full"]
        full_flat, _ = jfu.ravel_pytree(data)
        free_flat: Shaped[Array, " free"] = (
            full_flat if self.free_indices is None else full_flat[self.free_indices]
        )
        return free_flat

    @eqx.filter_jit
    def unflatten(
        self, free_flat: Shaped[Array, " free"], dtype: DTypeLike | None = None
    ) -> T:
        if dtype is None:
            dtype = self.full_flat.dtype
        free_flat = jnp.asarray(free_flat, dtype)
        full_flat: Shaped[Array, " data"] = (
            free_flat
            if self.free_indices is None
            else self.full_flat.at[self.free_indices].set(free_flat)
        )
        data: T = self.unravel(full_flat)
        tree: T = eqx.combine(data, self.static)
        return tree


def flatten[T](
    obj: T,
    *,
    fixed_mask: T | None = None,
    fixed_selector: Callable[[T], Iterable[Array]] | None = None,
) -> tuple[Array, FlatDef[T]]:
    # TODO: JIT?
    return _flatten(obj, fixed_mask=fixed_mask, fixed_selector=fixed_selector)


def _flatten[T](
    obj: T,
    *,
    fixed_mask: T | None = None,
    fixed_selector: Callable[[T], Iterable[Array]] | None = None,
) -> tuple[Array, FlatDef[T]]:
    data: T
    static: T
    data, static = eqx.partition(obj, eqx.is_array)
    full_flat: Shaped[Array, " full"]
    unravel: Callable[[Array], T]
    full_flat, unravel = jfu.ravel_pytree(data)
    free_indices: Integer[Array, " free"] | None = None
    if fixed_mask is not None:
        free_indices = _fixed_mask_to_free_indices(fixed_mask)
    elif fixed_selector is not None:
        free_indices = _fixed_selector_to_free_indices(
            fixed_selector, n_full=full_flat.size, unravel=unravel
        )
    free_flat: Shaped[Array, " free"] = (
        full_flat[free_indices] if free_indices is not None else full_flat
    )
    return free_flat, FlatDef(
        full_flat=full_flat, unravel=unravel, static=static, free_indices=free_indices
    )


def _fixed_mask_to_free_indices(
    fixed_mask: PyTree, *, n_free: int | None = None
) -> Integer[Array, " free"]:
    fixed_mask = eqx.filter(fixed_mask, eqx.is_array)
    fixed_mask_flat: Bool[Array, " full"]
    fixed_mask_flat, _ = jfu.ravel_pytree(fixed_mask)
    free_mask_flat: Bool[Array, " full"] = ~fixed_mask_flat
    free_idx_flat: Integer[Array, " free"] = jnp.flatnonzero(
        free_mask_flat, size=n_free
    )
    return free_idx_flat


def _fixed_selector_to_free_indices[T](
    fixed_selector: Callable[[T], PyTree],
    n_full: int,
    unravel: Callable[[Array], T],
    *,
    n_free: int | None = None,
) -> Integer[Array, " free"]:
    full_idx_flat: Integer[Array, " full"] = jnp.arange(n_full)
    full_idx_tree: T = unravel(full_idx_flat)
    fixed_idx_tree: PyTree = fixed_selector(full_idx_tree)
    fixed_idx_flat: Integer[Array, " fixed"]
    fixed_idx_flat, _ = jfu.ravel_pytree(fixed_idx_tree)
    free_mask_flat: Bool[Array, " full"] = jnp.ones(n_full, jnp.bool)
    free_mask_flat = free_mask_flat.at[fixed_idx_flat].set(False)
    free_idx_flat: Integer[Array, " free"] = jnp.flatnonzero(
        free_mask_flat, size=n_free
    )
    return free_idx_flat


_flatten_jit = eqx.filter_jit(_flatten)
