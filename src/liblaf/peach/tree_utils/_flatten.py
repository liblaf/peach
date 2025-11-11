from collections.abc import Callable

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
from jaxtyping import Array, Bool, Integer, Shaped

from ._define import define
from ._field import static


@define
class Unflatten[T]:
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
    def __call__(self, free_flat: Shaped[Array, " free"]) -> T:
        full_flat: Shaped[Array, " data"] = (
            free_flat
            if self.free_indices is None
            else self.full_flat.at[self.free_indices].set(free_flat)
        )
        data: T = self.unravel(full_flat)
        tree: T = eqx.combine(data, self.static)
        return tree


def flatten[T](
    obj: T, *, fixed_mask: T | None = None, n_fixed: int | None = None
) -> tuple[Array, Unflatten[T]]:
    if fixed_mask is None or n_fixed is not None:
        return _flatten_jit(obj, fixed_mask=fixed_mask, n_fixed=n_fixed)
    return _flatten(obj, fixed_mask=fixed_mask, n_fixed=n_fixed)


def _flatten[T](
    obj: T, *, fixed_mask: T | None = None, n_fixed: int | None = None
) -> tuple[Array, Unflatten[T]]:
    data: T
    static: T
    data, static = eqx.partition(obj, eqx.is_array)
    full_flat: Shaped[Array, " full"]
    unravel: Callable[[Array], T]
    full_flat, unravel = jfu.ravel_pytree(data)
    free_indices: Integer[Array, " free"] | None = None
    free_flat: Shaped[Array, " free"] = full_flat
    if fixed_mask is not None:
        fixed_mask, _ = eqx.partition(fixed_mask, eqx.is_array)
        fixed_mask_flat: Bool[Array, " full"]
        fixed_mask_flat, _ = jfu.ravel_pytree(fixed_mask)
        n_free: int | None = None if n_fixed is None else fixed_mask_flat.size - n_fixed
        free_indices = jnp.flatnonzero(~fixed_mask_flat, size=n_free)
        free_flat = full_flat[free_indices]
    return free_flat, Unflatten(
        full_flat=full_flat, unravel=unravel, static=static, free_indices=free_indices
    )


_flatten_jit = eqx.filter_jit(_flatten)
