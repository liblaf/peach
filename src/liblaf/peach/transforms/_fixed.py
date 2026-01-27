from typing import override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from ._base import Transform

type Free = Float[Array, " free"]
type Full = Float[Array, " full"]


@jarp.define
class FixedTransform[T](Transform[Free, T]):
    free_indices: Integer[Array, " free"]
    full_flat: Full
    structure: jarp.Structure[T] = jarp.static()

    def __init__(self, mask: T, values: T) -> None:
        full_flat: Full
        structure: jarp.Structure[T]
        full_flat, structure = jarp.ravel(values)
        mask_flat: Bool[Array, " full"] = structure.ravel(mask)
        free_indices: Integer[Array, " free"] = jnp.flatnonzero(~mask_flat)
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            free_indices=free_indices, full_flat=full_flat, structure=structure
        )

    @override
    @jarp.jit(inline=True)
    def forward_primals(self, primals_in: Free) -> T:
        full_flat: Full = self.full_flat
        full_flat: Full = full_flat.at[self.free_indices].set(primals_in)
        return self.structure.unravel(full_flat)

    @override
    @jarp.jit(inline=True)
    def forward_tangents(self, tangents_in: Free) -> T:
        full_flat: Full = jnp.zeros_like(self.full_flat)
        full_flat: Full = full_flat.at[self.free_indices].set(tangents_in)
        return self.structure.unravel(full_flat)

    @override
    @jarp.jit(inline=True)
    def backward_primals(self, primals_out: T) -> Free:
        full: Full = self.structure.ravel(primals_out)
        return full[self.free_indices]

    @override
    @jarp.jit(inline=True)
    def backward_tangents(self, tangents_out: T) -> Free:
        full_flat: Full = self.structure.ravel(tangents_out)
        return full_flat[self.free_indices]

    @override
    @jarp.jit(inline=True)
    def backward_hess_diag(self, hess_diag_out: T) -> Free:
        full_flat: Full = self.structure.ravel(hess_diag_out)
        return full_flat[self.free_indices]
