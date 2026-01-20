from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from liblaf.peach import compile_utils, tree

from ._abc import LinearTransform

type Free = Float[Array, " free"]
type Full = Float[Array, " full"]


@tree.define
class FixedTransform[T](LinearTransform[Free, T]):
    free_indices: Integer[Array, " free"]
    full_flat: Full
    structure: tree.Structure[T] = tree.static(repr=False, kw_only=True)

    def __init__(self, mask: T, values: T) -> None:
        full_flat: Full
        structure: tree.Structure[T]
        full_flat, structure = tree.flatten(values)
        mask_flat: Bool[Array, " full"] = structure.flatten(mask)
        free_indices: Integer[Array, " free"] = jnp.flatnonzero(~mask_flat)
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            free_indices=free_indices, full_flat=full_flat, structure=structure
        )

    @override
    def forward_primals(self, primals: Free) -> T:
        full_flat: Full = _fill_values(primals, self.full_flat, self.free_indices)
        return self.structure.unflatten(full_flat)

    @override
    def linearize(self, primals: Free) -> tuple[T, Callable[[Free], T]]:
        primals_out: T = self.forward_primals(primals)
        return primals_out, self._lin_fun

    @override
    def linear_transpose(self, tangents_out: T) -> Free:
        return self._flatten(tangents_out)

    @override
    def forward_tangents(self, primals: Free, tangents: Free) -> T:
        return self._lin_fun(tangents)

    @override
    def forward_hess_diag(self, hess_diag: Free) -> T:
        return self._lin_fun(hess_diag)

    @override
    def backward_params(self, primals_out: T) -> Free:
        return self._flatten(primals_out)

    @override
    def backward_hess_diag(self, hess_diag_out: T) -> Free:
        return self._flatten(hess_diag_out)

    def _lin_fun(self, tangents: Free) -> T:
        full_flat: Full = _fill_zeros(tangents, self.full_flat, self.free_indices)
        return self.structure.unflatten(full_flat)

    def _flatten(self, primals: T) -> Free:
        full_flat: Full = self.structure.flatten(primals)
        return full_flat[self.free_indices]


@compile_utils.jit(inline=True)
def _fill_values(free: Free, full: Full, free_indices: Integer[Array, " free"]) -> Full:
    return full.at[free_indices].set(free)


@compile_utils.jit(inline=True)
def _fill_zeros(free: Free, full: Full, free_indices: Integer[Array, " free"]) -> Full:
    full = jnp.zeros_like(full)
    return full.at[free_indices].set(free)
