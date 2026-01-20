from collections.abc import Callable
from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, PyTree

from liblaf.peach import tree

from ._abc import LinearTransform

type Free = Float[Array, " free"]
type Full = Float[Array, " full"]
type Params = PyTree


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
        full_flat: Full = self.full_flat
        full_flat = full_flat.at[self.free_indices].set(primals)
        return self.structure.unflatten(full_flat)

    @override
    def linearize(self, primals: Free) -> tuple[Params, Callable[[Free], Params]]:
        primals_out: Params = self.forward_primals(primals)
        return primals_out, self._lin_fun

    @override
    def linear_transpose(self, primals: Free) -> Callable[[Params], Free]:
        return self._flatten

    @override
    def forward_hess_diag(self, hess_diag: Free) -> Params:
        return self.forward_primals(hess_diag)

    @override
    def backward_params(self, primals_out: Params) -> Free:
        return self._flatten(primals_out)

    @override
    def backward_hess_diag(self, hess_diag_out: Params) -> Free:
        return self._flatten(hess_diag_out)

    def _lin_fun(self, tangents: Free) -> Params:
        full_flat: Full = jnp.zeros_like(self.full_flat)
        full_flat = full_flat.at[self.free_indices].set(tangents)
        return self.structure.unflatten(full_flat)

    def _flatten(self, primals: Params) -> Free:
        full_flat: Full = self.structure.flatten(primals)
        return full_flat[self.free_indices]
