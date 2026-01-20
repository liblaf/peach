from typing import override

import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from liblaf.peach import compile_utils, tree

from ._abc import Constraint

type Params = PyTree
type Vector = Float[Array, " N"]


@tree.define
class BoundConstraint(Constraint):
    lower = tree.TreeView()
    lower_flat: Vector | None = tree.field(default=None, kw_only=True)
    upper = tree.TreeView()
    upper_flat: Vector | None = tree.field(default=None, kw_only=True)
    keep_feasible: bool = tree.field(default=True, kw_only=True)

    def __init__(
        self,
        lower: Params | None = None,
        upper: Params | None = None,
        *,
        keep_feasible: bool = True,
    ) -> None:
        lower_flat: Vector | None = None
        upper_flat: Vector | None = None
        structure: tree.Structure | None = None
        if lower is not None:
            lower_flat, structure = tree.flatten(lower)
        if upper is not None:
            upper_flat, structure = tree.flatten(upper)
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            lower_flat=lower_flat,
            upper_flat=upper_flat,
            keep_feasible=keep_feasible,
            structure=structure,
        )

    @override
    @compile_utils.jit(inline=True)
    def project_params(self, params: Params | Vector) -> Vector:
        params_flat: Vector = self.structure.flatten(params)
        return jnp.clip(params_flat, min=self.lower_flat, max=self.upper_flat)
