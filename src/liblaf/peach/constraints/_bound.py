from typing import Self, override

from jaxtyping import PyTree

from liblaf.peach import tree

from ._abc import Constraint


@tree.define
class BoundConstraint(Constraint):
    lower: PyTree | None = tree.field(default=None)
    upper: PyTree | None = tree.field(default=None)
    keep_feasible: bool = tree.field(default=True, kw_only=True)

    @override
    def flatten(self, structure: tree.Structure[PyTree]) -> Self:
        lower_flat: PyTree | None = None
        upper_flat: PyTree | None = None
        if self.lower is not None:
            lower_flat = structure.flatten(self.lower)
        if self.upper is not None:
            upper_flat = structure.flatten(self.upper)
        return type(self)(lower=lower_flat, upper=upper_flat)
