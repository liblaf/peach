from jaxtyping import PyTree

from liblaf.peach import tree

from ._abc import Constraint


@tree.define
class FixedConstraint(Constraint):
    mask: PyTree | None = tree.field(default=None, kw_only=True)
    # TODO: support selector
