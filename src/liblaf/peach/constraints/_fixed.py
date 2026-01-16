from collections.abc import Iterable

from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree, utils

from ._abc import Constraint

type Vector = Float[Array, " free"]


@tree.define
class FixedConstraint(Constraint):
    mask: PyTree | None = tree.field(default=None, kw_only=True)


def flatten_with_constraints(
    params: PyTree, constraints: Iterable[Constraint]
) -> tuple[Vector, tree.Structure, list[Constraint]]:
    fixed_constraints: list[FixedConstraint]
    fixed_constraints, constraints = utils.partition_type(FixedConstraint, constraints)
    if len(fixed_constraints) > 1:
        raise NotImplementedError
    fixed_mask: PyTree | None = fixed_constraints[0].mask if fixed_constraints else None
    params_free: Vector
    structure: tree.Structure
    params_free, structure = tree.flatten(params, fixed_mask=fixed_mask)
    return params_free, structure, constraints
