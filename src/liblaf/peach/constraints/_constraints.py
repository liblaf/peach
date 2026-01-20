import collections
from typing import Any, Self

import jax.tree_util as jtu
from jaxtyping import PyTree

from ._abc import Constraint
from ._bound import BoundConstraint
from ._fixed import FixedConstraint

type AuxData = None
type Children = list[Constraint]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Constraint]
type KeyLeafPairs = list[KeyLeafPair]
type Params = PyTree
type Vector = PyTree


@jtu.register_pytree_with_keys_class
class Constraints(collections.UserList[Constraint]):
    @property
    def bound(self) -> BoundConstraint | None:
        return self.get(BoundConstraint)

    @property
    def fixed(self) -> FixedConstraint | None:
        return self.get(FixedConstraint)

    @property
    def other_constraints(self) -> list[Constraint]:
        known_types: tuple[type[Constraint], ...] = (BoundConstraint, FixedConstraint)
        return [c for c in self.data if not isinstance(c, known_types)]

    def get[T](self, cls: type[T]) -> T | None:
        matches: list[T] = [
            constraint for constraint in self.data if isinstance(constraint, cls)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise NotImplementedError
        return matches[0]

    def project_params(self, params: Vector) -> Vector:
        for constraint in self.data:
            params = constraint.project_params(params)
        return params

    def project_grads(self, params: Vector, grads: Vector) -> Vector:
        for constraint in self.data:
            grads = constraint.project_grads(params, grads)
        return grads

    def tree_flatten(self) -> tuple[Children, None]:
        return list(self.data), None

    def tree_flatten_with_keys(self) -> tuple[KeyLeafPairs, AuxData]:
        children: KeyLeafPairs = [
            (jtu.SequenceKey(i), constraint) for i, constraint in enumerate(self.data)
        ]
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux: AuxData, children: Children) -> Self:
        return cls(children)
