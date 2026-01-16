from collections.abc import Callable
from typing import Self, override

import attrs
import equinox as eqx
from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree
from liblaf.peach.functools import FunctionWrapper, MethodDescriptor

from ._abc import Constraint

type Vector = Float[Array, " N"]


@tree.define
class ProjectionConstraint(Constraint, FunctionWrapper):
    project_params = MethodDescriptor()
    _project_params_wrapped: Callable = tree.field(kw_only=True, alias="project_params")
    _project_params_wrapper: Callable | None = tree.field(default=None, init=False)

    project_grad = MethodDescriptor()
    _project_grad_wrapped: Callable | None = tree.field(
        default=None, kw_only=True, alias="project_grad"
    )
    _project_grad_wrapper: Callable | None = tree.field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        if self._project_grad_wrapped is None:
            self._project_grad_wrapped = self._default_project_grad

    @override  # impl Constraint
    def flatten(self, structure: tree.Structure) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return attrs.evolve(self, flatten=True, structure=structure)

    def _default_project_grad(self, grad: PyTree) -> PyTree:
        # project_params should be linear, so we can use any x here
        # to avoid creating a new pytree, we use grad itself as x
        x = grad
        tangents_out: PyTree
        _primals_out, tangents_out = eqx.filter_jvp(
            self._project_params_wrapped, (x,), (grad,)
        )
        return tangents_out
