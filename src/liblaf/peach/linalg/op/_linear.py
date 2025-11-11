from __future__ import annotations

from collections.abc import Callable

from jaxtyping import PyTree

from liblaf.peach import tree
from liblaf.peach.functools import FunctionDescriptor, FunctionWrapper


@tree.define
class LinearOperator(FunctionWrapper):
    matvec = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _matvec_wrapped: Callable | None = tree.field(default=None, alias="matvec")
    _matvec_wrapper: Callable | None = tree.field(default=None, init=False)

    preconditioner = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _preconditioner_wrapped: Callable | None = tree.field(
        default=None, alias="preconditioner"
    )
    _preconditioner_wrapper: Callable | None = tree.field(default=None, init=False)

    def __call__(self, *args, **kwargs) -> PyTree:
        assert self.matvec is not None
        return self.matvec(*args, **kwargs)
