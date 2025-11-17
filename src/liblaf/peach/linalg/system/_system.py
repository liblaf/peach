from collections.abc import Callable

from jaxtyping import PyTree

from liblaf.peach import tree
from liblaf.peach.functools import FunctionDescriptor, FunctionWrapper
from liblaf.peach.tree import FlatView


@tree.define
class LinearSystem(FunctionWrapper):
    matvec = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _matvec_wrapped: Callable | None = tree.field(alias="matvec")
    _matvec_wrapper: Callable | None = tree.field(default=None, init=False)

    b: PyTree = tree.field()
    b_flat = FlatView()

    preconditioner = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _preconditioner_wrapped: Callable | None = tree.field(
        default=None, alias="preconditioner"
    )
    _preconditioner_wrapper: Callable | None = tree.field(default=None, init=False)
