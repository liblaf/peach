from collections.abc import Callable
from typing import Self

from jaxtyping import PyTree

from liblaf.peach import tree
from liblaf.peach.functools import FunctionContext, MethodDescriptor
from liblaf.peach.tree import FlatView


@tree.define
class LinearSystem(FunctionContext):
    matvec = MethodDescriptor(
        n_outputs=1, in_structures={0: "input"}, out_structures={0: "input"}
    )
    """X -> X"""
    _matvec_wrapped: Callable | None = tree.field(alias="matvec")
    _matvec_wrapper: Callable | None = tree.field(default=None, init=False)

    b: PyTree = tree.field()
    b_flat = FlatView()

    rmatvec = MethodDescriptor(
        n_outputs=1, in_structures={0: "input"}, out_structures={0: "input"}
    )
    """X -> X"""
    _rmatvec_wrapped: Callable | None = tree.field(
        default=None, alias="rmatvec", kw_only=True
    )
    _rmatvec_wrapper: Callable | None = tree.field(default=None, init=False)

    preconditioner = MethodDescriptor(
        n_outputs=1, in_structures={0: "input"}, out_structures={0: "input"}
    )
    """X -> X"""
    _preconditioner_wrapped: Callable | None = tree.field(
        default=None, alias="preconditioner", kw_only=True
    )
    _preconditioner_wrapper: Callable | None = tree.field(default=None, init=False)

    rpreconditioner = MethodDescriptor(
        n_outputs=1, in_structures={0: "input"}, out_structures={0: "input"}
    )
    """X -> X"""
    _rpreconditioner_wrapped: Callable | None = tree.field(
        default=None, alias="rpreconditioner", kw_only=True
    )
    _rpreconditioner_wrapper: Callable | None = tree.field(default=None, init=False)

    def flatten(self, structure: tree.Structure) -> Self:
        return self.with_structures({"input": structure}, flatten=True)
