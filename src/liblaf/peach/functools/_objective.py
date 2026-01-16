from collections.abc import Callable

from liblaf.peach import tree

from ._context import FunctionContext
from ._descriptor import MethodDescriptor


@tree.define
class Objective(FunctionContext):
    fun = MethodDescriptor(in_structures={0: "params"}, out_structures={})
    _fun_wrapped: Callable | None = tree.field(default=None, alias="fun")
    _fun_wrapper: Callable | None = tree.field(default=None, init=False)
    """x (Params) -> fun (Scalar)"""

    grad = MethodDescriptor(in_structures={0: "params"}, out_structures={0: "params"})
    _grad_wrapped: Callable | None = tree.field(default=None, alias="grad")
    _grad_wrapper: Callable | None = tree.field(default=None, init=False)
    """x (Params) -> grad (Params)"""

    hess_prod = MethodDescriptor(
        in_structures={0: "params", 1: "params"}, out_structures={0: "params"}
    )
    _hess_prod_wrapped: Callable | None = tree.field(default=None, alias="hess_prod")
    _hess_prod_wrapper: Callable | None = tree.field(default=None, init=False)
    """x (Params), v (Params) -> H @ v (Params)"""

    hess_quad = MethodDescriptor(
        in_structures={0: "params", 1: "params"}, out_structures={}
    )
    """x (Params), v (Params) -> v.T @ H @ v (Scalar)"""
    _hess_quad_wrapped: Callable | None = tree.field(default=None, alias="hess_quad")
    _hess_quad_wrapper: Callable | None = tree.field(default=None, init=False)

    value_and_grad = MethodDescriptor(
        in_structures={0: "params"}, out_structures={1: "params"}
    )
    """x (Params) -> fun (Scalar), grad (Params)"""
    _value_and_grad_wrapped: Callable | None = tree.field(
        default=None, alias="value_and_grad"
    )
    _value_and_grad_wrapper: Callable | None = tree.field(default=None, init=False)

    preconditioner: MethodDescriptor = MethodDescriptor(
        in_structures={0: "params"}, out_structures={}
    )
    """x (Params) -> preconditioner (Params -> Params)"""
    _preconditioner_wrapped: Callable | None = tree.field(
        default=None, alias="preconditioner"
    )
    _preconditioner_wrapper: Callable | None = tree.field(default=None, init=False)
