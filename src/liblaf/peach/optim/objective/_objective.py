from __future__ import annotations

from collections.abc import Callable

from liblaf.peach import tree_utils
from liblaf.peach.functools import FunctionDescriptor, FunctionWrapper


@tree_utils.define
class Objective(FunctionWrapper):
    fun = FunctionDescriptor(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=())
    """X -> Scalar"""
    _fun_wrapped: Callable | None = tree_utils.field(default=None, alias="fun")
    _fun_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    grad = FunctionDescriptor(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> X"""
    _grad_wrapped: Callable | None = tree_utils.field(default=None, alias="grad")
    _grad_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    hess = FunctionDescriptor(n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,))
    """X -> H"""
    _hess_wrapped: Callable | None = tree_utils.field(default=None, alias="hess")
    _hess_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    hess_diag = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0,), flatten_outputs=(0,)
    )
    """X -> X"""
    _hess_diag_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_diag"
    )
    _hess_diag_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    hess_prod = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=(0,)
    )
    """X, P -> X"""
    _hess_prod_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_prod"
    )
    _hess_prod_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    hess_quad = FunctionDescriptor(
        n_outputs=1, unflatten_inputs=(0, 1), flatten_outputs=()
    )
    """X, P -> Scalar"""
    _hess_quad_wrapped: Callable | None = tree_utils.field(
        default=None, alias="hess_quad"
    )
    _hess_quad_wrapper: Callable | None = tree_utils.field(default=None, init=False)

    value_and_grad = FunctionDescriptor(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(1,)
    )
    """X -> Scalar, X"""
    _value_and_grad_wrapped: Callable | None = tree_utils.field(
        default=None, alias="value_and_grad"
    )
    _value_and_grad_wrapper: Callable | None = tree_utils.field(
        default=None, init=False
    )

    grad_and_hess_diag = FunctionDescriptor(
        n_outputs=2, unflatten_inputs=(0,), flatten_outputs=(0, 1)
    )
    """X -> X, X"""
    _grad_and_hess_diag_wrapped: Callable | None = tree_utils.field(
        default=None, alias="grad_and_hess_diag"
    )
    _grad_and_hess_diag_wrapper: Callable | None = tree_utils.field(
        default=None, init=False
    )
