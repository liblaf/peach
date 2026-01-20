from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self, overload, override

import jax.numpy as jnp
import wrapt
from jaxtyping import Array, Bool, Float, Integer, PyTree

from liblaf.peach import compile_utils, tree, utils
from liblaf.peach.functools import Objective, ObjectiveProtocol

from ._abc import Constraint

type Params = PyTree
type Vector = Float[Array, " free"]


@tree.define
class FixedConstraint(Constraint):
    fixed_indices: Integer[Array, " fixed"] = tree.field(kw_only=True)
    free_indices: Integer[Array, " free"] = tree.field(kw_only=True)
    mask = tree.TreeView()
    mask_flat: Vector = tree.field(kw_only=True)
    params = tree.TreeView()
    params_flat: Float[Array, " full"] = tree.field(kw_only=True)

    def __init__(self, mask: Params, params: Params) -> None:
        params_flat: Float[Array, " full"]
        structure: tree.Structure
        params_flat, structure = tree.flatten(params)
        mask_flat: Bool[Array, " N"] = structure.flatten(mask)
        fixed_indices: Integer[Array, " fixed"] = jnp.flatnonzero(mask_flat)
        free_indices: Integer[Array, " free"] = jnp.flatnonzero(~mask_flat)
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            fixed_indices=fixed_indices,
            free_indices=free_indices,
            mask_flat=mask_flat,
            params_flat=params_flat,
            structure=structure,
        )

    @compile_utils.jit(inline=True)
    def fill_params(self, free_params: Vector) -> Vector:
        params_flat: Vector = self.params_flat
        params_flat = params_flat.at[self.free_indices].set(free_params)
        return params_flat

    @compile_utils.jit(inline=True)
    def fill_grad(self, free_grad: Vector) -> Vector:
        grad_flat: Vector = jnp.zeros_like(self.params_flat)
        grad_flat = grad_flat.at[self.free_indices].set(free_grad)
        return grad_flat

    @compile_utils.jit(inline=True)
    def get_fixed(self, params: Params | Vector) -> Vector:
        params_flat: Vector = self.structure.flatten(params)
        fixed_params: Vector = params_flat[self.fixed_indices]
        return fixed_params

    @compile_utils.jit(inline=True)
    def get_free(self, params: Params | Vector) -> Vector:
        params_flat: Vector = self.structure.flatten(params)
        free_params: Vector = params_flat[self.free_indices]
        return free_params

    @override
    @compile_utils.jit(inline=True)
    def project_params(self, params: Params | Vector) -> Vector:
        params_flat: Vector = self.structure.flatten(params)
        params_flat = jnp.where(self.mask_flat, self.params_flat, params_flat)
        return params_flat

    @override
    @compile_utils.jit(inline=True)
    def project_grad(self, params: Params | Vector, grad: Params | Vector) -> Vector:
        grad_flat: Vector = self.structure.flatten(grad)
        grad_flat = jnp.where(self.mask_flat, 0.0, grad_flat)
        return grad_flat

    @overload
    def wraps[**P, T](
        self,
        func: Callable[P, T],
        *,
        fill_params: tuple[int, ...] = (),
        fill_grad: tuple[int, ...] = (),
        get_free: tuple[int, ...] = (),
    ) -> Callable[P, T]: ...
    @overload
    def wraps(
        self,
        func: None,
        *,
        fill_params: tuple[int, ...] = (),
        fill_grad: tuple[int, ...] = (),
        get_free: tuple[int, ...] = (),
    ) -> None: ...
    def wraps[**P, T](
        self,
        func: Callable[P, T] | None,
        *,
        fill_params: tuple[int, ...] = (),
        fill_grad: tuple[int, ...] = (),
        get_free: tuple[int, ...] = (),
    ) -> Callable[P, T] | None:
        if func is None:
            return None
        decorator = FixedConstraintDecorator(
            fixed_constraint=self,
            fill_params=fill_params,
            fill_grad=fill_grad,
            get_free=get_free,
        )
        return decorator(func)

    def wraps_objective(self, objective: Objective) -> FixedConstraintObjectiveWrapper:
        return FixedConstraintObjectiveWrapper(
            __wrapped__=objective, fixed_constraint=self
        )


@tree.define
class FixedConstraintDecorator:
    fixed_constraint: FixedConstraint
    fill_params: tuple[int, ...] = tree.static(default=(), kw_only=True)
    fill_grad: tuple[int, ...] = tree.static(default=(), kw_only=True)
    get_free: tuple[int, ...] = tree.static(default=(), kw_only=True)

    @wrapt.decorator
    def __call__[**P, T](
        self,
        wrapped: Callable[P, T],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        args: list[Any] = list(args)
        for i in self.fill_params:
            args[i] = self.fixed_constraint.fill_params(args[i])
        for i in self.fill_grad:
            args[i] = self.fixed_constraint.fill_grad(args[i])
        outputs: list[Any] = utils.pack(wrapped(*args, **kwargs))  # pyright: ignore[reportCallIssue]
        for i in self.get_free:
            outputs[i] = self.fixed_constraint.get_free(outputs[i])
        return utils.unpack(outputs)  # pyright: ignore[reportReturnType]


@tree.define
class _FixedConstraintFunctionDescriptor:
    name: str = tree.static(default=None, kw_only=True)
    fill_params: tuple[int, ...] = tree.static(default=(), kw_only=True)
    fill_grad: tuple[int, ...] = tree.static(default=(), kw_only=True)
    get_free: tuple[int, ...] = tree.static(default=(), kw_only=True)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(
        self, instance: FixedConstraintObjectiveWrapper, owner: type | None = None
    ) -> Callable: ...
    def __get__(
        self,
        instance: FixedConstraintObjectiveWrapper | None,
        owner: type | None = None,
    ) -> Any:
        if instance is None:
            return self
        wrapped: Callable | None = getattr(instance.__wrapped__, self.name, None)
        if wrapped is None:
            return None
        return instance.fixed_constraint.wraps(
            wrapped,
            fill_params=self.fill_params,
            fill_grad=self.fill_grad,
            get_free=self.get_free,
        )

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name


@tree.define
class FixedConstraintObjectiveWrapper(ObjectiveProtocol):
    __wrapped__: Objective
    fixed_constraint: FixedConstraint

    if TYPE_CHECKING:
        fun: Callable | None = tree.field(init=False)
        grad: Callable | None = tree.field(init=False)
        value_and_grad: Callable | None = tree.field(init=False)
        hess_prod: Callable | None = tree.field(init=False)
        hess_quad: Callable | None = tree.field(init=False)
        preconditioner: Callable | None = tree.field(init=False)
    else:
        fun = _FixedConstraintFunctionDescriptor(fill_params=(0,))
        grad = _FixedConstraintFunctionDescriptor(fill_params=(0,), get_free=(0,))
        value_and_grad = _FixedConstraintFunctionDescriptor(
            fill_params=(0,), get_free=(1,)
        )
        hess_prod = _FixedConstraintFunctionDescriptor(
            fill_params=(0,), fill_grad=(1,), get_free=(0,)
        )
        hess_quad = _FixedConstraintFunctionDescriptor(fill_params=(0,), fill_grad=(1,))
        preconditioner = _FixedConstraintFunctionDescriptor(fill_params=(0,))
