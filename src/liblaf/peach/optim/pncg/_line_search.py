import abc
from collections.abc import Callable
from typing import override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.abc import Params
from liblaf.peach.optim.objective import Objective
from liblaf.peach.optim.pncg import PNCGState

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearch(abc.ABC):
    @abc.abstractmethod
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
        state: PNCGState,
    ) -> Scalar:
        raise NotImplementedError


@tree.define
class LineSearchMin(LineSearch):
    methods: list[LineSearch] = tree.field(
        factory=lambda: [LineSearchNewton(), LineSearchUpper()], kw_only=True
    )

    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
        state: PNCGState,
    ) -> Scalar:
        return jnp.asarray(
            [
                method.search(objective, params, grad, search_direction, state)
                for method in self.methods
            ]
        ).min()


@tree.define
class LineSearchNewton(LineSearch):
    @override
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
        state: PNCGState,
    ) -> Scalar:
        assert objective.hess_quad is not None
        hess_quad: Scalar = objective.hess_quad(params, search_direction)
        return self._search(grad, search_direction, hess_quad)

    @eqx.filter_jit
    def _search(
        self, grad: Vector, search_direction: Vector, hess_quad: Scalar
    ) -> Scalar:
        return -jnp.vdot(grad, search_direction) / hess_quad


@tree.define
class LineSearchUpper(LineSearch):
    d_hat: Scalar = tree.field(default=jnp.asarray(jnp.inf), kw_only=True)
    norm: Callable[[Params], Scalar] | None = tree.field(default=None, kw_only=True)

    @override
    @eqx.filter_jit
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
        state: PNCGState,
    ) -> Scalar:
        p_norm: Scalar
        if self.norm is None:
            p_norm = jnp.linalg.norm(search_direction, ord=jnp.inf)
        else:
            assert state.flat_def is not None
            p_tree: Params = state.flat_def.unflatten(search_direction)
            p_norm = self.norm(p_tree)
        return self.d_hat / (2.0 * p_norm)  # pyright: ignore[reportAssignmentType]
