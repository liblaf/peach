import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import testing, tree
from liblaf.peach.functools import Objective
from liblaf.peach.optim import ScipyOptimizer
from liblaf.peach.optim.abc import Optimizer

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@tree.define
class Params:
    data: Vector = tree.array()


@tree.define
class Model:
    structure: tree.Structure[Params] = tree.field(kw_only=True)

    def fun(self, params: Params | Vector) -> Scalar:
        params: Params = self.structure.unflatten(params)
        return testing.rosen(params.data)

    def grad(self, params: Params | Vector) -> Vector:
        params: Params = self.structure.unflatten(params)
        grad_flat: Vector = testing.rosen_grad(params.data)
        grad: Params = Params(data=grad_flat)
        return self.structure.flatten(grad)

    def hess_prod(self, params: Params | Vector, p: Params | Vector) -> Vector:
        params: Params = self.structure.unflatten(params)
        p: Params = self.structure.unflatten(p)
        hess_prod_flat: Vector = testing.rosen_hess_prod(params.data, p.data)
        hess_prod: Params = Params(data=hess_prod_flat)
        return self.structure.flatten(hess_prod)

    def hess_quad(self, params: Params | Vector, p: Params | Vector) -> Scalar:
        params: Params = self.structure.unflatten(params)
        p: Params = self.structure.unflatten(p)
        return testing.rosen_hess_quad(params.data, p.data)

    def value_and_grad(self, params: Params | Vector) -> tuple[Scalar, Vector]:
        params: Params = self.structure.unflatten(params)
        value: Scalar
        grad_flat: Vector
        value, grad_flat = testing.rosen_value_and_grad(params.data)
        grad: Params = Params(data=grad_flat)
        return value, self.structure.flatten(grad)


def check_optimizer(
    objective: Objective, optimizer: Optimizer, *, atol: float = 1e-3
) -> None:
    params: Params = Params(data=jnp.zeros((7,)))
    params_flat: Vector
    structure: tree.Structure[Params]
    params_flat, structure = tree.flatten(params)
    model: Model = Model(structure=structure)
    objective = objective.partial(model)
    solution: Optimizer.Solution = optimizer.minimize(objective, params_flat)
    assert solution.success
    params: Params = structure.unflatten(solution.params)
    np.testing.assert_allclose(params.data, jnp.ones((7,)), atol=atol)


# def test_optax_adam(objective: Objective) -> None:
#     optimizer = Optax(
#         optax.adam(learning_rate=1e-2), max_steps=1000, rtol=0.0, jit=True, timer=True
#     )
#     check_optimizer(objective, optimizer, atol=1e-2)


# def test_pncg(objective: Objective) -> None:
#     optimizer = PNCG(rtol=1e-9, jit=True, timer=True)
#     check_optimizer(objective, optimizer)


def test_scipy_lbfgsb() -> None:
    objective: Objective = Objective(value_and_grad=Model.value_and_grad)
    optimizer = ScipyOptimizer(method="L-BFGS-B")
    check_optimizer(objective, optimizer)
