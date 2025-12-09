import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array, Float

from liblaf.peach import testing, tree
from liblaf.peach.optim import PNCG, Objective, Optax, ScipyOptimizer
from liblaf.peach.optim.abc import Optimizer

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@tree.define
class Params:
    data: Vector = tree.array()


@pytest.fixture(scope="package")
def objective() -> Objective:
    def fun(params: Params) -> Scalar:
        return testing.rosen(params.data)

    def grad(params: Params) -> Params:
        grad: Vector = testing.rosen_grad(params.data)
        return Params(data=grad)

    def hess_diag(params: Params) -> Params:
        hess_diag: Vector = testing.rosen_hess_diag(params.data)
        return Params(data=hess_diag)

    def hess_prod(params: Params, p: Params) -> Params:
        hess_prod: Vector = testing.rosen_hess_prod(params.data, p.data)
        return Params(data=hess_prod)

    def hess_quad(params: Params, p: Params) -> Scalar:
        return testing.rosen_hess_quad(params.data, p.data)

    def value_and_grad(params: Params) -> tuple[Scalar, Params]:
        value: Scalar
        grad: Vector
        value, grad = testing.rosen_value_and_grad(params.data)
        return value, Params(data=grad)

    def grad_and_hess_diag(params: Params) -> tuple[Params, Params]:
        grad: Vector
        hess_diag: Vector
        grad, hess_diag = testing.rosen_grad_and_hess_diag(params.data)
        return Params(data=grad), Params(data=hess_diag)

    return Objective(
        fun=fun,
        grad=grad,
        hess_diag=hess_diag,
        hess_prod=hess_prod,
        hess_quad=hess_quad,
        value_and_grad=value_and_grad,
        grad_and_hess_diag=grad_and_hess_diag,
    )


def check_optimizer(objective: Objective, optimizer: Optimizer) -> None:
    params: Params = Params(data=jnp.zeros((7,)))
    solution: Optimizer.Solution = optimizer.minimize(objective, params)
    assert solution.success
    params: Params = solution.params
    np.testing.assert_allclose(params.data, jnp.ones((7,)), atol=1e-3)


def test_optax_adam(objective: Objective) -> None:
    optimizer = Optax(
        optax.adam(learning_rate=1e-2), max_steps=1000, gtol=1e-4, jit=True, timer=True
    )
    check_optimizer(objective, optimizer)


def test_pncg(objective: Objective) -> None:
    optimizer = PNCG(rtol=1e-9, jit=True, timer=True)
    check_optimizer(objective, optimizer)


def test_scipy_lbfgsb(objective: Objective) -> None:
    objective = Objective(value_and_grad=objective.value_and_grad)
    optimizer = ScipyOptimizer(method="L-BFGS-B", jit=True, timer=True)
    check_optimizer(objective, optimizer)
