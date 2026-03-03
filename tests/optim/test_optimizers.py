import jarp
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from liblaf.peach import testing
from liblaf.peach.optim import PNCG, Objective, Optax, Optimizer, ScipyOptimizer

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@jarp.define
class Model:
    def update(self, _state: Vector, params: Vector) -> Vector:
        return params

    def fun(self, params: Vector) -> Scalar:
        return testing.rosen(params)

    def grad(self, params: Vector) -> Vector:
        return testing.rosen_grad(params)

    def hess_diag(self, params: Vector) -> Vector:
        return testing.rosen_hess_diag(params)

    def hess_prod(self, params: Vector, p: Vector) -> Vector:
        return testing.rosen_hess_prod(params, p)

    def hess_quad(self, params: Vector, p: Vector) -> Scalar:
        return testing.rosen_hess_quad(params, p)

    def value_and_grad(self, params: Vector) -> tuple[Scalar, Vector]:
        return testing.rosen_value_and_grad(params)


def check_optimizer(optimizer: Optimizer, *, atol: float = 1e-3) -> None:
    objective: Objective = Model()
    params: Vector = jnp.zeros((7,))
    model_state: Vector = params
    solution: Optimizer.Solution
    solution, model_state = optimizer.minimize(objective, model_state, params)
    assert solution.success
    np.testing.assert_allclose(solution.params, jnp.ones((7,)), atol=atol)


def test_optax_adam() -> None:
    optimizer = Optax(
        optax.adam(learning_rate=1e-2),
        max_steps=jnp.asarray(1000),
        rtol=jnp.asarray(0.0),
    )
    check_optimizer(optimizer, atol=1e-2)


def test_pncg() -> None:
    optimizer = PNCG(rtol=jnp.asarray(1e-9))
    check_optimizer(optimizer)


def test_scipy_lbfgsb() -> None:
    optimizer = ScipyOptimizer(method="L-BFGS-B")
    check_optimizer(optimizer)
