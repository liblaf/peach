import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import testing, tree
from liblaf.peach.functools import Objective
from liblaf.peach.optim import ScipyOptimizer
from liblaf.peach.optim.abc import Optimizer
from liblaf.peach.transforms._fixed import FixedTransform

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@tree.define
class Params:
    data: Vector = tree.array()


@tree.define
class Model:
    def fun(self, params: Params) -> Scalar:
        return testing.rosen(params.data)

    def grad(self, params: Params) -> Params:
        grad_flat: Vector = testing.rosen_grad(params.data)
        return Params(data=grad_flat)

    def hess_prod(self, params: Params, p: Params) -> Params:
        hess_prod_flat: Vector = testing.rosen_hess_prod(params.data, p.data)
        return Params(data=hess_prod_flat)

    def hess_quad(self, params: Params, p: Params) -> Scalar:
        return testing.rosen_hess_quad(params.data, p.data)

    def value_and_grad(self, params: Params) -> tuple[Scalar, Params]:
        value: Scalar
        grad_flat: Vector
        value, grad_flat = testing.rosen_value_and_grad(params.data)
        grad: Params = Params(data=grad_flat)
        return value, grad


def check_optimizer(
    objective: Objective, optimizer: Optimizer, *, atol: float = 1e-3
) -> None:
    model: Model = Model()
    params: Params = Params(data=jnp.zeros((7,)))
    fixed_mask: Params = Params(data=jnp.zeros((7,), bool).at[::2].set(True))
    fixed_values: Params = Params(data=jnp.ones((7,)))
    fixed_transform: FixedTransform = FixedTransform(fixed_mask, fixed_values)

    objective = objective.partial(model)
    solution: Optimizer.Solution = optimizer.minimize(
        objective, params, transform=fixed_transform
    )
    assert solution.success
    params: Params = solution.params
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
