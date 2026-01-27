import jarp
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from liblaf.peach import testing
from liblaf.peach.optim import Objective, Optimizer, ScipyOptimizer
from liblaf.peach.optim.optax._optax import Optax
from liblaf.peach.optim.pncg._pncg import PNCG
from liblaf.peach.transforms import FixedTransform
from liblaf.peach.transforms._unravel import UnravelTransform

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@jarp.define
class Params:
    data: Vector


@jarp.define
class Model:
    def update(self, _state: Params, params: Params) -> Params:
        return params

    def fun(self, params: Params) -> Scalar:
        return testing.rosen(params.data)

    def grad(self, params: Params) -> Params:
        grad_flat: Vector = testing.rosen_grad(params.data)
        return Params(data=grad_flat)

    def hess_diag(self, params: Params) -> Params:
        hess_diag_flat: Vector = testing.rosen_hess_diag(params.data)
        return Params(data=hess_diag_flat)

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
    objective: Objective,
    optimizer: Optimizer,
    *,
    atol: float = 1e-3,
    fixed: bool = True,
) -> None:
    params: Params = Params(data=jnp.zeros((7,)))
    if fixed:
        fixed_mask: Params = Params(data=jnp.zeros((7,), bool).at[::2].set(True))
        fixed_values: Params = Params(data=jnp.ones((7,)))
        objective.transform = FixedTransform(fixed_mask, fixed_values)
    else:
        structure: jarp.Structure[Params]
        _params_flat, structure = jarp.ravel(params)
        objective.transform = UnravelTransform(structure)

    model_state: Params = params
    solution: Optimizer.Solution
    solution, model_state = optimizer.minimize(objective, model_state, params)
    assert solution.success
    params_flat: Vector = solution.params
    params_tree: Params = objective.transform.forward_primals(params_flat)
    np.testing.assert_allclose(params_tree.data, jnp.ones((7,)), atol=atol)


def test_optax_adam() -> None:
    objective: Objective = Objective(
        update=Model.update, value_and_grad=Model.value_and_grad, args=(Model(),)
    )
    optimizer = Optax(
        optax.adam(learning_rate=1e-2),
        max_steps=jnp.asarray(1000),
        rtol=jnp.asarray(0.0),
        jit=False,
    )
    check_optimizer(objective, optimizer, atol=1e-2)


def test_pncg() -> None:
    objective: Objective = Objective(
        update=Model.update,
        grad=Model.grad,
        hess_diag=Model.hess_diag,
        hess_quad=Model.hess_quad,
        args=(Model(),),
    )
    optimizer = PNCG(rtol=jnp.asarray(1e-9), jit=True)
    # due to indefinite Hessian
    # PNCG is not able to solve rosen with fixed constraints
    check_optimizer(objective, optimizer, fixed=False)


def test_scipy_lbfgsb() -> None:
    objective: Objective = Objective(
        update=Model.update, value_and_grad=Model.value_and_grad, args=(Model(),)
    )
    optimizer = ScipyOptimizer(method="L-BFGS-B")
    check_optimizer(objective, optimizer)
