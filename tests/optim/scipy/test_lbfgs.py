import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import optim, testing, tree_utils


def test_scipy_lbfgs() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=testing.rosen_value_and_grad
    )
    params: Float[Array, " N"] = jnp.zeros((7,))
    optimizer: optim.Optimizer = optim.ScipyOptimizer(method="L-BFGS-B", tol=1e-10)
    solution: optim.OptimizeSolution = optimizer.minimize(objective, params)
    assert solution.success
    np.testing.assert_allclose(solution.params, jnp.ones((7,)))


@tree_utils.define
class Params:
    x: Float[Array, " N"]


def rosen_value_and_grad_tree(
    params: Params,
) -> tuple[Float[Array, ""], Params]:
    value: Float[Array, ""]
    grad: Float[Array, " N"]
    value, grad = testing.rosen_value_and_grad(params.x)
    return value, Params(grad)


def callback(state: optim.ScipyState, _stats: optim.ScipyStats) -> None:
    assert isinstance(state.params, Params)


def test_scipy_lbfgs_tree() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=rosen_value_and_grad_tree
    )
    params: Params = Params(jnp.zeros((7,)))
    optimizer: optim.Optimizer = optim.ScipyOptimizer(method="L-BFGS-B", tol=1e-10)
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective, params, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((7,)))
