import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import optim, testing


def test_scipy_lbfgs() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=testing.rosen_value_and_grad
    )
    params: Float[Array, " N"] = jnp.zeros((7,))
    optimizer: optim.Optimizer = optim.ScipyOptimizer(method="L-BFGS-B", tol=1e-10)
    solution: optim.OptimizeSolution = optimizer.minimize(objective, params)
    assert solution.success
    np.testing.assert_allclose(solution.params, jnp.ones((7,)))
