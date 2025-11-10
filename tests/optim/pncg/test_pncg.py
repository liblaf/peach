import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import optim, testing


def test_pncg() -> None:
    objective: optim.Objective = optim.Objective(
        grad_and_hess_diag=testing.rosen_grad_and_hess_diag,
        hess_quad=testing.rosen_hess_quad,
    )
    params: Float[Array, " N"] = jnp.zeros((7,))
    optimizer: optim.Optimizer = optim.PNCG(rtol=1e-16)
    solution: optim.OptimizeSolution = optimizer.minimize(objective, params)
    assert solution.success
    np.testing.assert_allclose(solution.params, jnp.ones((7,)))
