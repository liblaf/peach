import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from liblaf.peach.optim import PNCG, Objective, Optax, Optimizer, ScipyOptimizer
from liblaf.peach.testing import RosenObjective

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


def check_optimizer(optimizer: Optimizer, *, atol: float = 1e-3) -> None:
    objective: Objective = RosenObjective()
    params: Vector = jnp.zeros((7,))
    model_state: Vector = params
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
