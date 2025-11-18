import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import optim, testing, tree
from liblaf.peach.constraints import Constraint, FixedConstraint


def test_scipy_lbfgs() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=testing.rosen_value_and_grad
    )
    params: Float[Array, " N"] = jnp.zeros((7,))
    optimizer: optim.Optimizer = optim.ScipyOptimizer(method="L-BFGS-B", tol=1e-10)
    solution: optim.OptimizeSolution = optimizer.minimize(objective, params)
    assert solution.success
    np.testing.assert_allclose(solution.params, jnp.ones((7,)))


@tree.define
class Params:
    x: Float[Array, " N"]
    static_field: str = "foo"


def rosen_value_and_grad_tree(
    params: Params,
) -> tuple[Float[Array, ""], Params]:
    value: Float[Array, ""]
    grad: Float[Array, " N"]
    value, grad = testing.rosen_value_and_grad(params.x)
    return value, Params(grad)


def callback(
    state: optim.ScipyOptimizer.State, _stats: optim.ScipyOptimizer.Stats
) -> None:
    assert isinstance(state.params, Params)
    assert state.params.static_field == "foo"


def test_scipy_lbfgs_tree() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=rosen_value_and_grad_tree
    )
    params: Params = Params(jnp.zeros((7,)))
    optimizer: optim.Optimizer = optim.ScipyOptimizer(
        method="L-BFGS-B", tol=1e-10, jit=True, timer=True
    )
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective, params, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((7,)))


def test_scipy_lbfgs_tree_fixed() -> None:
    objective: optim.Objective = optim.Objective(
        value_and_grad=rosen_value_and_grad_tree
    )
    params: Params = Params(jnp.zeros((9,)))
    fixed_mask: Params = Params(jnp.zeros((9,), jnp.bool))
    params.x = params.x.at[-2:].set(1.0)
    fixed_mask.x = fixed_mask.x.at[-2:].set(True)
    constraints: list[Constraint] = [FixedConstraint(mask=fixed_mask)]

    optimizer: optim.Optimizer = optim.ScipyOptimizer(
        method="L-BFGS-B", tol=1e-10, jit=True, timer=True
    )
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective, params, constraints=constraints, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((9,)))
