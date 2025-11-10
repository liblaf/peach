import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import optim, testing, tree_utils


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


@tree_utils.define
class Params:
    x: Float[Array, " N"]


def rosen_grad_and_hess_diag_tree(params: Params) -> tuple[Params, Params]:
    grad: Float[Array, " N"]
    hess_diag: Float[Array, " N"]
    grad, hess_diag = testing.rosen_grad_and_hess_diag(params.x)
    return Params(grad), Params(hess_diag)


def rosen_hess_quad_tree(params: Params, p: Params) -> Float[Array, " N"]:
    return testing.rosen_hess_quad(params.x, p.x)


def callback(state: optim.PNCGState, _stats: optim.PNCGStats) -> None:
    assert isinstance(state.params, Params)
    assert isinstance(state.grad, Params)
    assert isinstance(state.hess_diag, Params)
    assert isinstance(state.preconditioner, Params)
    assert isinstance(state.search_direction, Params)


def test_pncg_tree() -> None:
    objective: optim.Objective = optim.Objective(
        grad_and_hess_diag=rosen_grad_and_hess_diag_tree, hess_quad=rosen_hess_quad_tree
    )
    params: Params = Params(jnp.zeros((7,)))
    optimizer: optim.Optimizer = optim.PNCG(rtol=1e-16)
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective, params, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((7,)))
