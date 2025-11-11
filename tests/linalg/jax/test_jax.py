import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach.linalg import LinearOperator, LinearSolution
from liblaf.peach.linalg.jax import JaxBiCGStab, JaxCG, JaxGMRES

type Vector = Float[Array, " free"]


def matvec(v: Array) -> Array:
    return v


def test_bicgstab() -> None:
    op = LinearOperator(matvec=matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxBiCGStab()
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


def test_cg() -> None:
    op = LinearOperator(matvec=matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxCG()
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


def test_gmres() -> None:
    op = LinearOperator(matvec=matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxGMRES()
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)
