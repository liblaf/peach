import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import tree_utils
from liblaf.peach.linalg import LinearOperator, LinearSolution
from liblaf.peach.linalg.jax import JaxBiCGStab, JaxCG, JaxCompositeSolver, JaxGMRES

type Vector = Float[Array, " free"]


def matvec(v: Array) -> Array:
    return v


def test_bicgstab() -> None:
    op = LinearOperator(matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxBiCGStab(jit=True, timer=True)
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


def test_cg() -> None:
    op = LinearOperator(matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxCG(jit=True, timer=True)
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


def test_gmres() -> None:
    op = LinearOperator(matvec)
    x: Vector = jnp.ones((7,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((7,))
    solver = JaxGMRES(jit=True, timer=True)
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


def test_composite() -> None:
    A: Array = jnp.asarray([[1.0, 10.0], [0.0, 1.0]])  # noqa: N806
    op = LinearOperator(lambda v: A @ v)
    x: Vector = jnp.ones((2,))
    b: Vector = op(x)
    x0: Vector = jnp.zeros((2,))

    cg = JaxCG(jit=True, timer=True)
    cg_solution: LinearSolution = cg.solve(op, b, x0)
    assert not cg_solution.success

    solver = JaxCompositeSolver(jit=True, timer=True)
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)


@tree_utils.define
class Params:
    x: Vector = tree_utils.array()
    static: str = "foo"


def matvec_tree(params: Params) -> Params:
    return Params(x=params.x, static=params.static)


def test_cg_tree() -> None:
    op = LinearOperator(matvec_tree)
    x = Params(x=jnp.ones((7,)))
    b: Vector = op(x)
    x0 = Params(x=jnp.zeros((7,)))
    solver = JaxCG(jit=True, timer=True)
    solution: LinearSolution = solver.solve(op, b, x0)
    assert solution.success
    assert isinstance(solution.params, Params)
    np.testing.assert_allclose(solution.params.x, x.x)
