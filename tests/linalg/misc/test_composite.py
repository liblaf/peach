import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach.linalg import CompositeSolver, JaxCG, JaxGMRES, LinearSystem

type Vector = Float[Array, " free"]


def test_composite() -> None:
    A: Array = jnp.asarray([[1.0, 10.0], [0.0, 1.0]])  # noqa: N806

    def matvec(v: Vector) -> Vector:
        return A @ v

    x: Vector = jnp.ones((2,))
    b: Vector = matvec(x)
    system = LinearSystem(matvec, b)
    x0: Vector = jnp.zeros((2,))

    cg = JaxCG(jit=True, timer=True)
    cg_solution: JaxCG.Solution = cg.solve(system, x0)
    assert not cg_solution.success

    solver = CompositeSolver([JaxCG(), JaxGMRES()], jit=True, timer=True)
    solution: CompositeSolver.Solution = solver.solve(system, x0)
    assert solution.success
    np.testing.assert_allclose(solution.params, x)
