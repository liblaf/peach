import jarp
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from liblaf import peach
from liblaf.peach.linalg import (
    CupyMinRes,
    FallbackSolver,
    JaxCG,
    LinearSolver,
)

type Vector = Float[Array, " N"]


@jarp.define
class System:
    b: Vector = jarp.field(factory=lambda: jnp.ones((3,)), kw_only=True)

    def matvec(self, x: Vector) -> Vector:
        return x

    def preconditioner(self, x: Vector) -> Vector:
        return x


def check_solver(solver: LinearSolver) -> None:
    system = System()
    x0: Vector = jnp.zeros_like(system.b)
    solution: LinearSolver.Solution = solver.solve(system, x0)
    assert solution.success
    b_actual: Vector = system.matvec(solution.params)
    np.testing.assert_allclose(b_actual, system.b)


@pytest.mark.skipif(not peach.cuda.is_available(), reason="CUDA not available")
def test_cupy_minres() -> None:
    check_solver(CupyMinRes())


@pytest.mark.skipif(not peach.cuda.is_available(), reason="CUDA not available")
def test_fallback() -> None:
    check_solver(FallbackSolver([JaxCG(), CupyMinRes()]))


def test_jax_cg() -> None:
    check_solver(JaxCG())
