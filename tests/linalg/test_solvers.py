import hypothesis
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Key

from liblaf.peach.linalg import (
    CompositeSolver,
    JaxBiCGStab,
    JaxCG,
    JaxGMRES,
    LinearSolver,
    LinearSystem,
    ScipyBiCG,
    ScipyBiCGStab,
    ScipyCG,
    ScipyMINRES,
)

type Vector = Float[Array, " free"]


@pytest.fixture(scope="package")
def system() -> LinearSystem:
    def matvec(x: Vector) -> Vector:
        return x

    rng: np.random.Generator = np.random.default_rng()
    b: Vector = jnp.asarray(rng.random((3,)))
    return LinearSystem(matvec, b, rmatvec=matvec)


def check_solver(solver: LinearSolver, system: LinearSystem, seed: int) -> None:
    assert system.matvec is not None
    key: Key = jax.random.key(seed)
    subkey: Key
    key, subkey = jax.random.split(key)
    x: Vector = jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)
    key, subkey = jax.random.split(key)
    x0: Vector = jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)
    system.b = system.matvec(x)
    solution: LinearSolver.Solution = solver.solve(system, x0)
    b_actual: Vector = system.matvec(solution.params)
    np.testing.assert_allclose(b_actual, system.b_flat)


def seed() -> st.SearchStrategy[int]:
    return st.integers(min_value=-(2**31), max_value=2**31 - 1)


@hypothesis.given(seed=seed())
def test_composite(seed: int, system: LinearSystem) -> None:
    check_solver(CompositeSolver(), system, seed)


@hypothesis.given(seed=seed())
def test_jax_bicgstab(seed: int, system: LinearSystem) -> None:
    check_solver(JaxBiCGStab(), system, seed)


@hypothesis.given(seed=seed())
def test_jax_cg(seed: int, system: LinearSystem) -> None:
    check_solver(JaxCG(), system, seed)


@hypothesis.given(seed=seed())
def test_jax_gmres(seed: int, system: LinearSystem) -> None:
    check_solver(JaxGMRES(), system, seed)


@hypothesis.given(seed=seed())
def test_scipy_bicg(seed: int, system: LinearSystem) -> None:
    check_solver(ScipyBiCG(), system, seed)


@hypothesis.given(seed=seed())
def test_scipy_bicgstab(seed: int, system: LinearSystem) -> None:
    check_solver(ScipyBiCGStab(), system, seed)


@hypothesis.given(seed=seed())
def test_scipy_cg(seed: int, system: LinearSystem) -> None:
    check_solver(ScipyCG(), system, seed)


@hypothesis.given(seed=seed())
def test_scipy_minres(seed: int, system: LinearSystem) -> None:
    check_solver(ScipyMINRES(), system, seed)
