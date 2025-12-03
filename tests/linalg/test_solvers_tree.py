import hypothesis
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Key

from liblaf.peach import tree
from liblaf.peach.linalg import (
    CompositeSolver,
    CupyCG,
    CupyMinRes,
    JaxBiCGStab,
    JaxCG,
    JaxGMRES,
    LinearSolver,
    LinearSystem,
    ScipyBiCG,
    ScipyBiCGStab,
    ScipyCG,
    ScipyMinRes,
)

type Vector = Float[Array, " free"]


@tree.define
class Params:
    data: Vector = tree.array()
    meta: str = "foo"


@pytest.fixture(scope="package")
def system() -> LinearSystem:
    def matvec(x: Params) -> Params:
        return x

    def preconditioner(x: Params) -> Params:
        return x

    rng: np.random.Generator = np.random.default_rng()
    b: Params = Params(data=jnp.asarray(rng.random((3,))))
    return LinearSystem(
        matvec,
        b,
        rmatvec=matvec,
        preconditioner=preconditioner,
        rpreconditioner=preconditioner,
    )


def check_solver(solver: LinearSolver, system: LinearSystem, seed: int) -> None:
    assert system.matvec is not None
    key: Key = jax.random.key(seed)
    subkey: Key
    key, subkey = jax.random.split(key)
    x: Params = Params(
        data=jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)
    )
    key, subkey = jax.random.split(key)
    x0: Params = Params(
        data=jax.random.uniform(subkey, shape=(3,), minval=-1.0, maxval=1.0)
    )
    system.b = system.matvec(x)
    solution: LinearSolver.Solution = solver.solve(system, x0)
    b_actual: Params = system.matvec(solution.params)
    np.testing.assert_allclose(b_actual.data, system.b.data)


def seed() -> st.SearchStrategy[int]:
    return st.integers(min_value=-(2**31), max_value=2**31 - 1)


@hypothesis.given(seed=seed())
def test_composite(seed: int, system: LinearSystem) -> None:
    check_solver(CompositeSolver(), system, seed)


@hypothesis.given(seed=seed())
def test_cupy_cg(seed: int, system: LinearSystem) -> None:
    check_solver(CupyCG(), system, seed)


@hypothesis.given(seed=seed())
def test_cupy_minres(seed: int, system: LinearSystem) -> None:
    check_solver(CupyMinRes(), system, seed)


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
    check_solver(ScipyMinRes(), system, seed)
