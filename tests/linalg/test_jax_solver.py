from typing import Any, override

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import Problem
from liblaf.peach.linalg.jax import JaxSolver

type Vector = Float[Array, " N"]


class IdentityProblem(Problem):
    @property
    @override
    def b(self) -> Vector:
        return jnp.asarray([1.0, 2.0])

    @override
    def matvec(self, x: Vector) -> Vector:
        return x

    @override
    def rmatvec(self, x: Vector) -> Vector:
        return x


class PreconditionedIdentityProblem(IdentityProblem):
    @override
    def precondition(self, x: Vector) -> Vector:
        return 2.0 * x


def test_jax_solver_omits_protocol_stub_preconditioner() -> None:
    captured_options: list[dict[str, Any]] = []

    class CapturingSolver(JaxSolver):
        @override
        def _wrapped(self, *args: Any, **kwargs: Any) -> tuple[Vector, None]:
            del args
            captured_options.append(kwargs)
            return jnp.asarray([1.0, 2.0]), None

    problem = IdentityProblem()
    solver = CapturingSolver()
    state = solver.init(problem, jnp.zeros(2))

    solver.compute(problem, state)

    assert captured_options == [
        {"tol": solver.rtol_primary, "atol": solver.atol_primary, "maxiter": 2}
    ]


def test_jax_solver_uses_overridden_preconditioner() -> None:
    captured_options: list[dict[str, Any]] = []

    class CapturingSolver(JaxSolver):
        @override
        def _wrapped(self, *args: Any, **kwargs: Any) -> tuple[Vector, None]:
            del args
            captured_options.append(kwargs)
            return jnp.asarray([1.0, 2.0]), None

    problem = PreconditionedIdentityProblem()
    solver = CapturingSolver()
    state = solver.init(problem, jnp.zeros(2))

    solver.compute(problem, state)

    [options] = captured_options
    np.testing.assert_allclose(
        np.asarray(options["M"](jnp.asarray([3.0, 4.0]))),
        np.asarray([6.0, 8.0]),
    )
