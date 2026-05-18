from typing import override

import attrs
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.peach.linalg.base import (
    BaseProblem,
    LinearSolver,
    Problem,
    Result,
    Solution,
    State,
    Stats,
)
from liblaf.peach.linalg.fallback import FallbackSolver

type Vector = Float[Tensor, " N"]


@attrs.define
class IdentityProblem(Problem):
    rhs: Vector

    @property
    @override
    def b(self) -> Vector:
        return self.rhs

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

    @override
    def rprecondition(self, x: Vector) -> Vector:
        return 3.0 * x


@attrs.define
class FixedState(State):
    params: Vector


@attrs.define
class FixedStats(Stats):
    pass


@attrs.define
class FixedSolver(LinearSolver[FixedState, FixedStats]):
    solution_params: Vector
    result: Result
    init_params: list[Vector] = attrs.field(factory=list)

    State = FixedState
    Stats = FixedStats
    type Solution = LinearSolver.Solution[FixedState, FixedStats]

    @override
    def init(self, problem: BaseProblem, params: Vector) -> FixedState:
        del problem
        self.init_params.append(params.clone())
        return FixedState(params=self.solution_params.clone())

    @override
    def compute(self, problem: BaseProblem, state: FixedState) -> Result:
        del problem, state
        return self.result


def test_fallback_solver_stops_on_first_success_and_tracks_best_residual() -> None:
    problem = IdentityProblem(rhs=torch.tensor([1.0, 2.0]))
    first = FixedSolver(
        solution_params=torch.tensor([10.0, 10.0]), result=Result.UNKNOWN_ERROR
    )
    second = FixedSolver(
        solution_params=torch.tensor([1.0, 2.0]), result=Result.SUCCESS
    )
    third = FixedSolver(solution_params=torch.tensor([0.0, 0.0]), result=Result.SUCCESS)
    solver = FallbackSolver(solvers=[first, second, third])

    solution = solver.solve(problem, torch.zeros(2))

    assert solution.result is Result.SUCCESS
    assert solution.success
    torch.testing.assert_close(solution.params, torch.tensor([1.0, 2.0]))
    assert len(solution.state.solutions) == 2
    assert len(third.init_params) == 0
    torch.testing.assert_close(
        solution.state.absolute_residuals,
        torch.tensor([torch.linalg.vector_norm(torch.tensor([9.0, 8.0])), 0.0]),
    )
    assert int(solution.state.best_index) == 1


def test_fallback_solver_returns_lowest_residual_when_all_solvers_fail() -> None:
    problem = IdentityProblem(rhs=torch.tensor([1.0, 2.0]))
    worse = FixedSolver(
        solution_params=torch.tensor([5.0, 2.0]), result=Result.BREAKDOWN
    )
    better = FixedSolver(
        solution_params=torch.tensor([1.0, 3.0]), result=Result.MAX_STEPS_REACHED
    )
    solver = FallbackSolver(solvers=[worse, better])

    solution = solver.solve(problem, torch.zeros(2))

    assert solution.result is Result.MAX_STEPS_REACHED
    assert not solution.success
    torch.testing.assert_close(solution.params, torch.tensor([1.0, 3.0]))
    expected_relative = torch.tensor([4.0, 1.0]) / torch.linalg.vector_norm(problem.b)
    torch.testing.assert_close(solution.state.relative_residuals, expected_relative)
    assert int(solution.state.best_index) == 1


def test_cupy_solver_omits_protocol_stub_preconditioner() -> None:
    from liblaf.peach.linalg.cupy import CupyCG

    problem = IdentityProblem(rhs=torch.tensor([1.0, 2.0]))
    solver = CupyCG(maxiter=5, rtol=1e-6, atol=1e-9)

    options = solver._options(problem)  # noqa: SLF001

    assert options == {"maxiter": 5, "M": None, "atol": 1e-9, "rtol": 1e-6}


def test_cupy_solver_uses_overridden_preconditioner() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.is_available():
        pytest.skip("CuPy CUDA runtime is unavailable")
    from liblaf.peach.linalg.cupy import CupyCG

    problem = PreconditionedIdentityProblem(rhs=torch.tensor([1.0, 2.0]))
    solver = CupyCG()

    preconditioner = solver._options(problem)["M"]  # noqa: SLF001

    cp.testing.assert_allclose(
        preconditioner.matvec(cp.asarray([3.0, 4.0])),
        cp.asarray([6.0, 8.0]),
    )
    cp.testing.assert_allclose(
        preconditioner.rmatvec(cp.asarray([3.0, 4.0])),
        cp.asarray([9.0, 12.0]),
    )


def test_cupy_cg_solves_identity_system_when_cuda_is_available() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.is_available() or not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from liblaf.peach.linalg.cupy import CupyCG

    rhs = torch.tensor([1.0, 2.0], device="cuda")
    problem = IdentityProblem(rhs=rhs)
    solver = CupyCG(maxiter=4, rtol=1e-12, atol=0.0)

    solution: Solution = solver.solve(problem, torch.zeros_like(rhs))

    assert solution.result is Result.SUCCESS
    assert solution.success
    torch.testing.assert_close(solution.params, rhs)
    torch.testing.assert_close(
        solution.stats.absolute_residual, torch.tensor(0.0, device="cuda")
    )
    torch.testing.assert_close(
        solution.stats.relative_residual, torch.tensor(0.0, device="cuda")
    )
