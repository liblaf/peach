from typing import override

import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.peach.optim.base import Problem
from liblaf.peach.optim.scipy import ScipyOptimizer, ScipyState

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


class QuadraticProblem(Problem[Vector]):
    def __init__(self, target: Vector) -> None:
        self.target = target

    @override
    def update(self, state: Vector, params: Vector, /) -> None:
        state.copy_(params)

    @override
    def fun(self, state: Vector, /) -> Scalar:
        residual: Vector = state - self.target
        return 0.5 * torch.dot(residual, residual)

    @override
    def grad(self, state: Vector, /) -> Vector:
        return state - self.target


def test_scipy_optimizer_returns_solution_and_mutates_model_state() -> None:
    params: Vector = torch.tensor([0.0], dtype=torch.float64)
    model_state: Vector = params.clone()
    problem = QuadraticProblem(target=torch.tensor([3.0], dtype=torch.float64))
    optimizer = ScipyOptimizer(method="BFGS", options={"gtol": 1e-9})

    solution = optimizer.minimize(problem, model_state, params)

    assert solution.success
    assert isinstance(solution.state, ScipyState)
    torch.testing.assert_close(
        solution.params, torch.tensor([3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(model_state, solution.params)
    torch.testing.assert_close(params, torch.tensor([0.0], dtype=torch.float64))
