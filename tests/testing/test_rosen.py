import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.peach.testing import RosenProblem

type Vector = Float[Tensor, " N"]


def test_rosen_objective_minimum_has_zero_value_and_gradient() -> None:
    objective = RosenProblem()
    x: Vector = torch.tensor([1.0, 1.0, 1.0])

    torch.testing.assert_close(objective.fun(x), torch.tensor(0.0))
    torch.testing.assert_close(objective.grad(x), torch.zeros(3))
    torch.testing.assert_close(
        objective.hess_diag(x), torch.tensor([802.0, 1002.0, 200.0])
    )


def test_rosen_hess_quad_matches_explicit_hessian_product() -> None:
    objective = RosenProblem()
    x: Vector = torch.tensor([-1.0, 1.5, 0.5])
    p: Vector = torch.tensor([0.25, -0.5, 1.0])

    hessian_p = objective.hess_prod(x, p)

    torch.testing.assert_close(objective.hess_quad(x, p), torch.dot(p, hessian_p))
