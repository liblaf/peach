import numpy as np
import scipy
import torch
from jaxtyping import Float
from torch import Tensor

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


class RosenProblem:
    """Rosenbrock objective exposed through Peach optimizer hooks.

    The implementation delegates value and derivative calculations to SciPy and
    converts inputs and outputs to torch tensors.

    Examples:
        >>> import torch
        >>> from liblaf.peach.testing import RosenProblem
        >>> problem = RosenProblem()
        >>> x = torch.tensor([1.0, 1.0, 1.0])
        >>> torch.allclose(problem.fun(x), torch.tensor(0.0))
        True
        >>> torch.allclose(problem.grad(x), torch.zeros(3))
        True
    """

    def update(self, state: Vector, params: Vector, /) -> Vector:
        """Return `params` as the next model state."""
        state: Vector = params
        return state

    def fun(self, x: Vector, /) -> Scalar:
        """Evaluate the Rosenbrock objective."""
        x: Vector = torch.as_tensor(x)
        return torch.as_tensor(scipy.optimize.rosen(x))

    def grad(self, x: Vector, /) -> Vector:
        """Evaluate the Rosenbrock gradient."""
        x: Vector = torch.as_tensor(x)
        return torch.as_tensor(scipy.optimize.rosen_der(x))

    def hess_diag(self, x: Vector, /) -> Vector:
        """Evaluate the diagonal of the Rosenbrock Hessian."""
        x: Vector = torch.as_tensor(x)
        return torch.tensor(np.diagonal(scipy.optimize.rosen_hess(x)))

    def hess_prod(self, x: Vector, p: Vector, /) -> Vector:
        """Evaluate a Rosenbrock Hessian-vector product."""
        x: Vector = torch.as_tensor(x)
        p: Vector = torch.as_tensor(p)
        return torch.as_tensor(scipy.optimize.rosen_hess_prod(x, p))

    def hess_quad(self, x: Vector, p: Vector, /) -> Scalar:
        """Evaluate `p.T @ H(x) @ p`."""
        x: Vector = torch.as_tensor(x)
        p: Vector = torch.as_tensor(p)
        return torch.dot(p, self.hess_prod(x, p))
