import numpy as np
import scipy
import torch
from jaxtyping import Float
from torch import Tensor

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


class RosenProblem:
    def update(self, state: Vector, params: Vector, /) -> Vector:
        state: Vector = params
        return state

    def fun(self, x: Vector, /) -> Scalar:
        x: Vector = torch.as_tensor(x)
        return torch.as_tensor(scipy.optimize.rosen(x))

    def grad(self, x: Vector, /) -> Vector:
        x: Vector = torch.as_tensor(x)
        return torch.as_tensor(scipy.optimize.rosen_der(x))

    def hess_diag(self, x: Vector, /) -> Vector:
        x: Vector = torch.as_tensor(x)
        return torch.tensor(np.diagonal(scipy.optimize.rosen_hess(x)))

    def hess_prod(self, x: Vector, p: Vector, /) -> Vector:
        x: Vector = torch.as_tensor(x)
        p: Vector = torch.as_tensor(p)
        return torch.as_tensor(scipy.optimize.rosen_hess_prod(x, p))

    def hess_quad(self, x: Vector, p: Vector, /) -> Scalar:
        x: Vector = torch.as_tensor(x)
        p: Vector = torch.as_tensor(p)
        return torch.dot(p, self.hess_prod(x, p))
