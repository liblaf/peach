# Peach

Peach is a small torch-based toolbox for optimization and linear-solver
experiments. It provides protocol-shaped objective interfaces, nonlinear
optimization drivers, SciPy adapters, and CuPy linear-solver wrappers that report
residuals through a shared solution object.

## Install

```bash
uv add liblaf-peach
```

## Optimize A Small Problem

Optimizers operate on problem objects rather than inheritance-heavy base classes.
A [`Pncg`](reference/liblaf/peach/optim/pncg/README.md) problem provides an
`update` hook plus objective, gradient, Hessian-diagonal, and Hessian
quadratic-form methods. The model state passed into each step represents the
current parameters; accepted line-search trials mutate that model state in
place.

```python
import torch

from liblaf.peach.optim.pncg import Pncg


class QuadraticProblem:
    def __init__(self, target):
        self.target = target

    def update(self, state, params, /):
        state.copy_(params)

    def fun(self, state, /):
        residual = state - self.target
        return 0.5 * torch.dot(residual, residual)

    def grad(self, state, /):
        return state - self.target

    def hess_diag(self, state, /):
        return torch.ones_like(state)

    def hess_quad(self, state, direction, /):
        return torch.dot(direction, direction)


params = torch.tensor([0.0])
model_state = params.clone()
problem = QuadraticProblem(target=torch.tensor([3.0]))
solution = Pncg().minimize(problem, model_state, params)

print(solution.params)
print(model_state)
```

The optimizer keeps model state separate from optimizer state. A step mutates
the model state to the accepted trial and updates the optimizer state in place,
so callbacks and the final solution observe the same parameter vector.
Capabilities such as `max_step_size` and `callback` are optional: Peach only
calls hooks explicitly implemented on the concrete problem. For line-search
problems, `max_step_size` receives the proposed displacement and returns a safe
fraction of that displacement in `[0, 1]`.

## Solve Linear Systems

The linear-solver API uses a small protocol: provide `b` and `matvec`, then
optionally add transpose and preconditioner hooks.
[`CupyCG`](reference/liblaf/peach/linalg/cupy/README.md) and
[`CupyMinRes`](reference/liblaf/peach/linalg/cupy/README.md) wrap
`cupyx.scipy.sparse.linalg` routines while accepting and returning torch tensors.
[`FallbackSolver`](reference/liblaf/peach/linalg/fallback/README.md) can try a
configured list of solvers and keep residual diagnostics for each attempt.

See the [API reference](reference/liblaf/peach/README.md) for the full module
surface.
