# Peach

Peach is a small JAX-first toolbox for optimization experiments. It provides
protocol-based objective interfaces, nonlinear optimization drivers, Hessian
product helpers, and linear solver wrappers that report residuals in a common
solution object.

## Install

```bash
uv add liblaf-peach
```

## Optimize A Small Problem

Optimizers operate on protocol-shaped problem objects. A `PNCG` problem provides
a line-search trial hook plus objective, gradient, Hessian-diagonal, and Hessian
quadratic-form methods. The model state passed into each PNCG step represents the
current parameters; accepted line-search trials become the next model state.

```python
import jax.numpy as jnp

from liblaf.peach.optim.pncg import PNCG


class QuadraticProblem:
    def __init__(self, target):
        self.target = target

    def before_trial(self, state, params, /):
        return params

    def fun(self, state, /):
        residual = state - self.target
        return 0.5 * jnp.vdot(residual, residual)

    def grad(self, state, /):
        return state - self.target

    def hess_diag(self, state, /):
        return jnp.ones_like(state)

    def hess_quad(self, state, direction, /):
        return jnp.vdot(direction, direction)


params = jnp.asarray([0.0])
problem = QuadraticProblem(target=jnp.asarray([3.0]))
solution, state = PNCG().minimize(problem, params, params)

print(solution.params)
```

The optimizer keeps model state separate from optimizer state. Capabilities such
as `max_step_size` and `callback` are optional: Peach only calls hooks explicitly
implemented on the concrete problem. For line-search problems, `max_step_size`
receives the proposed displacement and returns a safe fraction of that
displacement in `[0, 1]`.

## Compute Hessian Products

`hess_prod` evaluates Hessian-vector products with JAX forward-over-reverse
automatic differentiation, without materializing a dense Hessian. The
`RosenObjective` helper provides a compact Rosenbrock objective for tests,
examples, and derivative checks.

```python
import jax.numpy as jnp

from liblaf.peach.math import hess_prod

matrix = jnp.asarray([[3.0, 1.0], [1.0, 2.0]])


def quadratic(x):
    return 0.5 * jnp.vdot(x, matrix @ x)


direction = jnp.asarray([2.0, 0.5])
print(hess_prod(quadratic, jnp.asarray([1.0, -1.0]), direction))
```

## Solve Linear Systems

The linear-solver API uses a small protocol: provide `b` and `matvec`, then
optionally add transpose and preconditioner hooks. `JaxCG` wraps
`jax.scipy.sparse.linalg.cg`; `FallbackSolver` can try a configured list of
solvers and keep residual diagnostics for each attempt.

See the [API reference](reference/liblaf/peach/README.md) for the full module
surface.
