import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach import math

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.frozen_static
class RosenObjective:
    """Rosenbrock objective with derivative helpers.

    The state is the parameter vector itself. `fun`, `grad`, `hess_diag`,
    `hess_prod`, `hess_quad`, and `value_and_grad` make this class useful as a
    small deterministic derivative fixture.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from liblaf.peach.testing import RosenObjective
        >>> objective = RosenObjective()
        >>> x = jnp.asarray([1.0, 1.0, 1.0])
        >>> float(objective.fun(x))
        0.0
        >>> np.asarray(objective.grad(x)).tolist()
        [0.0, 0.0, 0.0]
    """

    def update(self, _state: Vector, params: Vector, /) -> Vector:
        """Return `params` as the new model state."""
        return params

    @jax.jit(inline=True)
    def fun(self, x: Vector, /) -> Scalar:
        """Evaluate the Rosenbrock function."""
        return jnp.sum(
            100.0 * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(1.0 - x[:-1])
        )

    @jax.jit(inline=True)
    def grad(self, x: Vector, /) -> Vector:
        """Evaluate the gradient of [`fun`][liblaf.peach.testing.RosenObjective.fun]."""
        return jax.grad(self.fun)(x)

    @jax.jit(inline=True)
    def hess_diag(self, x: Vector, /) -> Vector:
        """Evaluate the diagonal of the dense Hessian."""
        return jnp.diagonal(jax.hessian(self.fun)(x))

    @jax.jit(inline=True)
    def hess_prod(self, x: Vector, p: Vector, /) -> Vector:
        """Evaluate the Hessian-vector product at `x` along `p`."""
        return math.hess_prod(self.fun, x, p)

    @jax.jit(inline=True)
    def hess_quad(self, x: Vector, p: Vector, /) -> Scalar:
        """Evaluate the quadratic form `p.T @ H(x) @ p`."""
        return jnp.vdot(p, self.hess_prod(x, p))

    @jax.jit(inline=True)
    def value_and_grad(self, x: Vector, /) -> tuple[Scalar, Vector]:
        """Evaluate the Rosenbrock value and gradient together."""
        return jax.value_and_grad(self.fun)(x)
