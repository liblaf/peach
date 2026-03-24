import jarp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import math

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@jarp.frozen_static
class RosenObjective:
    def update(self, _state: Vector, params: Vector, /) -> Vector:
        return params

    @jarp.jit(inline=True)
    def fun(self, x: Vector, /) -> Scalar:
        return jnp.sum(
            100.0 * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(1.0 - x[:-1])
        )

    @jarp.jit(inline=True)
    def grad(self, x: Vector, /) -> Vector:
        return jax.grad(self.fun)(x)

    @jarp.jit(inline=True)
    def hess_diag(self, x: Vector, /) -> Vector:
        return jnp.diagonal(jax.hessian(self.fun)(x))

    @jarp.jit(inline=True)
    def hess_prod(self, x: Vector, p: Vector, /) -> Vector:
        return math.hess_prod(self.fun, x, p)

    @jarp.jit(inline=True)
    def hess_quad(self, x: Vector, p: Vector, /) -> Scalar:
        return jnp.vdot(p, self.hess_prod(x, p))

    @jarp.jit(inline=True)
    def value_and_grad(self, x: Vector, /) -> tuple[Scalar, Vector]:
        return jax.value_and_grad(self.fun)(x)
