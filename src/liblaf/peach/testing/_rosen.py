import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import math
from liblaf.peach.optim import Objective

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@jax.jit
def rosen(x: Vector) -> Scalar:
    return jnp.sum(
        100.0 * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(1.0 - x[:-1])
    )


@jax.jit
def rosen_grad(x: Vector) -> Vector:
    return jax.grad(rosen)(x)


@jax.jit
def rosen_hess_diag(x: Vector) -> Vector:
    return jnp.diagonal(jax.hessian(rosen)(x))


@jax.jit
def rosen_hess_prod(x: Vector, p: Vector) -> Vector:
    return math.hess_prod(rosen, x, p)


@jax.jit
def rosen_hess_quad(x: Vector, p: Vector) -> Scalar:
    return jnp.vdot(p, rosen_hess_prod(x, p))


@jax.jit
def rosen_value_and_grad(x: Vector) -> tuple[Scalar, Vector]:
    return jax.value_and_grad(rosen)(x)


@jax.jit
def rosen_grad_and_hess_diag(x: Vector) -> tuple[Vector, Vector]:
    return rosen_grad(x), rosen_hess_diag(x)


def rosen_objective() -> Objective:
    return Objective(
        fun=rosen,
        grad=rosen_grad,
        hess_diag=rosen_hess_diag,
        hess_prod=rosen_hess_prod,
        hess_quad=rosen_hess_quad,
        value_and_grad=rosen_value_and_grad,
        grad_and_hess_diag=rosen_grad_and_hess_diag,
    )
