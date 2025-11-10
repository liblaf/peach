import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.peach import math, optim


@jax.jit
def rosen(x: Float[Array, " N"]) -> Float[Array, ""]:
    return jnp.sum(
        100.0 * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(1.0 - x[:-1])
    )


@jax.jit
def rosen_grad(x: Float[Array, " N"]) -> Float[Array, " N"]:
    return jax.grad(rosen)(x)


@jax.jit
def rosen_hess_diag(x: Float[Array, " N"]) -> Float[Array, " N"]:
    return jnp.diagonal(jax.hessian(rosen)(x))


@jax.jit
def rosen_hess_prod(x: Float[Array, " N"], p: Float[Array, " N"]) -> Float[Array, " N"]:
    return math.hess_prod(rosen, x, p)


@jax.jit
def rosen_hess_quad(x: Float[Array, " N"], p: Float[Array, " N"]) -> Float[Array, " N"]:
    return jnp.vdot(p, rosen_hess_prod(x, p))


@jax.jit
def rosen_value_and_grad(
    x: Float[Array, " N"],
) -> tuple[Float[Array, ""], Float[Array, " N"]]:
    return jax.value_and_grad(rosen)(x)


@jax.jit
def rosen_grad_and_hess_diag(
    x: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    return rosen_grad(x), rosen_hess_diag(x)


def rosen_objective() -> optim.Objective:
    return optim.Objective(
        fun=rosen,
        grad=rosen_grad,
        hess_diag=rosen_hess_diag,
        hess_prod=rosen_hess_prod,
        hess_quad=rosen_hess_quad,
        value_and_grad=rosen_value_and_grad,
        grad_and_hess_diag=rosen_grad_and_hess_diag,
    )
