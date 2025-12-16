from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


def absolute_residual(
    matvec: Callable[[Vector], Vector], x: Vector, b: Vector
) -> Scalar:
    r: Vector = matvec(x) - b
    r_norm: Scalar = jnp.linalg.norm(r)
    return r_norm


def relative_residual(
    matvec: Callable[[Vector], Vector], x: Vector, b: Vector
) -> Scalar:
    r_norm: Scalar = absolute_residual(matvec, x, b)
    b_norm: Scalar = jnp.linalg.norm(b)
    return r_norm / b_norm


def satisfies_tolerance(
    matvec: Callable[[Vector], Vector],
    x: Vector,
    b: Vector,
    *,
    atol: Float[ArrayLike, ""] = 1e-15,
    rtol: Float[ArrayLike, ""] = 1e-5,
) -> Bool[Array, ""]:
    residual: Scalar = absolute_residual(matvec, x, b)
    b_norm: Scalar = jnp.linalg.norm(b)
    return residual <= atol + rtol * b_norm
