import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import math

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


def test_hess_prod_matches_quadratic_hessian_product() -> None:
    matrix: Float[Array, "2 2"] = jnp.asarray([[3.0, 1.0], [1.0, 2.0]])

    def quadratic(x: Vector) -> Scalar:
        return 0.5 * jnp.vdot(x, matrix @ x)

    x: Vector = jnp.asarray([1.5, -2.0])
    p: Vector = jnp.asarray([4.0, -1.0])

    np.testing.assert_allclose(
        np.asarray(math.hess_prod(quadratic, x, p)),
        np.asarray(matrix @ p),
    )
