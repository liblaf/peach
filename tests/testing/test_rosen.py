import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach.testing import RosenObjective

type Vector = Float[Array, " N"]


def test_rosen_objective_minimum_has_zero_value_and_gradient() -> None:
    objective = RosenObjective()
    x: Vector = jnp.asarray([1.0, 1.0, 1.0])

    value, grad = objective.value_and_grad(x)

    np.testing.assert_allclose(np.asarray(value), 0.0)
    np.testing.assert_allclose(np.asarray(grad), np.zeros(3))
    np.testing.assert_allclose(
        np.asarray(objective.hess_diag(x)),
        np.asarray([802.0, 1002.0, 200.0]),
    )


def test_rosen_hess_quad_matches_explicit_hessian_product() -> None:
    objective = RosenObjective()
    x: Vector = jnp.asarray([-1.0, 1.5, 0.5])
    p: Vector = jnp.asarray([0.25, -0.5, 1.0])

    hessian_p = objective.hess_prod(x, p)

    np.testing.assert_allclose(
        np.asarray(objective.hess_quad(x, p)),
        np.asarray(jnp.vdot(p, hessian_p)),
    )
