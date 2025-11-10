import operator

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def tree_dot(a: PyTree, b: PyTree) -> Float[Array, ""]:
    return jax.tree.reduce(operator.add, jax.tree.map(jnp.vdot, a, b))
