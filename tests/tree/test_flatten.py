from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import tree


def test_flatten() -> None:
    obj: dict[str, Any] = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
    flat: Float[Array, " N"]
    structure: tree.Structure[dict[str, Any]]
    flat, structure = tree.flatten(obj)
    actual: dict[str, Any] = structure.unflatten(flat)
    np.testing.assert_allclose(actual["a"], jnp.zeros((3,)))
    np.testing.assert_allclose(actual["b"], jnp.ones((4,)))
    assert flat.shape == (7,)
    assert actual["static"] == obj["static"]
