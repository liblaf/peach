from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import tree as pt
from liblaf.peach.tree import FlatDef


def test_flatten() -> None:
    tree: dict[str, Any] = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
    flat: Float[Array, " N"]
    unflatten: FlatDef[dict[str, Any]]
    flat, unflatten = pt.flatten(tree)
    actual: dict[str, Any] = unflatten.unflatten(flat)
    np.testing.assert_allclose(actual["a"], jnp.zeros((3,)))
    np.testing.assert_allclose(actual["b"], jnp.ones((4,)))
    assert flat.shape == (7,)
    assert actual["static"] == tree["static"]


def test_flatten_fixed() -> None:
    tree: dict[str, Any] = {"a": jnp.zeros((5,)), "b": jnp.ones((4,)), "static": "foo"}
    fixed_mask: dict[str, Any] = {
        "a": jnp.asarray([False, True, False, True, False], jnp.bool),
        "b": jnp.asarray([False, False, True, True], jnp.bool),
        "static": "foo",
    }
    flat: Float[Array, " N"]
    flat_def: FlatDef[dict[str, Any]]
    flat, flat_def = pt.flatten(tree, fixed_mask=fixed_mask)
    assert flat.shape == (5,)
    actual: dict[str, Any] = flat_def.unflatten(flat)
    np.testing.assert_allclose(actual["a"], tree["a"])
    np.testing.assert_allclose(actual["b"], tree["b"])
    assert actual["static"] == tree["static"]
