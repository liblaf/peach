import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from liblaf.peach import tree


@tree.define
class Node:
    x: Array = tree.array(factory=lambda: jnp.zeros((1,)))
    static: str = "foo"


@tree.define
class ObjectWithTreeView:
    structure: tree.Structure[Node] | None = None

    a = tree.TreeView[Node]()
    a_flat: Array = tree.array(default=None)


def test_tree_view() -> None:
    a = Node()
    tree = ObjectWithTreeView()
    tree.a = a
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
    a.x = jnp.ones((1,))
    tree.a_flat = a.x
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)


@tree.define
class ObjectWithFlatView:
    structure: tree.Structure[Node] | None = None

    a: Node | None = None
    a_flat = tree.FlatView[Node]()


def test_flat_view() -> None:
    a = Node()
    tree = ObjectWithFlatView()
    tree.a = a
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
    a.x = jnp.ones((1,))
    tree.a_flat = a.x
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
