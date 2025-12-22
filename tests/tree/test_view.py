import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from liblaf.peach import tree


@tree.define
class Node:
    x: Array = tree.array()
    static: str = "foo"


@tree.define
class TreeWithTreeView:
    structure: tree.Structure[Node] | None = None

    a = tree.TreeView[Node]()
    a_flat: Array = tree.array(default=None)


def test_tree_view() -> None:
    a = Node(x=jnp.zeros((3,)))
    tree = TreeWithTreeView()
    tree.a = a
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
    a.x = jnp.ones((3,))
    tree.a_flat = a.x
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)


@tree.define
class TreeWithFlatView:
    structure: tree.Structure[Node] | None = None

    a: Node | None = None
    a_flat = tree.FlatView[Node]()


def test_flat_view() -> None:
    a = Node(x=jnp.zeros((3,)))
    tree = TreeWithFlatView()
    tree.a = a
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
    a.x = jnp.ones((3,))
    tree.a_flat = a.x
    assert tree.a.static == a.static
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.a.x, a.x)
