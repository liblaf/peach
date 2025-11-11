import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from liblaf.peach import tree


@tree.define
class Node:
    x: Array = tree.array()
    static: str = "foo"


@tree.define
class Tree:
    a: Node
    a_flat = tree.FlatView[Node]()
    b = tree.TreeView[Node]()
    b_flat: Array = tree.array(default=None)
    unflatten: tree.Unflatten[Node] | None = None


def test_tree_view() -> None:
    a = Node(x=jnp.zeros((3,)), static="a")
    tree = Tree(a=a)
    # access flat view to initialize unflatten
    tree.a_flat  # noqa: B018
    tree.b_flat = jnp.ones((3,))
    np.testing.assert_allclose(tree.a_flat, a.x)
    np.testing.assert_allclose(tree.b.x, tree.b_flat)
    assert tree.b.static == a.static
