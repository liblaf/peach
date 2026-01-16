from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest_codspeed import BenchmarkFixture

from liblaf.peach import tree

type PyTreeDef = Any


@tree.define
class A:  # noqa: PLW1641
    data: Array = tree.array(factory=lambda: jnp.zeros(()))
    static: str = "static"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, A):
            return NotImplemented
        return jnp.array_equal(self.data, other.data) and self.static == other.static  # pyright: ignore[reportReturnType]


@pytest.mark.benchmark(group="flatten")
def test_flatten(benchmark: BenchmarkFixture) -> None:
    a = A()
    flat: Array
    structure: tree.Structure[A]
    flat, structure = benchmark(tree.flatten, a)
    assert structure.unflatten(flat) == a


@pytest.mark.benchmark(group="flatten")
def test_flatten_baseline(benchmark: BenchmarkFixture) -> None:
    a = A()
    leaves: list[Any]
    treedef: PyTreeDef
    leaves, treedef = benchmark(jax.tree.flatten, a)
    assert jax.tree.unflatten(treedef, leaves) == a


@pytest.mark.benchmark(group="unflatten")
def test_unflatten(benchmark: BenchmarkFixture) -> None:
    a = A()
    flat: Array
    structure: tree.Structure[A]
    flat, structure = tree.flatten(a)
    assert benchmark(structure.unflatten, flat) == a


@pytest.mark.benchmark(group="unflatten")
def test_unflatten_baseline(benchmark: BenchmarkFixture) -> None:
    a = A()
    leaves: list[Any]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(a)
    assert benchmark(jax.tree.unflatten, treedef, leaves) == a
