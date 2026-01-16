import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest_codspeed import BenchmarkFixture

from liblaf.peach import tree


@tree.define
class A:
    data: Array = tree.array(factory=lambda: jnp.zeros(()))
    static: str = tree.static(default="static")

    @tree.method
    def method(self) -> Array:
        return self.data


def func(a: A) -> Array:
    return a.method()


@pytest.mark.benchmark(group="jit_function")
def test_jit_function(benchmark: BenchmarkFixture) -> None:
    a = A()
    func_jit = tree.jit(func)
    assert jnp.array_equal(benchmark(func_jit, a), a.data)


@pytest.mark.benchmark(group="jit_function")
def test_jit_function_baseline(benchmark: BenchmarkFixture) -> None:
    a = A()
    func_jit = jax.jit(func)
    assert jnp.array_equal(benchmark(func_jit, a), a.data)


@pytest.mark.benchmark(group="jit_method")
def test_jit_method(benchmark: BenchmarkFixture) -> None:
    a = A()
    a_method_jit = tree.jit(a.method)
    assert jnp.array_equal(benchmark(a_method_jit), a.data)


@pytest.mark.benchmark(group="jit_method")
def test_jit_method_baseline(benchmark: BenchmarkFixture) -> None:
    a = A()
    a_method_jit = jax.jit(a.method)
    assert jnp.array_equal(benchmark(a_method_jit), a.data)
