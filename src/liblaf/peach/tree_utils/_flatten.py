from collections.abc import Callable

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
from jaxtyping import Array


def flatten[T](obj: T) -> tuple[Array, Callable[[Array], T]]:
    data: T
    static: T
    data, static = eqx.partition(obj, eqx.is_array)
    flat: Array
    unravel: Callable[[Array], T]
    flat, unravel = jfu.ravel_pytree(data)

    def unflatten(a: Array) -> T:
        a = jnp.asarray(a, dtype=flat.dtype)
        data: T = unravel(a)
        return eqx.combine(data, static)

    return flat, unflatten
