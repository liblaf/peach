from typing import overload

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike


@overload
def asarray(value: None) -> None: ...
@overload
def asarray(value: ArrayLike) -> Array: ...
def asarray(value: ArrayLike | None) -> Array | None:
    if value is None:
        return None
    return jnp.asarray(value)
