from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
from jaxtyping import Array, Float

type Scalar = Float[Array, ""]


def hess_prod[T](
    func: Callable[..., Scalar],
    x: T,
    p: T,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = {},
) -> T:
    def wrapper(x: T) -> Scalar:
        return func(x, *args, **kwargs)

    output: T
    _, output = jax.jvp(jax.grad(wrapper), (x,), (p,))
    return output
