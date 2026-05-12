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
    """Evaluate a Hessian-vector product with JAX autodiff.

    `func` must return a scalar. `x` and `p` may be arrays or matching PyTree
    structures accepted by `jax.grad` and `jax.jvp`.

    Args:
        func: Scalar-valued function whose Hessian is sampled.
        x: Point where the Hessian is evaluated.
        p: Direction multiplied by the Hessian.
        args: Extra positional arguments passed to `func`.
        kwargs: Extra keyword arguments passed to `func`.

    Returns:
        The Hessian-vector product `H(x) @ p`, with the same structure as `x`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from liblaf.peach.math import hess_prod
        >>> matrix = jnp.asarray([[3.0, 1.0], [1.0, 2.0]])
        >>> def quadratic(x):
        ...     return 0.5 * jnp.vdot(x, matrix @ x)
        >>> product = hess_prod(
        ...     quadratic,
        ...     jnp.asarray([1.0, -1.0]),
        ...     jnp.asarray([2.0, 0.5]),
        ... )
        >>> np.asarray(product).tolist()
        [6.5, 3.0]
    """

    def wrapper(x: T) -> Scalar:
        return func(x, *args, **kwargs)

    _, output = jax.jvp(jax.grad(wrapper), (x,), (p,))
    return output
