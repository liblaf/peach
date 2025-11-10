from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
from jaxtyping import PyTree


def hess_prod(
    func: Callable,
    x: PyTree,
    p: PyTree,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = {},
) -> PyTree:
    def wrapper(x: PyTree) -> PyTree:
        return func(x, *args, **kwargs)

    output: PyTree
    _, output = eqx.filter_jvp(eqx.filter_grad(wrapper), (x,), (p,))
    return output
