import functools
from collections.abc import Callable

import jax

from liblaf import grapes

from ._dynamic import jit_dynamic


@grapes.wraps(jax.jit)
def jit(fun: Callable | None = None, **kwargs) -> Callable:
    _warnings_hide = True
    if fun is None:
        return functools.partial(jit, **kwargs)
    return jit_dynamic(fun, **kwargs)
