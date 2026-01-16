import functools
from collections.abc import Callable, Iterable
from typing import Any


def update_wrapper[T: Callable](
    wrapper: T,
    wrapped: Any,
    assigned: Iterable[str] = functools.WRAPPER_ASSIGNMENTS,
    updated: Iterable[str] = functools.WRAPPER_UPDATES,
) -> T:
    for attr in assigned:
        try:
            value: Any = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # we intentionally do not set __wrapped__ since wrapt already does that
    return wrapper
