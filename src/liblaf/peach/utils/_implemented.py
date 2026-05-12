from collections.abc import Callable
from typing import Any


def not_implemented[F](func: F) -> F:
    func.__not_implemented__ = True  # ty:ignore[unresolved-attribute]
    return func


def implemented(obj: Any, method: str | Callable) -> bool:
    if not isinstance(method, str):
        method: str = method.__name__  # ty:ignore[unresolved-attribute]
    try:
        method: Any = getattr(obj, method)
    except AttributeError:
        return False
    return not (method is None or getattr(method, "__not_implemented__", False))
