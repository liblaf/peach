from collections.abc import Iterable
from typing import Any, overload


def partition_type[T1, T2](
    cls: type[T1], iterable: Iterable[T1 | T2]
) -> tuple[list[T1], list[T2]]:
    positive: list[T1] = []
    negative: list[T2] = []
    for i in iterable:
        if isinstance(i, cls):
            positive.append(i)
        else:
            negative.append(i)  # pyright: ignore[reportArgumentType]
    return positive, negative


@overload
def pack[T](x: tuple[T, ...]) -> list[T]: ...
@overload
def pack[T](x: T) -> list[T]: ...
def pack(x: Any) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def unpack[T](x: list[T]) -> T | list[T]:
    return x[0] if len(x) == 1 else x
