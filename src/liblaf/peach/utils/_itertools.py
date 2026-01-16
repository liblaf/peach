from collections.abc import Iterable


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
