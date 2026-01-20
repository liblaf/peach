from collections.abc import Iterable

from ._abc import Constraint


def filter_constraints[T: Constraint](
    constraints: Iterable[Constraint], cls: type[T]
) -> list[T]:
    return [c for c in constraints if isinstance(c, cls)]


def partition_constraints[A: Constraint, B: Constraint](
    constraints: Iterable[B], cls: type[A]
) -> tuple[list[A], list[B]]:
    matched: list[A] = []
    unmatched: list[B] = []
    for c in constraints:
        if isinstance(c, cls):
            matched.append(c)
        else:
            unmatched.append(c)  # pyright: ignore[reportArgumentType]
    return matched, unmatched


def pop_constraint[A: Constraint, B: Constraint](
    constraints: Iterable[B], cls: type[A]
) -> tuple[A | None, list[B]]:
    found: A | None = None
    remaining: list[B] = []
    for c in constraints:
        if isinstance(c, cls):
            if found is None:
                found = c
            else:
                msg: str = f"Multiple {cls.__name__} constraints found."
                raise NotImplementedError(msg)
        else:
            remaining.append(c)
    return found, remaining
