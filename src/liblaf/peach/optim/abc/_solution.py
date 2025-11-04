import enum
from typing import Any

from liblaf.peach import tree_utils

from ._typing import Aux, X


class Result(enum.StrEnum):
    SUCCESSFUL = enum.auto()
    FAILURE = enum.auto()
    MAX_STEPS_REACHED = enum.auto()


@tree_utils.tree
class OptimizeSolution[State]:
    aux: Aux
    result: Result
    state: State
    stats: dict[str, Any]
    x: X

    @property
    def success(self) -> bool:
        return self.result == Result.SUCCESSFUL
