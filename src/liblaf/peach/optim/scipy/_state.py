from collections.abc import Iterator, Mapping
from typing import Any

from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach import tree
from liblaf.peach.optim.abc import State

type Vector = Float[Array, " N"]


@tree.define()
class ScipyState(Mapping[str, Any], State):
    result: OptimizeResult = tree.container(factory=OptimizeResult)

    def __getitem__(self, key: str, /) -> Any:
        return self.result[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.result)

    def __len__(self) -> int:
        return len(self.result)

    @property
    def fun(self) -> float:
        return self.result["fun"]

    @property
    def params(self) -> Vector:
        return self.result["x"]
