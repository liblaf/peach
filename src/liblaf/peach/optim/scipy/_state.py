from collections.abc import Iterator, Mapping
from typing import Any

from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach import tree
from liblaf.peach.optim.abc import Params, State

type Vector = Float[Array, " N"]


@tree.define
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
    def params(self) -> Params:
        return self.structure.unflatten(self.result["x"])

    @params.setter
    def params(self, value: Params, /) -> None:
        self.result["x"] = self.structure.flatten(value)  # pyright: ignore[reportIndexIssue]

    @property
    def params_flat(self) -> Vector:
        return self.result["x"]

    @params_flat.setter
    def params_flat(self, value: Vector, /) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        self.result["x"] = value  # pyright: ignore[reportIndexIssue]
