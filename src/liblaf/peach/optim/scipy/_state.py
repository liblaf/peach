from collections.abc import Iterator, Mapping
from typing import Any

from scipy.optimize import OptimizeResult

from liblaf.peach import tree
from liblaf.peach.optim.abc import Params, State
from liblaf.peach.tree import FlatDef


@tree.define
class ScipyState(Mapping[str, Any], State):
    flat_def: FlatDef[Params]
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
        return self.flat_def.unflatten(self.result["x"])

    @params.setter
    def params(self, value: Params, /) -> None:
        self.result["x"] = self.flat_def.flatten(value)
