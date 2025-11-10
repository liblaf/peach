from collections.abc import Callable, Iterator, Mapping
from typing import Any

from jaxtyping import Array
from scipy.optimize import OptimizeResult

from liblaf.peach import tree_utils
from liblaf.peach.optim.abc import Params, State


@tree_utils.define
class ScipyState(Mapping[str, Any], State):
    result: OptimizeResult = tree_utils.container(factory=OptimizeResult)
    unflatten: Callable[[Array], Params] = lambda x: x

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
        return self.unflatten(self.result["x"])

    @params.setter
    def params(self, value: Params, /) -> None:
        flat: Array
        flat, _ = tree_utils.flatten(value)
        self.result["x"] = flat
