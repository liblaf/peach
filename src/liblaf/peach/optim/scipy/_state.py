from collections.abc import Iterator, Mapping
from typing import Any

import attrs
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach import tree
from liblaf.peach.optim.abc import State

type Vector = Float[Array, " N"]


def _field_transformer(
    _cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    # filter out `params`
    fields = [field for field in fields if field.name != "params"]
    return fields


@tree.define(field_transformer=_field_transformer)
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

    @params.setter
    def params(self, value: Vector, /) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        self.result["x"] = value  # pyright: ignore[reportIndexIssue]
