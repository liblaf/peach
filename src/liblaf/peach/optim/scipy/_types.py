from collections.abc import Iterator, Mapping
from typing import Any

import attrs
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf.peach.optim.base import State, Stats

type Vector = Float[Array, " N"]


def _field_transformer(
    _cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    return [field for field in fields if field.name != "params"]


@jarp.define(field_transformer=_field_transformer)
class ScipyState(State, Mapping[str, Any]):
    __wrapped__: OptimizeResult = jarp.field(factory=OptimizeResult, alias="wrapped")

    def __init__(self, wrapped: OptimizeResult | None = None) -> None:
        if wrapped is None:
            wrapped = OptimizeResult()
        self.__attrs_init__(wrapped)  # pyright: ignore[reportAttributeAccessIssue]

    def __getitem__(self, key: str) -> Any:
        return self.__wrapped__[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.__wrapped__

    def __len__(self) -> int:
        return len(self.__wrapped__)

    @property
    def params(self) -> Vector:  # pyright: ignore[reportIncompatibleVariableOverride]
        return jnp.asarray(self.__wrapped__["x"], float)


@jarp.define
class ScipyStats(Stats):
    pass
