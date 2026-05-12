from __future__ import annotations

import enum

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from liblaf import jarp

from ._protocols import State, Stats

type Vector = Float[Array, " N"]


class Result(jarp.Enum):
    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    MAX_STEPS_REACHED = enum.auto()
    NAN = enum.auto()
    STAGNATION = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    @property
    def success(self) -> Bool[Array, ""]:
        return jnp.any(
            jnp.asarray(
                [
                    self == Result.SUCCESS,
                    self == Result.PRIMARY_SUCCESS,
                    self == Result.SECONDARY_SUCCESS,
                ]
            )
        )


@jarp.define
class Solution[S: State, T: Stats]:
    result: Result = jarp.field()
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        return self.state.params

    @property
    def success(self) -> Bool[Array, ""]:
        return self.result.success
