from __future__ import annotations

import enum
from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from liblaf import jarp

type Vector = Float[Array, " N"]


class Result(jarp.Enum):
    """Result code returned by linear solvers."""

    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    BREAKDOWN = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    @property
    def success(self) -> Bool[Array, ""]:
        """Whether the result represents an accepted solution."""
        return jnp.any(
            jnp.asarray(
                [
                    self == Result.SUCCESS,
                    self == Result.PRIMARY_SUCCESS,
                    self == Result.SECONDARY_SUCCESS,
                ]
            )
        )


class State(Protocol):
    """Protocol for solver states that expose solution parameters."""

    @property
    def params(self) -> Vector:
        """Current solution estimate."""
        ...


class Stats(Protocol):
    """Protocol for solver-specific summary statistics."""


@jarp.define
class Solution[S: State, T: Stats]:
    """Linear-solver output bundle."""

    result: Result
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        """Final solution parameters."""
        return self.state.params

    @property
    def success(self) -> Bool[Array, ""]:
        """Whether [`result`][liblaf.peach.linalg.base.Solution.result] is successful."""
        return self.result.success
