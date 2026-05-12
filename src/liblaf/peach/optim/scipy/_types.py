from collections.abc import Iterator, Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult

from liblaf import jarp
from liblaf.peach.optim.base import State, Stats

type Vector = Float[Array, " N"]


@jarp.define
class ScipyState(State, Mapping[str, Any]):
    """Optimizer state backed by SciPy's `OptimizeResult`."""

    __wrapped__: OptimizeResult = jarp.field(factory=OptimizeResult)

    def __init__(self, wrapped: OptimizeResult | None = None) -> None:
        """Create state from an optional SciPy result object."""
        if wrapped is None:
            wrapped = OptimizeResult()
        self.__attrs_init__(wrapped)  # ty:ignore[unresolved-attribute]

    def __getitem__(self, key: str) -> Any:
        """Return an item from the wrapped SciPy result."""
        return self.__wrapped__[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys in the wrapped SciPy result."""
        yield from self.__wrapped__

    def __len__(self) -> int:
        """Return the number of keys in the wrapped SciPy result."""
        return len(self.__wrapped__)

    @property
    def params(self) -> Vector:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Final parameter vector from the SciPy result."""
        return jnp.asarray(self.__wrapped__["x"], float)


@jarp.define
class ScipyStats(Stats):
    """Stats placeholder for the SciPy optimizer adapter."""
