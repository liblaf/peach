from collections.abc import Iterator, Mapping
from typing import Any

import attrs
import torch
from jaxtyping import Float
from scipy.optimize import OptimizeResult
from torch import Tensor

from liblaf.peach.optim.base import State, Stats

type Vector = Float[Tensor, " N"]


@attrs.define
class ScipyState(Mapping[str, Any], State):
    """Optimizer state backed by SciPy's `OptimizeResult`."""

    __wrapped__: OptimizeResult = attrs.field(factory=OptimizeResult)

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
    def params(self) -> Vector:
        """Final parameter vector from the SciPy result."""
        return torch.as_tensor(self.__wrapped__["x"])


@attrs.define
class ScipyStats(Stats):
    """Stats placeholder for the SciPy optimizer adapter."""
