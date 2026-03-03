from typing import Protocol

import jarp
from jaxtyping import Array, Float

from liblaf.peach.linalg.base import LinearSystem, State, Stats, SupportsMatvec

type Scalar = Float[Array, ""]


class CupyLinearSystem(LinearSystem, SupportsMatvec, Protocol): ...


@jarp.define
class CupyState(State): ...


@jarp.define
class CupyStats(Stats):
    info: int = -1
    n_steps: int | None = jarp.field(default=None, kw_only=True)
    relative_residual: Scalar = jarp.array(default=None, kw_only=True)
