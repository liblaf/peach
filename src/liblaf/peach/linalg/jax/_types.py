from typing import Protocol

from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.peach.linalg.base import LinearSystem, State, Stats, SupportsMatvec

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class JaxLinearSystem(LinearSystem, SupportsMatvec, Protocol): ...


@jarp.define
class JaxState(State): ...


@jarp.define
class JaxStats(Stats):
    info: int | None = None
    residual_relative: Scalar = jarp.array(default=None)
