from typing import Any, Protocol

from jaxtyping import PyTree

type Aux = Any
type X = PyTree


class Callback[State](Protocol):
    def __call__(self, state: State, n_steps: int) -> None: ...
