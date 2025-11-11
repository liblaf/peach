from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

from jaxtyping import Array, PyTree, Shaped
from scipy.optimize import Bounds

from liblaf.peach.tree_utils import Unflatten

class Objective:
    def __init__(
        self,
        fun: Callable | None = None,
        grad: Callable | None = None,
        hess: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_prod: Callable | None = None,
        hess_quad: Callable | None = None,
        value_and_grad: Callable | None = None,
        grad_and_hess_diag: Callable | None = None,
        bounds: Bounds | None = None,
    ) -> None: ...
    @property
    def fun(self) -> Callable | None: ...
    @property
    def grad(self) -> Callable | None: ...
    @property
    def hess(self) -> Callable | None: ...
    @property
    def hess_diag(self) -> Callable | None: ...
    @property
    def hess_prod(self) -> Callable | None: ...
    @property
    def hess_quad(self) -> Callable | None: ...
    @property
    def value_and_grad(self) -> Callable | None: ...
    @property
    def grad_and_hess_diag(self) -> Callable | None: ...
    @property
    def bounds(
        self,
    ) -> tuple[Shaped[Array, " free"] | None, Shaped[Array, " free"] | None]: ...

    # ------------------------------- wrappers ------------------------------- #

    def flatten[T](
        self,
        params: T,
        *,
        fixed_mask: T | None = None,
        n_fixed: int | None = None,
        lower_bound: T | None = None,
        upper_bound: T | None = None,
    ) -> tuple[Self, Shaped[Array, " free"]]: ...
    unflatten: Unflatten[PyTree] | None
    _flatten: bool
    _lower_bound_flat: Array | None
    _upper_bound_flat: Array | None

    def jit(self, enable: bool = True) -> Objective: ...
    _jit: bool

    def partial(self, *args: Any, **kwargs: Any) -> Objective: ...
    _args: Sequence[Any]
    _kwargs: Mapping[str, Any]

    def timer(self, enable: bool = True) -> Objective: ...
    _timer: bool

    def with_aux(self, enable: bool = True) -> Objective: ...
    _with_aux: bool
