from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

import jarp
import jax.tree_util as jtu
from jaxtyping import Array, Float

from liblaf.peach.transforms import IdentityTransform, Transform

type Vector = Float[Array, " free"]


@jarp.define
class LinearSystem[Params]:
    _matvec: Callable = jarp.field(alias="matvec")
    b: Params = jarp.field()
    _rmatvec: Callable | None = jarp.field(default=None, alias="rmatvec")
    _rpreconditioner: Callable | None = jarp.field(
        default=None, alias="rpreconditioner"
    )
    _preconditioner: Callable | None = jarp.field(default=None, alias="preconditioner")

    args: Sequence[Any] = jarp.field(default=(), kw_only=True)
    kwargs: Mapping[str, Any] = jarp.field(factory=dict, kw_only=True)
    transform: Transform[Vector, Params] = jarp.field(
        factory=IdentityTransform, kw_only=True
    )

    @property
    def b_flat(self) -> Vector:
        return self.transform.backward_tangents(self.b)

    @property
    def matvec(self) -> Callable[[Vector], Vector]:
        return self._wraps(self._matvec)

    @property
    def rmatvec(self) -> Callable[[Vector], Vector] | None:
        return self._wraps(self._rmatvec)

    @property
    def preconditioner(self) -> Callable[[Vector], Vector] | None:
        return self._wraps(self._preconditioner)

    @property
    def rpreconditioner(self) -> Callable[[Vector], Vector] | None:
        return self._wraps(self._rpreconditioner)

    @overload
    def _wraps(self, func: None) -> None: ...
    @overload
    def _wraps[T](self, func: Callable[..., T]) -> Callable[[Vector], Vector]: ...
    def _wraps[T](
        self, func: Callable[..., T] | None
    ) -> Callable[[Vector], Vector] | None:
        if func is None:
            return None
        if self.args or self.kwargs:
            func = jtu.Partial(func, *self.args, **self.kwargs)
        return _MatvecWrapper(matvec=func, transform=self.transform)  # pyright: ignore[reportArgumentType]


@jarp.define
class _MatvecWrapper[Params]:
    matvec: Callable[[Params], Params]
    transform: Transform[Vector, Params]

    def __call__(self, x_flat: Vector) -> Vector:
        x_tree: Params = self.transform.forward_tangents(x_flat)
        y_tree: Params = self.matvec(x_tree)
        y_flat: Vector = self.transform.backward_tangents(y_tree)
        return y_flat
