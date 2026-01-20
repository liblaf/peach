from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import attrs
import wrapt

from liblaf.peach import tree
from liblaf.peach.transforms import IdentityTransform, LinearTransform, chain_transforms


@tree.define
class FunctionContext:
    args: Sequence[Any] = tree.field(default=(), kw_only=True)
    kwargs: Mapping[str, Any] = tree.field(factory=dict, kw_only=True)
    transform: LinearTransform = tree.field(factory=IdentityTransform, kw_only=True)

    def apply_transform(self, transform: LinearTransform | None) -> Self:
        return attrs.evolve(self, transform=chain_transforms(transform, self.transform))

    def partial(self, /, *args: Any, **kwargs: Any) -> Self:
        new_args: Sequence[Any] = (*self.args, *args)
        new_kwargs: Mapping[str, Any] = {**self.kwargs, **kwargs}
        return attrs.evolve(self, args=new_args, kwargs=new_kwargs)

    def _wraps(self, func: Callable | None, *, method: str) -> Callable | None:
        if func is None:
            return None
        if self.args or self.kwargs:
            func: Callable = wrapt.partial(func, *self.args, **self.kwargs)
        func = self.transform.wraps(method, func)
        return func
