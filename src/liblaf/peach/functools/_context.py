from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import attrs

from liblaf.peach import tree
from liblaf.peach.transforms import LinearTransform, chain_transforms

from ._wrapper import FunctionWrapper


@tree.define
class FunctionContext:
    args: Sequence[Any] = tree.field(default=(), kw_only=True)
    kwargs: Mapping[str, Any] = tree.field(factory=dict, kw_only=True)
    _transform: LinearTransform | None = tree.field(
        default=None, kw_only=True, alias="transform"
    )

    def partial(self, /, *args: Any, **kwargs: Any) -> Self:
        new_args: Sequence[Any] = (*self.args, *args)
        new_kwargs: Mapping[str, Any] = {**self.kwargs, **kwargs}
        return attrs.evolve(self, args=new_args, kwargs=new_kwargs)

    def transform(self, transform: LinearTransform | None) -> Self:
        return attrs.evolve(
            self, transform=chain_transforms(transform, self._transform)
        )

    def _wraps(
        self,
        func: Callable | None,
        *,
        input_params: tuple[int, ...] = (),
        input_grad: tuple[int, ...] = (),
        input_hess_diag: tuple[int, ...] = (),
        output_params: tuple[int, ...] = (),
        output_grad: tuple[int, ...] = (),
        output_hess_diag: tuple[int, ...] = (),
    ) -> Callable | None:
        if func is None:
            return None
        wrapper = FunctionWrapper(
            args=self.args,
            kwargs=self.kwargs,
            transform=self._transform,
            input_params=input_params,
            input_grad=input_grad,
            input_hess_diag=input_hess_diag,
            output_params=output_params,
            output_grad=output_grad,
            output_hess_diag=output_hess_diag,
        )
        return wrapper(func)
