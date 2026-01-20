from collections.abc import Callable, Mapping, Sequence
from typing import Any

import wrapt

from liblaf.peach import tree, utils
from liblaf.peach.transforms import LinearTransform


@tree.define
class FunctionWrapper:
    args: Sequence[Any] = tree.field(default=(), kw_only=True)
    kwargs: Mapping[str, Any] = tree.field(factory=dict, kw_only=True)
    transform: LinearTransform | None = tree.field(default=None, kw_only=True)

    input_params: tuple[int, ...] = tree.static(default=(), kw_only=True)
    input_grad: tuple[int, ...] = tree.static(default=(), kw_only=True)
    input_hess_diag: tuple[int, ...] = tree.static(default=(), kw_only=True)
    output_params: tuple[int, ...] = tree.static(default=(), kw_only=True)
    output_grad: tuple[int, ...] = tree.static(default=(), kw_only=True)
    output_hess_diag: tuple[int, ...] = tree.static(default=(), kw_only=True)

    @wrapt.decorator
    def __call__(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self.transform is not None:
            args: list[Any] = list(args)
            params: Any = args[0]
            for idx in self.input_grad:
                args[idx] = self.transform.forward_tangents(params, args[idx])
            for idx in self.input_hess_diag:
                args[idx] = self.transform.forward_hess_diag(params, args[idx])
            for idx in self.input_params:
                args[idx] = self.transform.forward_primals(args[idx])
        input_args: tuple[Any, ...] = (*self.args, *args)
        input_kwargs: dict[str, Any] = {**self.kwargs, **kwargs}
        outputs: list[Any] = utils.pack(wrapped(*input_args, **input_kwargs))
        if self.transform is not None:
            params_out: Any = args[0]
            for idx in self.output_grad:
                outputs[idx] = self.transform.backward_grad(params_out, outputs[idx])
            for idx in self.output_hess_diag:
                outputs[idx] = self.transform.backward_hess_diag(
                    params_out, outputs[idx]
                )
            for idx in self.output_params:
                outputs[idx] = self.transform.backward_params(outputs[idx])
        return utils.unpack(outputs)
