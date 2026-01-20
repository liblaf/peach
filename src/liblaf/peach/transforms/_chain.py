import collections
from collections.abc import Callable
from typing import Any, Self, override

import jax.tree_util as jtu
import tlz

from ._abc import LinearTransform

type AuxData = None
type Children = list[LinearTransform]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, LinearTransform]
type KeyLeafPairs = list[KeyLeafPair]


@jtu.register_pytree_with_keys_class
class TransformChain[I, O](
    collections.UserList[LinearTransform], LinearTransform[I, O]
):
    @override
    def wraps(self, method: str, wrapped: Callable[..., Any]) -> Callable[..., Any]:
        for transform in reversed(self.data):
            wrapped = transform.wraps(method, wrapped)
        return wrapped

    @override
    def forward_primals(self, primals: I) -> O:
        for transform in self.data:
            primals = transform.forward_primals(primals)
        return primals  # pyright: ignore[reportReturnType]

    @override
    def linearize(self, primals: I) -> tuple[O, Callable[[I], O]]:
        lin_fun_chain: list[Callable[[Any], Any]] = []
        for transform in self.data:
            lin_fun: Callable
            primals, lin_fun = transform.linearize(primals)
            lin_fun_chain.append(lin_fun)
        return primals, tlz.compose_left(*lin_fun_chain)  # pyright: ignore[reportReturnType]

    @override
    def linear_transpose(self, primals: I) -> Callable[[O], I]:
        f_transpose_chain: list[Callable[[Any], Any]] = []
        for transform in self.data:
            f_transpose: Callable
            f_transpose = transform.linear_transpose(primals)
            f_transpose_chain.append(f_transpose)
            primals = transform.forward_primals(primals)
        return tlz.compose(*f_transpose_chain)

    @override
    def forward_hess_diag(self, hess_diag: I) -> O:
        for transform in self.data:
            hess_diag = transform.forward_hess_diag(hess_diag)
        return hess_diag  # pyright: ignore[reportReturnType]

    @override
    def backward_params(self, primals_out: O) -> I:
        for transform in reversed(self.data):
            primals_out = transform.backward_params(primals_out)
        return primals_out  # pyright: ignore[reportReturnType]

    @override
    def backward_hess_diag(self, hess_diag_out: O) -> I:
        for transform in reversed(self.data):
            hess_diag_out = transform.backward_hess_diag(hess_diag_out)
        return hess_diag_out  # pyright: ignore[reportReturnType]

    def tree_flatten(self) -> tuple[Children, AuxData]:
        return self.data, None

    def tree_flatten_with_keys(self) -> tuple[KeyLeafPairs, AuxData]:
        children: KeyLeafPairs = [
            (jtu.SequenceKey(i), transform) for i, transform in enumerate(self.data)
        ]
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux: AuxData, children: Children) -> Self:
        return cls(children)


def chain_transforms(*transforms: LinearTransform | None) -> LinearTransform | None:
    flatten: list[LinearTransform] = []
    for t in transforms:
        if t is None:
            continue
        if isinstance(t, TransformChain):
            flatten.extend(t.data)
        else:
            flatten.append(t)
    if not flatten:
        return None
    if len(flatten) == 1:
        return flatten[0]
    return TransformChain(flatten)
