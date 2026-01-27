from collections.abc import Generator, Iterable
from typing import Any, override

import jarp

from ._base import Transform
from ._identity import IdentityTransform


@jarp.define
class ChainTransform[I, O](Transform[I, O]):
    data: list[Transform] = jarp.field(factory=list)

    @override
    @jarp.jit(inline=True)
    def forward_primals(self, primals_in: I) -> O:
        primals: Any = primals_in
        for transform in self.data:
            primals = transform.forward_primals(primals)
        return primals

    @override
    @jarp.jit(inline=True)
    def forward_tangents(self, tangents_in: I) -> O:
        tangents: Any = tangents_in
        for transform in self.data:
            tangents = transform.forward_tangents(tangents)
        return tangents

    @override
    @jarp.jit(inline=True)
    def backward_primals(self, primals_out: O) -> I:
        primals: Any = primals_out
        for transform in reversed(self.data):
            primals = transform.backward_primals(primals)
        return primals

    @override
    @jarp.jit(inline=True)
    def backward_tangents(self, tangents_out: O) -> I:
        tangents: Any = tangents_out
        for transform in reversed(self.data):
            tangents = transform.backward_tangents(tangents)
        return tangents

    @override
    @jarp.jit(inline=True)
    def backward_hess_diag(self, hess_diag_out: O) -> I:
        hess_diag: Any = hess_diag_out
        for transform in reversed(self.data):
            hess_diag = transform.backward_hess_diag(hess_diag)
        return hess_diag


def chain_transforms(*transforms: Transform | None) -> Transform:
    transforms: list[Transform] = list(_iter_transforms(transforms))
    if not transforms:
        return IdentityTransform()
    if len(transforms) == 1:
        return transforms[0]
    return ChainTransform(data=transforms)


def _iter_transforms(transforms: Iterable[Transform | None]) -> Generator[Transform]:
    for transform in transforms:
        if transform is None:
            continue
        if isinstance(transform, IdentityTransform):
            continue
        if isinstance(transform, ChainTransform):
            yield from _iter_transforms(transform.data)
        else:
            yield transform
