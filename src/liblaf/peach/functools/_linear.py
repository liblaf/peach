from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree

from ._context import FunctionContext
from ._descriptor import MethodDescriptor

type Vector = Float[Array, " N"]


@tree.define
class AbstractLinearOperator:
    matvec = MethodDescriptor(in_structures={0: "input"}, out_structures={0: "output"})


@tree.define
class LinearOperator(AbstractLinearOperator, FunctionContext):
    matvec = MethodDescriptor(in_structures={0: "input"}, out_structures={0: "output"})  # pyright: ignore[reportAssignmentType]
    _matvec_wrapped: Callable | None = tree.field(default=None, alias="matvec")
    _matvec_wrapper: Callable | None = tree.field(default=None, init=False)

    def flatten(
        self,
        input_structure: tree.Structure,
        output_structure: tree.Structure | None = None,
    ) -> FunctionContext:
        if output_structure is None:
            output_structure = input_structure
        return self.with_structures(
            {"input": input_structure, "output": output_structure}, flatten=True
        )


@tree.define
class DiagonalLinearOperator(AbstractLinearOperator, FunctionContext):
    diagonal: PyTree
    diagonal_flat = tree.FlatView()
    # explicitly bypass structure handling here, so that we can directly work with flat vectors
    matvec = MethodDescriptor(in_structures={}, out_structures={})

    @property
    def structure(self) -> tree.Structure | None:
        return self._structures.get("input", None)

    @structure.setter
    def structure(self, structure: tree.Structure) -> None:
        self._structures["input"] = structure

    def flatten(self, structure: tree.Structure) -> FunctionContext:
        return self.with_structures({"input": structure}, flatten=True)

    @eqx.filter_jit
    def _matvec_wrapped(self, x: PyTree | Vector, **kwargs) -> PyTree | Vector:
        flatten: bool = kwargs.pop("_flatten", self._flatten)
        # trigger flattening, this will generate self.structure if is None
        diag_flat: Vector = self.diagonal_flat
        assert self.structure is not None
        x_flat: Vector = x if flatten else self.structure.flatten(x)
        y_flat: Vector = diag_flat * x_flat
        return y_flat if flatten else self.structure.unflatten(y_flat)


@tree.define
class SquareLinearOperator(AbstractLinearOperator, FunctionContext):
    matvec = MethodDescriptor(in_structures={0: "input"}, out_structures={0: "input"})
    _matvec_wrapped: Callable | None = tree.field(default=None, alias="matvec")
    _matvec_wrapper: Callable | None = tree.field(default=None, init=False)

    def flatten(self, structure: tree.Structure) -> FunctionContext:
        return self.with_structures({"input": structure}, flatten=True)
