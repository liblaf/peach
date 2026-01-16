# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Self, overload

import attrs
import jax.tree_util as jtu
import wrapt

from liblaf.peach import tree

type Children[**P, T] = tuple[Callable[P, T], Sequence[Any], Mapping[str, Any]]
type KeyEntry = Any
type KeyLeafPair = tuple[KeyEntry, Any]
type KeyLeafPairs = Iterable[KeyLeafPair]


_FIELD_NAMES: list[str] = [
    "flatten",
    "in_structures",
    "out_structures",
    "jit",
    "args",
    "kwargs",
    "n_outputs",
    "with_aux",
]


@attrs.frozen
class _FunctionWrapperAuxData:
    # flatten
    flatten: bool
    in_structures: frozenset[tuple[int, tree.Structure]]
    out_structures: frozenset[tuple[int, tree.Structure]]
    # jit
    jit: bool
    # with_aux
    n_outputs: int
    with_aux: bool

    @classmethod
    def from_wrapper(cls, wrapper: FunctionWrapper) -> Self:
        return cls(
            n_outputs=wrapper._self_n_outputs,
            flatten=wrapper._self_flatten,
            in_structures=frozenset(wrapper._self_in_structures.items()),
            out_structures=frozenset(wrapper._self_out_structures.items()),
            jit=wrapper._self_jit,
            with_aux=wrapper._self_with_aux,
        )

    def dump(self) -> dict[str, Any]:
        return {
            "flatten": self.flatten,
            "in_structures": dict(self.in_structures),
            "out_structures": dict(self.out_structures),
            "jit": self.jit,
            "n_outputs": self.n_outputs,
            "with_aux": self.with_aux,
        }


@jtu.register_pytree_with_keys_class
class FunctionWrapper[**P, T](wrapt.CallableObjectProxy):
    __wrapped__: Callable[P, T]

    # flatten
    _self_flatten: bool = False
    _self_in_structures: Mapping[int, tree.Structure] = {}
    _self_out_structures: Mapping[int, tree.Structure] = {}
    # jit
    _self_jit: bool = False
    # partial
    _self_args: Sequence[Any] = ()
    _self_kwargs: Mapping[str, Any] = {}
    # with_aux
    _self_n_outputs: int = 1
    _self_with_aux: bool = False

    @overload
    def __init__(  # pyright: ignore[reportInconsistentOverload]
        self,
        wrapped: Callable[P, T],
        *,
        flatten: bool = False,
        in_structures: Mapping[int, tree.Structure] = {},
        out_structures: Mapping[int, tree.Structure] = {},
        jit: bool = True,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        n_outputs: int = 1,
        with_aux: bool = False,
    ) -> None: ...
    def __init__(self, wrapped: Callable[P, T], **kwargs) -> None:
        super().__init__(wrapped)
        for key, value in kwargs.items():
            setattr(self, f"_self_{key}", value)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        jit: bool = kwargs.pop("_jit", self._self_jit)  # pyright: ignore[reportAssignmentType]
        if jit:
            return self._call_jit(*args, **kwargs)
        return self._call(*args, **kwargs)

    def __replace__(self, **changes: Any) -> Self:
        for field in _FIELD_NAMES:
            if field not in changes:
                changes[field] = getattr(self, f"_self_{field}")
        return type(self)(self.__wrapped__, **changes)

    def flatten(
        self,
        inputs: Mapping[int, tree.Structure] = {},
        outputs: Mapping[int, tree.Structure] = {},
        *,
        flatten: bool = True,
    ) -> Self:
        return self.__replace__(
            flatten=flatten, in_structures=inputs, out_structures=outputs
        )

    def jit(self, *, jit: bool = True) -> Self:
        return self.__replace__(jit=jit)

    def partial(self, *args: Any, **kwargs: Any) -> Self:
        return self.__replace__(
            args=(*self._self_args, *args), kwargs={**self._self_kwargs, **kwargs}
        )

    def with_aux(self, *, with_aux: bool = True) -> Self:
        return self.__replace__(with_aux=with_aux)

    def tree_flatten(self) -> tuple[Children[P, T], _FunctionWrapperAuxData]:
        children: Children[P, T] = (
            self.__wrapped__,
            self._self_args,
            self._self_kwargs,
        )
        return children, _FunctionWrapperAuxData.from_wrapper(self)

    def tree_flatten_with_keys(
        self,
    ) -> tuple[KeyLeafPairs, _FunctionWrapperAuxData]:
        children: KeyLeafPairs = [
            (jtu.GetAttrKey("__wrapped__"), self.__wrapped__),
            (jtu.GetAttrKey("args"), self._self_args),
            (jtu.GetAttrKey("kwargs"), self._self_kwargs),
        ]
        return children, _FunctionWrapperAuxData.from_wrapper(self)

    @classmethod
    def tree_unflatten(
        cls, aux: _FunctionWrapperAuxData, children: Children[P, T]
    ) -> Self:
        wrapped: Callable[P, T]
        args: Sequence[Any]
        kwargs: Mapping[str, Any]
        wrapped, args, kwargs = children
        return cls(wrapped, args=args, kwargs=kwargs, **aux.dump())

    def _call(self, *args: P.args, **kwargs: P.kwargs) -> T:
        flatten: bool = kwargs.pop("_flatten", self._self_flatten)  # pyright: ignore[reportAssignmentType]
        with_aux: bool = kwargs.pop("_with_aux", self._self_with_aux)  # pyright: ignore[reportAssignmentType]
        args: Sequence[Any] = (*self._self_args, *args)
        kwargs: Mapping[str, Any] = {**self._self_kwargs, **kwargs}
        if flatten:
            args = _unflatten_inputs(args, self._self_in_structures)
        outputs: Sequence[Any] = _as_tuple(self.__wrapped__(*args, **kwargs))  # pyright: ignore[reportCallIssue]
        if flatten:
            outputs = _flatten_outputs(outputs, self._self_out_structures)
        outputs = _with_aux(outputs, n_outputs=self._self_n_outputs, with_aux=with_aux)
        return outputs[0] if len(outputs) == 1 else outputs  # pyright: ignore[reportReturnType]

    @tree.jit
    def _call_jit(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._call(*args, **kwargs)


def _unflatten_inputs(
    inputs: Iterable[Any], structures: Mapping[int, tree.Structure]
) -> list[Any]:
    inputs: list[Any] = list(inputs)
    for i, structure in structures.items():
        inputs[i] = structure.unflatten(inputs[i])
    return inputs


def _as_tuple(outputs: Any) -> tuple[Any, ...]:
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    return outputs


def _flatten_outputs(
    outputs: Iterable[Any], structures: Mapping[int, tree.Structure]
) -> list[Any]:
    outputs: list[Any] = list(outputs)
    for i, structure in structures.items():
        outputs[i] = structure.flatten(outputs[i])
    return outputs


def _with_aux(outputs: Sequence[Any], *, n_outputs: int, with_aux: bool) -> Any:
    if with_aux:
        if len(outputs) == n_outputs:
            return *outputs, None
        if len(outputs) == n_outputs + 1:
            return outputs
        raise ValueError(outputs)
    if len(outputs) == n_outputs:
        return outputs
    if len(outputs) == n_outputs + 1:
        return outputs[:-1]
    raise ValueError(outputs)
