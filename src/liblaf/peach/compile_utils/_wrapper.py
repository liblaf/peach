import functools
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import wrapt

from liblaf.peach import tree

type InnerCallable = Callable[
    [Iterable[Any], Iterable[Any], tree.AuxData], tuple[Iterable[Any], tree.AuxData]
]


@tree.frozen
class Inner:
    fun_static: tree.AuxData = tree.static()

    def __call__(
        self,
        fun_dynamic: Iterable[Any],
        inputs_dynamic: Iterable[Any],
        inputs_static: tree.AuxData,
    ) -> tuple[Iterable[Any], tree.AuxData]:
        __tracebackhide__ = True
        fun: Callable[..., Any] = tree.combine(fun_dynamic, self.fun_static)
        args: Sequence[Any]
        kwargs: Mapping[str, Any]
        args, kwargs = tree.combine(inputs_dynamic, inputs_static)
        output: Any = fun(*args, **kwargs)
        output_dynamic: Iterable[Any]
        output_static: tree.AuxData
        output_dynamic, output_static = tree.partition(output)
        return output_dynamic, output_static


@tree.define
class Outer:
    fun_dynamic: Iterable[Any]

    @wrapt.decorator
    def __call__(
        self,
        wrapped: Callable[..., Any],
        _instance: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        __tracebackhide__ = True
        if inspect.ismethod(wrapped):
            args = (wrapped.__self__, *args)
            wrapped = wrapped.__func__
        inputs_dynamic: Iterable[Any]
        inputs_static: tree.AuxData
        inputs_dynamic, inputs_static = tree.partition((args, kwargs))
        outputs_dynamic: Iterable[Any]
        outputs_static: tree.AuxData
        outputs_dynamic, outputs_static = wrapped(
            self.fun_dynamic, inputs_dynamic, inputs_static
        )
        outputs: Any = tree.combine(outputs_dynamic, outputs_static)
        return outputs


# Cache inner wrappers to preserve JAX's compiled function cache. JAX caches
# compiled functions internally, but discards them when inner wrappers are
# garbage collected. By caching wrappers with lru_cache, we keep them alive
# across calls, avoiding redundant recompilation.
#
# We use lru_cache instead of WeakKeyDictionary because the inner wrapper holds
# a reference to the static function part (used as cache key), creating a
# circular reference that prevents garbage collection.
@functools.lru_cache
def wraps_inner(fun_static: tree.AuxData) -> Inner:
    return Inner(fun_static)
