from typing import override

from liblaf.peach.optim.base import Problem
from liblaf.peach.utils import is_implemented, not_implemented


class ProtocolOnlyProblem(Problem[object]): ...


class CallbackProblem(Problem[object]):
    @override
    def callback(self, model_state: object, opt_state: object, /) -> None:
        del model_state, opt_state


def test_implemented_rejects_protocol_stub_methods() -> None:
    problem = ProtocolOnlyProblem()

    assert hasattr(problem, "callback")
    assert not is_implemented(problem, Problem.callback)
    assert not is_implemented(problem, "callback")


def test_implemented_accepts_overridden_protocol_methods() -> None:
    problem = CallbackProblem()

    assert is_implemented(problem, Problem.callback)
    assert is_implemented(problem, "callback")


def test_implemented_rejects_missing_none_and_explicit_markers() -> None:
    class Hooks:
        skipped = None

        @not_implemented
        def fallback(self) -> None: ...

    hooks = Hooks()

    assert not is_implemented(hooks, "missing")
    assert not is_implemented(hooks, "skipped")
    assert not is_implemented(hooks, Hooks.fallback)
