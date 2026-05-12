from typing import override

from liblaf.peach.optim.base import Problem
from liblaf.peach.utils import implemented, not_implemented


class ProtocolOnlyProblem(Problem[object]): ...


class CallbackProblem(Problem[object]):
    @override
    def callback(self, model_state: object, opt_state: object, /) -> None:
        del model_state, opt_state


def test_implemented_rejects_protocol_stub_methods() -> None:
    problem = ProtocolOnlyProblem()

    assert hasattr(problem, "callback")
    assert not implemented(problem, Problem.callback)
    assert not implemented(problem, "callback")


def test_implemented_accepts_overridden_protocol_methods() -> None:
    problem = CallbackProblem()

    assert implemented(problem, Problem.callback)
    assert implemented(problem, "callback")


def test_implemented_rejects_missing_none_and_explicit_markers() -> None:
    class Hooks:
        skipped = None

        @not_implemented
        def fallback(self) -> None: ...

    hooks = Hooks()

    assert not implemented(hooks, "missing")
    assert not implemented(hooks, "skipped")
    assert not implemented(hooks, Hooks.fallback)
