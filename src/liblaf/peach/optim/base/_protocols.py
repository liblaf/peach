from typing import Protocol

from jaxtyping import Array, Float

from liblaf.peach.utils import not_implemented

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class BaseProblem[X](Protocol):
    """Marker protocol for optimizer problems."""


class State(Protocol):
    """Protocol for optimizer states that expose current parameters."""

    @property
    def params(self) -> Vector:
        """Current optimizer parameters."""
        ...


class Stats(Protocol):
    """Protocol for optimizer-specific summary statistics."""


class Problem[X](Protocol):
    """Protocol implemented by differentiable optimization problems.

    Optimizers evaluate objective derivatives on a model state. Line-search
    optimizers build candidate states with [`before_trial`][liblaf.peach.optim.base.Problem.before_trial],
    while adapters that receive fresh parameter vectors from an external driver
    can use [`before_step`][liblaf.peach.optim.base.Problem.before_step] to
    synchronize that state first.
    """

    @not_implemented
    def fun(self, state: X, /) -> Scalar:
        """Evaluate the scalar objective value."""
        ...

    @not_implemented
    def grad(self, state: X, /) -> Vector:
        """Evaluate the objective gradient."""
        ...

    @not_implemented
    def hess_diag(self, state: X, /) -> Vector:
        """Evaluate or approximate the Hessian diagonal."""
        ...

    @not_implemented
    def hess_prod(self, state: X, p: Vector, /) -> Vector:
        """Evaluate the Hessian-vector product along `p`."""
        ...

    @not_implemented
    def hess_quad(self, state: X, p: Vector, /) -> Scalar:
        """Evaluate the quadratic form `p.T @ H @ p`."""
        ...

    @not_implemented
    def value_and_grad(self, state: X, /) -> tuple[Scalar, Vector]:
        """Evaluate the objective and gradient together."""
        ...

    @not_implemented
    def before_step(self, state: X, x: Vector, /) -> X:
        """Synchronize model state with externally supplied parameters."""
        ...

    @not_implemented
    def before_trial(self, state: X, x: Vector, /) -> X:
        """Build model state for a line-search trial at `x`."""
        ...

    @not_implemented
    def max_step_size(self, state: X, p: Vector, /) -> Scalar:
        """Return a safe fraction of proposed trial displacement `p`."""
        ...

    @not_implemented
    def callback(self, model_state: X, opt_state: State, /) -> None:
        """Run an optional side-effect after an optimizer step."""
        ...
