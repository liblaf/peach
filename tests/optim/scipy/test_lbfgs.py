import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.peach import testing, tree
from liblaf.peach.constraints import BoundConstraint, Constraint, FixedConstraint
from liblaf.peach.functools import Objective
from liblaf.peach.optim import ScipyOptimizer

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


@tree.define
class Params:
    data: Vector = tree.array()


@tree.define
class Model:
    structure: tree.Structure[Params] = tree.field(kw_only=True)

    def fun(self, params: Params | Vector) -> Scalar:
        params: Params = self.structure.unflatten(params)
        return testing.rosen(params.data)

    def grad(self, params: Params | Vector) -> Vector:
        params: Params = self.structure.unflatten(params)
        grad_flat: Vector = testing.rosen_grad(params.data)
        grad: Params = Params(data=grad_flat)
        return self.structure.flatten(grad)

    def hess_prod(self, params: Params | Vector, p: Params | Vector) -> Vector:
        params: Params = self.structure.unflatten(params)
        p: Params = self.structure.unflatten(p)
        hess_prod_flat: Vector = testing.rosen_hess_prod(params.data, p.data)
        hess_prod: Params = Params(data=hess_prod_flat)
        return self.structure.flatten(hess_prod)

    def hess_quad(self, params: Params | Vector, p: Params | Vector) -> Scalar:
        params: Params = self.structure.unflatten(params)
        p: Params = self.structure.unflatten(p)
        return testing.rosen_hess_quad(params.data, p.data)

    def value_and_grad(self, params: Params | Vector) -> tuple[Scalar, Vector]:
        params: Params = self.structure.unflatten(params)
        value: Scalar
        grad_flat: Vector
        value, grad_flat = testing.rosen_value_and_grad(params.data)
        grad: Params = Params(data=grad_flat)
        return value, self.structure.flatten(grad)


def callback(state: ScipyOptimizer.State, _stats: ScipyOptimizer.Stats) -> None:
    assert isinstance(state.params, Params)
    assert state.params.static_field == "foo"


def test_scipy_lbfgs_tree() -> None:
    objective: Objective = Objective(value_and_grad=rosen_value_and_grad_tree)
    params: Params = Params(jnp.zeros((7,)))
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        tol=1e-10,
    )
    solution: ScipyOptimizer.Solution = optimizer.minimize(
        objective, params, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((7,)))


def test_scipy_lbfgs_tree_fixed() -> None:
    objective: Objective = Objective(value_and_grad=rosen_value_and_grad_tree)
    params: Params = Params(jnp.zeros((9,)))
    fixed_mask: Params = Params(jnp.zeros((9,), jnp.bool))
    params.x = params.x.at[-2:].set(1.0)
    fixed_mask.x = fixed_mask.x.at[-2:].set(True)
    constraints: list[Constraint] = [FixedConstraint(mask=fixed_mask)]

    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        tol=1e-10,
    )
    solution: ScipyOptimizer.Solution = optimizer.minimize(
        objective, params, constraints=constraints, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((9,)))


def test_scipy_lbfgs_tree_bound() -> None:
    def fun(x: Params) -> Float[Array, ""]:
        return jnp.sum(jnp.square(x.x))

    objective: Objective = Objective(
        fun=fun,
        grad=eqx.filter_grad(fun),
        value_and_grad=eqx.filter_value_and_grad(fun),
    )
    params: Params = Params(jnp.full((7,), 2.0))
    constraints: list[Constraint] = [BoundConstraint(lower=Params(jnp.ones((7,))))]
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        tol=1e-10,
    )
    solution: ScipyOptimizer.Solution = optimizer.minimize(
        objective, params, constraints=constraints, callback=callback
    )
    assert solution.success
    params = solution.params
    np.testing.assert_allclose(params.x, jnp.ones((7,)))
