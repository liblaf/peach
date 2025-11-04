from collections.abc import Callable

from scipy.optimize import Bounds

from liblaf.peach import tree_utils


@tree_utils.tree
class ObjectiveStruct:
    fun: Callable | None = None
    grad: Callable | None = None
    hess: Callable | None = None
    hess_diag: Callable | None = None
    hess_prod: Callable | None = None
    hess_quad: Callable | None = None
    value_and_grad: Callable | None = None
    grad_and_hess_diag: Callable | None = None
    bounds: Bounds | None = None
