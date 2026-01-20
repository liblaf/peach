from jaxtyping import Array, Float, PyTree

from liblaf.peach import tree

type Params = PyTree
type Vector = Float[Array, " N"]


@tree.define
class Constraint:
    structure: tree.Structure = tree.static(default=None, repr=False, kw_only=True)

    def process_input_params(self, params: Params) -> Params:
        raise NotImplementedError

    def process_input_grad(self, params: Params, grad: Params) -> Params:
        raise NotImplementedError

    def process_output_grad(self, grad: Params) -> Params:
        raise NotImplementedError
