import abc

from jaxtyping import Array, Float

from liblaf.peach import tree
from liblaf.peach.optim.objective import Objective

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.define
class LineSearch:
    @abc.abstractmethod
    def search(
        self,
        objective: Objective,
        params: Vector,
        grad: Vector,
        search_direction: Vector,
    ) -> Scalar:
        raise NotImplementedError
