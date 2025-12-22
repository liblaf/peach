from typing import Self

from liblaf.peach import tree


@tree.define
class Constraint:
    def flatten(self, structure: tree.Structure) -> Self:
        raise NotImplementedError
