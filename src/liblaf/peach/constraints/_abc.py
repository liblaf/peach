from typing import Self

from liblaf.peach import tree


@tree.define
class Constraint:
    def flatten(self, flat_def: tree.FlatDef) -> Self:
        raise NotImplementedError
