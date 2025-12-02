from liblaf.peach import tree
from liblaf.peach.linalg.abc import State, Stats


@tree.define
class CompositeState(State):
    state: list[State] = tree.field(factory=list)


@tree.define
class CompositeStats(Stats):
    stats: list[Stats] = tree.field(factory=list)
