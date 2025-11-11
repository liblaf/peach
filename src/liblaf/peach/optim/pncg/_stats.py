from liblaf.peach import tree
from liblaf.peach.optim.abc import Stats


@tree.define
class PNCGStats(Stats):
    n_steps: int = 0
    time: float = 0.0
