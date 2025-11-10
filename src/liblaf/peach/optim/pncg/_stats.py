from liblaf.peach import tree_utils
from liblaf.peach.optim.abc import Stats


@tree_utils.tree
class PNCGStats(Stats):
    n_steps: int = 0
    time: float = 0.0
