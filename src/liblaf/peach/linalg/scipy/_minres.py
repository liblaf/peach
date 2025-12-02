from typing import Any, override

import numpy as np
import scipy
from jaxtyping import Float

from liblaf.peach import tree
from liblaf.peach.linalg.system import LinearSystem

from ._base import ScipySolver

type FreeNp = Float[np.ndarray, " free"]


@tree.define
class ScipyMINRES(ScipySolver):
    shift: float = tree.field(default=0.0, kw_only=True)
    show: bool = tree.field(default=False, kw_only=True)
    check: bool = tree.field(default=False, kw_only=True)

    @override
    def _options(self, system: LinearSystem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(system)
        options.update({"shift": self.shift, "show": self.show, "check": self.check})
        options.pop("atol")
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeNp, int]:
        return scipy.sparse.linalg.minres(*args, **kwargs)
