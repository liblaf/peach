from typing import override

import numpy as np
import scipy
from jaxtyping import Float

from liblaf.peach import tree

from ._base import ScipySolver

type FreeNp = Float[np.ndarray, " free"]


@tree.define
class ScipyBiCG(ScipySolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeNp, int]:
        return scipy.sparse.linalg.bicg(*args, **kwargs)
