from typing import override

import numpy as np
import scipy
from jaxtyping import Float

from ._base import ScipySolver

type FreeNp = Float[np.ndarray, " free"]


class ScipyBiCGStab(ScipySolver):
    @override
    def _wrapped(self, *args, **kwargs) -> tuple[FreeNp, int]:
        return scipy.sparse.linalg.bicgstab(*args, **kwargs)
