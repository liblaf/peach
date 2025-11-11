import time

from liblaf.peach import tree


@tree.define
class ScipyStats:
    n_steps: int = 0
    start_time: float = tree.field(factory=time.perf_counter)
    end_time: float | None = None

    @property
    def time(self) -> float:
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
