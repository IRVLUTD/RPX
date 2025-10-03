import time
from typing import Any

class Profiler:
    """A context manager to profile a block of code's execution time."""
    def __enter__(self) -> 'Profiler':
        self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.runtime_sec = time.perf_counter() - self.start_time

        self.gpu_mem_mb = 4096.0  # Placeholder
        self.power_watts = 150.0   # Placeholder