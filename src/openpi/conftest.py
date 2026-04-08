import os

import pytest

try:
    import pynvml
except ModuleNotFoundError:
    pynvml = None


def set_jax_cpu_backend_if_no_gpu() -> None:
    if pynvml is None:
        # NVML bindings are optional; default to CPU if unavailable.
        os.environ["JAX_PLATFORMS"] = "cpu"
        return

    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()