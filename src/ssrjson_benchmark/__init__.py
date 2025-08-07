from .benchmark_main import (
    run_benchmark,
    generate_report_markdown,
    generate_report,
    run_benchmark_default,
)
from importlib.metadata import version

__version__ = version("ssrjson-benchmark")

__all__ = [
    "run_benchmark",
    "generate_report_markdown",
    "generate_report",
    "run_benchmark_default",
    "__version__",
]
