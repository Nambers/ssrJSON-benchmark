from .benchmark_main import (
    run_benchmark,
    generate_report_markdown,
    generate_report,
    run_benchmark_default,
)
from ._ssrjson_benchmark import __version__

__all__ = [
    "run_benchmark",
    "generate_report_markdown",
    "generate_report",
    "run_benchmark_default",
    "__version__",
]
