from . import _ssrjson_benchmark as internal
from .benchmark_impl import (
    generate_report_markdown,
    generate_report_pdf,
    run_benchmark,
)

try:
    from importlib.metadata import version

    __version__ = version("ssrjson-benchmark")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "generate_report_markdown",
    "generate_report_pdf",
    "internal",
    "run_benchmark",
    "__version__",
]
