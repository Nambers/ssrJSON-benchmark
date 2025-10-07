from .benchmark_impl import (
    generate_report_pdf,
    generate_report_markdown,
    run_benchmark,
)

try:
    from importlib.metadata import version

    __version__ = version("ssrjson-benchmark")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "run_benchmark",
    "generate_report_markdown",
    "generate_report_pdf",
    "__version__",
]
