from . import _ssrjson_benchmark as internal
from .benchmark import run_benchmark, parse_file_result
from .report import generate_report_markdown, generate_report_pdf

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
    "parse_file_result",
    "__version__",
]
