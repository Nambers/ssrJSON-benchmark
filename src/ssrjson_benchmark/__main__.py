from importlib.util import find_spec
import sys


def _check_visual_deps():
    libs = ["matplotlib", "svglib", "reportlab"]
    for lib in libs:
        if find_spec(lib) is None:
            raise ImportError(
                f"Library {lib} is required for visual report generation. "
                f"Please install it by `pip install ssrjson_benchmark[visual]` "
                f"or `pip install {' '.join(libs)}`."
            )


def main():
    import argparse
    import json
    import os
    import pathlib

    from .benchmark import parse_file_result, run_benchmark

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="Use a result JSON file generated in previous benchmark to print report. Will skip all tests.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--in-dir",
        help="Benchmark JSON files directory. If not provided, use the files bundled in this package.",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--markdown",
        help="Generate Markdown report",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--no-pdf",
        help="Don't generate PDF report",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--process-gigabytes",
        help="Total gigabytes to process per test case, default 0.25 (float)",
        required=False,
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--bin-process-megabytes",
        help="Maximum bytes to process per bin, default 8 (int)",
        required=False,
        default=8,
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for reports",
        required=False,
        default=os.getcwd(),
    )
    parser.add_argument(
        "-t",
        "--text-only",
        help="Only collect benchmark into JSON file.",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()
    skip_tests = bool(args.file)
    if skip_tests and args.no_pdf and not args.markdown:
        print("Nothing to do.")
        return 0

    _benchmark_files_dir = args.in_dir
    if not _benchmark_files_dir:
        _benchmark_files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_files"
        )
    benchmark_files_dir = sorted(pathlib.Path(_benchmark_files_dir).glob("*.json"))
    if not benchmark_files_dir:
        print(f"No benchmark file found using given path: {_benchmark_files_dir}")
        return 1

    if skip_tests:
        with open(args.file, "rb") as f:
            result_ = json.load(f)
        result = parse_file_result(result_)
        file = args.file.split("/")[-1]
    else:
        process_bytes = int(args.process_gigabytes * 1024 * 1024 * 1024)
        bin_process_bytes = args.bin_process_megabytes * 1024 * 1024
        if process_bytes <= 0 or bin_process_bytes <= 0:
            print("process-gigabytes and bin-process-megabytes must be positive.")
            return 1
        result, file = run_benchmark(
            benchmark_files_dir, process_bytes, bin_process_bytes
        )
        file = file.split("/")[-1]

    if not args.text_only:
        _check_visual_deps()
        from .report import generate_report_markdown, generate_report_pdf

        if args.markdown:
            generate_report_markdown(result, file, args.out_dir)
        if not args.no_pdf:
            generate_report_pdf(result, file, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
