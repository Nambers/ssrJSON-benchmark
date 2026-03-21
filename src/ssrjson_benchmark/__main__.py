import sys


def _check_visual_deps():
    from importlib.util import find_spec

    libs = ["matplotlib", "svglib", "reportlab"]
    for lib in libs:
        if find_spec(lib) is None:
            raise ImportError(
                f"Library {lib} is required for visual report generation. "
                f"Please install it by `pip install ssrjson_benchmark[visual]` "
                f"or `pip install {' '.join(libs)}`."
            )


def _cmd_benchmark(args) -> int:
    import os
    import pathlib

    from .benchmark import run_benchmark

    _benchmark_files_dir = args.in_dir
    if not _benchmark_files_dir:
        _benchmark_files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_files"
        )
    benchmark_files = sorted(pathlib.Path(_benchmark_files_dir).glob("*.json"))
    if not benchmark_files:
        print(f"No benchmark file found using given path: {_benchmark_files_dir}")
        return 1

    process_bytes = int(args.process_gigabytes * 1024 * 1024 * 1024)
    bin_process_bytes = args.bin_process_megabytes * 1024 * 1024
    if process_bytes <= 0 or bin_process_bytes <= 0:
        print("process-gigabytes and bin-process-megabytes must be positive.")
        return 1

    _, default_file = run_benchmark(benchmark_files, process_bytes, bin_process_bytes)
    # If user specified -o, move result to that path
    if args.output:
        import shutil

        if args.output != default_file:
            shutil.move(default_file, args.output)
        print(f"Benchmark result saved to {args.output}")
    else:
        print(f"Benchmark result saved to {default_file}")
    return 0


def _cmd_print(args) -> int:
    import json

    from .benchmark import parse_file_result

    _check_visual_deps()

    with open(args.result_json, "rb") as f:
        result_ = json.load(f)
    result = parse_file_result(result_)

    from .report import generate_report_markdown, generate_report_pdf

    file = args.result_json.split("/")[-1].split("\\")[-1]
    out_dir = args.out_dir

    if args.markdown:
        generate_report_markdown(result, file, out_dir)
    if not args.no_pdf:
        generate_report_pdf(result, file, out_dir)
    if args.no_pdf and not args.markdown:
        print("Nothing to do. Use --markdown or remove --no-pdf.")
        return 1
    return 0


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="ssrjson_benchmark",
        description="ssrJSON benchmark tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- benchmark subcommand ---
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks and save result to a JSON file.",
    )
    bench_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. If not provided, uses a default name based on ssrjson version.",
        required=False,
        default=None,
    )
    bench_parser.add_argument(
        "-d",
        "--in-dir",
        help="Benchmark JSON files directory. If not provided, use the files bundled in this package.",
        required=False,
    )
    bench_parser.add_argument(
        "--process-gigabytes",
        help="Total gigabytes to process per test case, default 0.25 (float)",
        required=False,
        default=0.25,
        type=float,
    )
    bench_parser.add_argument(
        "--bin-process-megabytes",
        help="Maximum bytes to process per bin, default 8 (int)",
        required=False,
        default=8,
        type=int,
    )

    # --- print subcommand ---
    print_parser = subparsers.add_parser(
        "print",
        help="Generate PDF/Markdown report from a benchmark result JSON file.",
    )
    print_parser.add_argument(
        "result_json",
        help="Path to a benchmark result JSON file.",
    )
    print_parser.add_argument(
        "-m",
        "--markdown",
        help="Generate Markdown report",
        action="store_true",
    )
    print_parser.add_argument(
        "--no-pdf",
        help="Don't generate PDF report",
        action="store_true",
    )
    print_parser.add_argument(
        "--out-dir",
        help="Output directory for reports",
        required=False,
        default=os.getcwd(),
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "benchmark":
        return _cmd_benchmark(args)
    elif args.command == "print":
        return _cmd_print(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
