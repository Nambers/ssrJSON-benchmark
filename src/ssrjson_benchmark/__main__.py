def main():
    import argparse
    import json
    import os
    import pathlib
    import sys

    from .benchmark_impl import (
        generate_report_markdown,
        generate_report_pdf,
        parse_file_result,
        run_benchmark,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--file", help="record JSON file", required=False, default=None
    )
    parser.add_argument(
        "-d",
        "--in-dir",
        help="Benchmark JSON files directory",
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
        help="Total process gigabytes per test, default 0.1 (float)",
        required=False,
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--bin-process-megabytes",
        help="Maximum bytes to process per read for binary formats, default 32 (int)",
        required=False,
        default=32,
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for reports",
        required=False,
        default=os.getcwd(),
    )
    args = parser.parse_args()
    if args.file and args.no_pdf and not args.markdown:
        print("Nothing to do.")
        sys.exit(0)

    _benchmark_files_dir = args.in_dir
    if not _benchmark_files_dir:
        _benchmark_files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_files"
        )
    benchmark_files_dir = sorted(pathlib.Path(_benchmark_files_dir).glob("*.json"))
    if not benchmark_files_dir:
        print(f"No benchmark file found using given path: {_benchmark_files_dir}")
        sys.exit(0)

    if args.file:
        with open(args.file, "rb") as f:
            result_ = json.load(f)
        result = parse_file_result(result_)
        file = args.file.split("/")[-1]
    else:
        process_bytes = int(args.process_gigabytes * 1024 * 1024 * 1024)
        bin_process_bytes = args.bin_process_megabytes * 1024 * 1024
        if process_bytes <= 0 or bin_process_bytes <= 0:
            print("process-gigabytes and bin-process-megabytes must be positive.")
            sys.exit(1)
        result, file = run_benchmark(
            benchmark_files_dir, process_bytes, bin_process_bytes
        )
        file = file.split("/")[-1]

    if args.markdown:
        generate_report_markdown(result, file, args.out_dir)
    if not args.no_pdf:
        generate_report_pdf(result, file, args.out_dir)


if __name__ == "__main__":
    main()
