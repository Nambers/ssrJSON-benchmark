import argparse
import json
import os
from ssrjson_benchmark import (
    run_benchmark,
    generate_report_markdown,
    generate_report,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--file", help="record JSON file", required=False, default=None
    )
    parser.add_argument(
        "-m",
        "--markdown",
        help="Generate markdown report",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--process-bytes",
        help="Total process bytes per test, default 1e8",
        required=False,
        default=1e8,
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for reports",
        required=False,
        default=os.getcwd(),
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            j = json.load(f)
        file = args.file.split("/")[-1]
    else:
        j, file = run_benchmark(args.process_bytes)
        file = file.split("/")[-1]

    if args.markdown:
        generate_report_markdown(j, file, args.out_dir)
    else:
        generate_report(j, file, args.out_dir)


if __name__ == "__main__":
    main()
