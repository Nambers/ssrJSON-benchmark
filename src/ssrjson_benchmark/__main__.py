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


def _derive_sibling_path(primary_path: str, new_ext: str) -> str:
    """Derive a sibling file path from *primary_path* by changing the extension.

    If primary_path (lowercased) ends with .pdf or .json, replace that suffix
    with *new_ext* (e.g. ".md", ".json").  Otherwise append *new_ext* directly.
    """
    lower = primary_path.lower()
    for suffix in (".pdf", ".json"):
        if lower.endswith(suffix):
            return primary_path[: -len(suffix)] + new_ext
    return primary_path + new_ext


def _ensure_parent_dir(path: str) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _add_benchmark_args(parser):
    """Add common benchmark arguments to a parser."""
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. For benchmark: JSON path; for print/full: PDF path "
        "(or Markdown path when only Markdown is generated).",
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
        "--process-gigabytes",
        help="Total gigabytes to process per test case, default 0.1 (float)",
        required=False,
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--bin-process-megabytes",
        help="Maximum megabytes to process per bin, default 8 (int)",
        required=False,
        default=8,
        type=int,
    )
    from .benchmark import BenchmarkCategory

    parser.add_argument(
        "--only",
        choices=[c.value for c in BenchmarkCategory],
        help="Only run a subset of tests: loads (2 groups), dumps (2 groups), or dumps_to_bytes (up to 4 groups).",
        required=False,
        default=None,
    )


def _resolve_benchmark_files(in_dir):
    """Resolve benchmark input files from a directory."""
    import os
    import pathlib

    if not in_dir:
        in_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_files")
    benchmark_files = sorted(pathlib.Path(in_dir).glob("*.json"))
    if not benchmark_files:
        print(f"No benchmark file found using given path: {in_dir}")
        return None
    return benchmark_files


def _run_benchmark_from_args(args):
    """Run benchmark from parsed args. Returns (result, json_path) or None on error."""
    from .benchmark import run_benchmark

    benchmark_files = _resolve_benchmark_files(args.in_dir)
    if benchmark_files is None:
        return None

    process_bytes = int(args.process_gigabytes * 1024 * 1024 * 1024)
    bin_process_bytes = args.bin_process_megabytes * 1024 * 1024
    if process_bytes <= 0 or bin_process_bytes <= 0:
        print("process-gigabytes and bin-process-megabytes must be positive.")
        return None

    result, default_file = run_benchmark(
        benchmark_files, process_bytes, bin_process_bytes, only=args.only
    )
    return result, default_file


def _cmd_benchmark(args) -> int:
    ret = _run_benchmark_from_args(args)
    if ret is None:
        return 1
    _, json_path = ret

    if args.output:
        import shutil

        if args.output != json_path:
            shutil.move(json_path, args.output)
        json_path = args.output

    print(f"Benchmark result saved to {json_path}")
    return 0


def _cmd_print(args) -> int:
    import json
    import os
    import shutil

    from .benchmark import parse_file_result

    _check_visual_deps()

    with open(args.result_json, "rb") as f:
        result_ = json.load(f)
    result = parse_file_result(result_)

    from .report import generate_report_markdown, generate_report_pdf

    file = os.path.basename(args.result_json)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.result_json))

    pdf_only = not args.no_pdf and not args.gen_markdown
    md_only = args.no_pdf and args.gen_markdown
    both = not args.no_pdf and args.gen_markdown

    if args.no_pdf and not args.gen_markdown:
        print("Nothing to do. Use --gen-markdown or remove --no-pdf.")
        return 1

    if args.gen_markdown:
        md_path = generate_report_markdown(result, file, out_dir)
        if args.output and md_only:
            if args.output != md_path:
                shutil.move(md_path, args.output)
            md_path = args.output
        elif args.output and both:
            md_dest = _derive_sibling_path(args.output, ".md")
            if md_dest != md_path:
                shutil.move(md_path, md_dest)
            md_path = md_dest
        print(f"Markdown report saved to {md_path}")

    if not args.no_pdf:
        pdf_path = generate_report_pdf(result, file, out_dir)
        if args.output and (pdf_only or both):
            if args.output != pdf_path:
                shutil.move(pdf_path, args.output)
            pdf_path = args.output
        print(f"PDF report saved to {pdf_path}")

    return 0


def _cmd_full(args) -> int:
    import os
    import shutil

    _check_visual_deps()

    do_pdf = not args.no_pdf
    do_md = args.gen_markdown
    if not do_pdf and not do_md:
        # Default: generate both when neither flag is given
        do_pdf = True
        do_md = True

    ret = _run_benchmark_from_args(args)
    if ret is None:
        return 1
    result, json_path = ret
    print(f"Benchmark result saved to {json_path}")

    from .report import generate_report_markdown, generate_report_pdf

    file = os.path.basename(json_path)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(json_path))

    pdf_only = do_pdf and not do_md
    md_only = do_md and not do_pdf
    both = do_pdf and do_md

    if args.output:
        _ensure_parent_dir(args.output)

    if do_md:
        md_path = generate_report_markdown(result, file, out_dir)
        if args.output and md_only:
            if args.output != md_path:
                shutil.move(md_path, args.output)
            md_path = args.output
        elif args.output and both:
            md_dest = _derive_sibling_path(args.output, ".md")
            if md_dest != md_path:
                shutil.move(md_path, md_dest)
            md_path = md_dest
        print(f"Markdown report saved to {md_path}")

    if do_pdf:
        pdf_path = generate_report_pdf(result, file, out_dir)
        if args.output and (pdf_only or both):
            if args.output != pdf_path:
                shutil.move(pdf_path, args.output)
            pdf_path = args.output
        print(f"PDF report saved to {pdf_path}")

    if args.keep_json:
        if args.output:
            json_dest = _derive_sibling_path(args.output, ".json")
            if json_dest != json_path:
                shutil.move(json_path, json_dest)
            json_path = json_dest
        print(f"JSON result saved to {json_path}")
    else:
        os.remove(json_path)
        print(f"Removed intermediate JSON file: {json_path}")

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="ssrjson_benchmark",
        description="ssrJSON benchmark tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    command_parsers = {}

    # --- benchmark subcommand ---
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks and save result to a JSON file.",
    )
    command_parsers["benchmark"] = bench_parser
    _add_benchmark_args(bench_parser)

    # --- print subcommand ---
    print_parser = subparsers.add_parser(
        "print",
        help="Generate PDF/Markdown report from a benchmark result JSON file.",
    )
    command_parsers["print"] = print_parser
    print_parser.add_argument(
        "result_json",
        help="Path to a benchmark result JSON file.",
    )
    print_parser.add_argument(
        "-o",
        "--output",
        help="Output file path. PDF path by default; Markdown path when only Markdown is generated. "
        "When both are generated, Markdown path is derived by replacing .pdf with .md (or appending .md).",
        required=False,
        default=None,
    )
    print_parser.add_argument(
        "--gen-markdown",
        help="Also generate a Markdown report.",
        action="store_true",
    )
    print_parser.add_argument(
        "--no-pdf",
        help="Don't generate PDF report",
        action="store_true",
    )
    print_parser.add_argument(
        "--out-dir",
        help="Output directory for reports. Defaults to the directory containing the result JSON.",
        required=False,
        default=None,
    )

    # --- full subcommand ---
    full_parser = subparsers.add_parser(
        "full",
        help="Run benchmarks, generate reports (PDF/Markdown), then delete the intermediate JSON.",
    )
    command_parsers["full"] = full_parser
    _add_benchmark_args(full_parser)
    full_parser.add_argument(
        "--gen-markdown",
        help="Also generate a Markdown report.",
        action="store_true",
    )
    full_parser.add_argument(
        "--no-pdf",
        help="Don't generate PDF report",
        action="store_true",
    )
    full_parser.add_argument(
        "--out-dir",
        help="Output directory for reports. Defaults to the directory containing the result JSON.",
        required=False,
        default=None,
    )
    full_parser.add_argument(
        "--keep-json",
        help="Keep the intermediate JSON file instead of deleting it.",
        action="store_true",
    )

    parser.epilog = "\n".join(
        [
            "Subcommand usage:",
            command_parsers["benchmark"].format_usage().strip(),
            command_parsers["print"].format_usage().strip(),
            command_parsers["full"].format_usage().strip(),
        ]
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "benchmark":
        return _cmd_benchmark(args)
    elif args.command == "print":
        return _cmd_print(args)
    elif args.command == "full":
        return _cmd_full(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
