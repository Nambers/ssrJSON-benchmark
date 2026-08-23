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
    from .benchmark import (
        DEFAULT_COLD_MULTIPLE,
        DEFAULT_MIN_ITERATIONS,
        DEFAULT_ROUNDS,
        DEFAULT_STATISTIC,
    )

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
        help="Deprecated and ignored. Object temperature is now controlled by "
        "--locality and --cold-working-set-multiple instead of a memory knob.",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--min-iterations",
        help="Floor on measured iterations per test case, default 200 (int). "
        "The byte budget alone leaves large files with too few samples.",
        required=False,
        default=DEFAULT_MIN_ITERATIONS,
        type=int,
    )
    parser.add_argument(
        "--locality",
        choices=["hot", "cold", "both"],
        help="Whether the measured object is in cache when the call starts. "
        "'hot' keeps one live copy; 'cold' keeps a ring larger than the last "
        "level cache. Default 'both'.",
        required=False,
        default="both",
    )
    parser.add_argument(
        "--cold-working-set-multiple",
        help="Cold ring size as a multiple of the last level cache, default 2.0 (float).",
        required=False,
        default=DEFAULT_COLD_MULTIPLE,
        type=float,
    )
    parser.add_argument(
        "--llc-bytes",
        help="Override the detected last level cache size, in bytes. Use when "
        "auto-detection reports 'fallback'.",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--statistic",
        choices=["median", "min", "mean"],
        help="Summary statistic per test case, default median. mean is skewed "
        "by interrupts and can disagree with min about which library wins; all "
        "of min/median/mean/p95 are recorded either way.",
        required=False,
        default=DEFAULT_STATISTIC,
    )
    parser.add_argument(
        "--rounds",
        help="Split each measurement into N interleaved chunks so drift hits "
        "every library equally instead of penalising whichever runs last. "
        "Default 5; use 1 for the old sequential behaviour.",
        required=False,
        default=DEFAULT_ROUNDS,
        type=int,
    )
    parser.add_argument(
        "--runs",
        help="Run the whole benchmark in N fresh processes and report the "
        "median across runs with the observed spread. This is what captures "
        "code-layout and process-level variance. Default 1.",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--pin-core",
        help="Pin to this CPU id. Default: auto-pick the first thread of a "
        "fastest-class core.",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "--no-pin",
        help="Do not pin to a CPU core. On hybrid CPUs this lets the process "
        "land on an efficiency core, which can invert the comparison.",
        action="store_true",
    )
    parser.add_argument(
        "--no-verify-output",
        help="Skip the round-trip check that each library produces the expected value.",
        action="store_true",
    )
    parser.add_argument(
        "--allow-output-mismatch",
        help="Record output mismatches and continue instead of aborting.",
        action="store_true",
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
    import os

    from .benchmark import Locality, RunOptions, run_benchmark

    benchmark_files = _resolve_benchmark_files(args.in_dir)
    if benchmark_files is None:
        return None

    if args.bin_process_megabytes is not None:
        print(
            "warning: --bin-process-megabytes is deprecated and ignored. It used to "
            "set the bin size, which silently controlled how cold the measured "
            "object was; use --locality and --cold-working-set-multiple instead."
        )

    process_bytes = int(args.process_gigabytes * 1024 * 1024 * 1024)
    if process_bytes <= 0:
        print("process-gigabytes must be positive.")
        return None
    if args.min_iterations <= 0:
        print("min-iterations must be positive.")
        return None
    if args.cold_working_set_multiple <= 0:
        print("cold-working-set-multiple must be positive.")
        return None

    if args.runs < 1:
        print("runs must be at least 1.")
        return None
    if args.rounds < 1:
        print("rounds must be at least 1.")
        return None

    localities = {
        "hot": [Locality.HOT],
        "cold": [Locality.COLD],
        "both": [Locality.HOT, Locality.COLD],
    }[args.locality]

    if args.runs > 1:
        return _run_repeated(args)

    result, default_file = run_benchmark(
        benchmark_files,
        process_bytes,
        only=args.only,
        localities=localities,
        min_iterations=args.min_iterations,
        cold_multiple=args.cold_working_set_multiple,
        llc_bytes=args.llc_bytes,
        opts=RunOptions(
            statistic=args.statistic,
            rounds=args.rounds,
            verify_output=not args.no_verify_output,
            allow_output_mismatch=args.allow_output_mismatch,
        ),
        pin_core=args.pin_core,
        pin=not args.no_pin,
    )
    return result, default_file


def _child_argv(args, out_path: str) -> list[str]:
    """Rebuild this benchmark invocation as a single-run child command."""
    argv = [
        "benchmark",
        "--process-gigabytes",
        str(args.process_gigabytes),
        "--min-iterations",
        str(args.min_iterations),
        "--locality",
        args.locality,
        "--cold-working-set-multiple",
        str(args.cold_working_set_multiple),
        "--statistic",
        args.statistic,
        "--rounds",
        str(args.rounds),
        "--runs",
        "1",
        "-o",
        out_path,
    ]
    if args.in_dir:
        argv += ["--in-dir", args.in_dir]
    if args.only:
        argv += ["--only", args.only]
    if args.llc_bytes:
        argv += ["--llc-bytes", str(args.llc_bytes)]
    if args.pin_core is not None:
        argv += ["--pin-core", str(args.pin_core)]
    if args.no_pin:
        argv.append("--no-pin")
    if args.no_verify_output:
        argv.append("--no-verify-output")
    if args.allow_output_mismatch:
        argv.append("--allow-output-mismatch")
    return argv


def _run_repeated(args):
    """Run the whole benchmark in N fresh processes and merge.

    A single process cannot see code/data layout effects: heap and binary
    layout are fixed for its lifetime, so repeating inside one process
    re-measures the same layout. Two libraries this close need the
    across-process distribution, not the within-process one.
    """
    import json as _json
    import os
    import subprocess
    import sys as _sys
    import tempfile

    from .benchmark import (
        merge_run_results,
        parse_file_result,
        _get_real_output_file_name,
    )

    results = []
    with tempfile.TemporaryDirectory(prefix="ssrjson_bench_runs_") as tmpdir:
        for run_index in range(args.runs):
            out_path = os.path.join(tmpdir, f"run{run_index}.json")
            print(f"=== run {run_index + 1}/{args.runs} (fresh process) ===")
            completed = subprocess.run(
                [
                    _sys.executable,
                    "-m",
                    "ssrjson_benchmark",
                    *_child_argv(args, out_path),
                ]
            )
            if completed.returncode != 0:
                print(
                    f"run {run_index + 1} failed with exit code {completed.returncode}"
                )
                return None
            with open(out_path, "rb") as f:
                results.append(parse_file_result(_json.load(f)))

    merged = merge_run_results(results)
    out_file = _get_real_output_file_name()
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(merged.dumps())
    return merged, out_file


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
