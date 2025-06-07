import sys
import os
import gc
import json
from collections import defaultdict
from typing import Any, Callable
from matplotlib import ticker
import matplotlib.pyplot as plt
import shutil
import time
import platform
import re
import pathlib
import psutil
import math

import orjson
import ssrjson

CUR_FILE = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(CUR_FILE)
_NS_IN_ONE_S = 1000000000

LIBRARIES = {
    "dumps": {
        "orjson.dumps+decode": lambda x: orjson.dumps(x).decode("utf-8"),
        "ssrjson.dumps": ssrjson.dumps,
    },
    "dumps_to_bytes": {
        "orjson.dumps": orjson.dumps,
        "ssrjson.dumps_to_bytes": ssrjson.dumps_to_bytes,
    },
    "loads(str)": {
        "orjson.loads": orjson.loads,
        "ssrjson.loads": ssrjson.loads,
    },
    "loads(bytes)": {
        "orjson.loads": orjson.loads,
        "ssrjson.loads": ssrjson.loads,
    },
}

INDEXES = ["elapsed", "user_cpu"]


def benchmark(repeat_time: int, func, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    # warm up
    ssrjson.run_object_accumulate_benchmark(func, 100, args)
    return ssrjson.run_object_accumulate_benchmark(func, repeat_time, args)


def benchmark_unicode_arg(repeat_time: int, func, unicode: str, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    # warm up
    ssrjson.run_unicode_accumulate_benchmark(func, 100, unicode, args)
    return ssrjson.run_unicode_accumulate_benchmark(func, repeat_time, unicode, args)


def benchmark_use_dump_cache(repeat_time: int, func, raw_bytes: bytes, *args):
    """
    let orjson use utf-8 cache for the same input.
    returns time used (ns).
    """
    new_args = (json.loads(raw_bytes), *args)
    # warm up
    for _ in range(100):
        ssrjson.run_object_benchmark(func, new_args)
    #
    total = 0
    for _ in range(repeat_time):
        total += ssrjson.run_object_benchmark(func, new_args)
    return total


def benchmark_invalidate_dump_cache(repeat_time: int, func, raw_bytes: bytes, *args):
    """
    orjson will use utf-8 cache for the same input,
    so we need to invalidate it.
    returns time used (ns).
    """
    # warm up
    for _ in range(10):
        new_args = (json.loads(raw_bytes), *args)
        ssrjson.run_object_benchmark(func, new_args)
    #
    total = 0
    for _ in range(repeat_time):
        new_args = (json.loads(raw_bytes), *args)
        total += ssrjson.run_object_benchmark(func, new_args)
    return total


def get_benchmark_files() -> list[pathlib.Path]:
    return pathlib.Path(CUR_DIR, "_files").glob("*.json")


def _run_benchmark(
    curfile_obj: defaultdict[str, Any],
    repeat_times: int,
    input_data: str | bytes,
    mode: str,  # "dumps", "dumps_to_bytes", "loads(str)", "loads(bytes)"
):
    print(f"Running benchmark for {mode}")
    funcs = LIBRARIES[mode]
    cur_obj = curfile_obj[mode]

    def pick_benchmark_func() -> Callable:
        if "dumps" in mode and "loads" not in mode:
            return benchmark_invalidate_dump_cache
        if isinstance(input_data, str) and "loads" in mode:
            return benchmark_unicode_arg
        return benchmark

    process = psutil.Process()

    for name, func in funcs.items():
        benchmark_func = pick_benchmark_func()
        gc.collect()
        t0 = time.perf_counter()
        cpu_times_before = process.cpu_times()
        ctx_before = process.num_ctx_switches()
        mem_before = process.memory_info().rss

        elapsed = benchmark_func(repeat_times, func, input_data)

        # End measuring
        t1 = time.perf_counter()
        cpu_times_after = process.cpu_times()
        ctx_after = process.num_ctx_switches()

        user_cpu = cpu_times_after.user - cpu_times_before.user
        system_cpu = cpu_times_after.system - cpu_times_before.system
        voluntary_ctx = ctx_after.voluntary - ctx_before.voluntary
        involuntary_ctx = ctx_after.involuntary - ctx_before.involuntary
        mem_after = process.memory_info().rss

        cur_obj[name] = {
            "elapsed": elapsed,
            "user_cpu": user_cpu,
            "system_cpu": system_cpu,
            "ctx_vol": voluntary_ctx,
            "ctx_invol": involuntary_ctx,
            "mem_diff": mem_after - mem_before,
            "wall_time": t1 - t0,
        }

    ssrjson_name = next(k for k in funcs if k.startswith("ssrjson"))
    ssrjson_func = funcs[ssrjson_name]
    if "dumps" in mode:
        data_obj = json.loads(input_data)
        output = ssrjson_func(data_obj)
        if "bytes" in mode:
            size = len(output)
        else:
            _, size, _, _ = ssrjson.inspect_pyunicode(output)
    else:
        size = (
            len(input_data)
            if isinstance(input_data, bytes)
            else ssrjson.inspect_pyunicode(input_data)[1]
        )

    orjson_name = next(k for k in funcs if k.startswith("orjson"))
    orjson_time = cur_obj[orjson_name]
    ssrjson_time = cur_obj[ssrjson_name]

    for index in INDEXES:
        if orjson_time[index] == 0:
            cur_obj[f"{index}_ratio"] = math.inf
        else:
            cur_obj[f"{index}_ratio"] = ssrjson_time[index] / orjson_time[index]
    cur_obj["ssrjson_bytes_per_sec"] = ssrjson.dumps(
        size * repeat_times / (ssrjson_time["elapsed"] / _NS_IN_ONE_S)
    )


def run_file_benchmark(
    file: str, result: defaultdict[str, defaultdict[str, Any]], process_bytes: int
):
    with open(file, "rb") as f:
        raw_bytes = f.read()
    raw = raw_bytes.decode("utf-8")
    base_file_name = os.path.basename(file)
    curfile_obj = result[base_file_name]
    curfile_obj["byte_size"] = bytes_size = len(raw_bytes)
    kind, str_size, is_ascii, _ = ssrjson.inspect_pyunicode(raw)
    curfile_obj["pyunicode_size"] = str_size
    curfile_obj["pyunicode_kind"] = kind
    curfile_obj["pyunicode_is_ascii"] = is_ascii
    repeat_times = (process_bytes + bytes_size - 1) // bytes_size

    for mode in LIBRARIES.keys():
        _run_benchmark(curfile_obj, repeat_times, raw_bytes, mode)


def get_head_rev_name():
    return ssrjson.__version__


def get_real_output_file_name(output: str):
    if output:
        file = output
    else:
        rev = get_head_rev_name()
        if not rev:
            file = "benchmark_result.json"
        else:
            file = f"benchmark_result_{rev}.json"
    return file


def get_cpu_name() -> str:
    cpu_name: str = platform.processor()
    if not cpu_name or cpu_name == "":
        with open(file="/proc/cpuinfo", mode="r") as file:
            cpu_info_lines = file.readlines()
            for line in cpu_info_lines:
                if "model name" in line:
                    cpu_name = re.sub(
                        pattern=r"model name\s+:\s+", repl="", string=line
                    )
                    # remove extra spaces
                    cpu_name = re.sub(pattern=r"\s+", repl=" ", string=cpu_name).strip()
                    break

    return cpu_name


def get_mem_total() -> str:
    mem_total: int = 0
    if platform.system() == "Linux":
        with open(file="/proc/meminfo", mode="r") as file:
            mem_info_lines = file.readlines()
            for line in mem_info_lines:
                if "MemTotal" in line:
                    mem_total = int(re.sub(pattern=r"[^0-9]", repl="", string=line))
                    break
    elif platform.system() == "Windows":
        import psutil

        mem_total = psutil.virtual_memory().total // (1024 * 1024)
    return f"{mem_total / (1024 ** 2):.3f}GiB"


def get_ratio_color(ratio: float) -> str:
    if ratio <= 0.6:
        return "#d63031"  # deep coral red
    elif ratio <= 0.8:
        return "#e67e22"  # warm orange
    elif ratio == 1:
        return "black"
    elif ratio <= 1.2:
        return "#f1c40f"  # strong yellow
    elif ratio <= 1.5:
        return "#27ae60"  # green
    else:
        return "#2980b9"  # blue (rare)


def plot_relative_ops(data: dict, doc_name: str, index_s: str, output_path: str):
    categories = ["dumps", "dumps_to_bytes", "loads(str)", "loads(bytes)"]
    libraries = ["orjson", "ssrjson"]
    colors = {"orjson": "#6baed6", "ssrjson": "#fd8d3c"}

    total_groups = len(categories)
    bar_width = 0.35
    x = list(range(total_groups))

    # Prepare grouped values
    orjson_vals = [1.0 for _ in categories]
    ssrjson_vals = [1 / data[cat][f"{index_s}_ratio"] for cat in categories]
    values = [orjson_vals, ssrjson_vals]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = []
    for i in range(len(libraries)):
        bars.append(
            ax.bar(
                [j + i * bar_width for j in x],
                values[i],
                width=bar_width,
                label=libraries[i],
                color=colors[libraries[i]],
            )
        )

    # Annotate bars
    for bars, vals in zip(bars, values):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.05,
                f"{val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=7,
                color=get_ratio_color(val),
            )

    # Formatting
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, max(ssrjson_vals + [1.0]) * 1.25)
    ax.set_ylabel("ratio", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))
    fig.text(
        0.5,
        0.01,
        "Higher is better",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(doc_name, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return f"Plot saved to: {output_path}"


def generate_report(result: dict[str, dict[str, Any]], output: str, file: str):
    file = file.removesuffix(".json")
    report_folder = f"{file}_report"

    if os.path.exists(report_folder):
        shutil.rmtree(report_folder)
    os.mkdir(report_folder)

    files_reports = ""

    for index_s in INDEXES:
        files_reports += f"### {index_s}  \n"
        for bench_file in get_benchmark_files():
            print(f"Processing {bench_file.name}")
            plot_relative_ops(
                result[bench_file.name],
                bench_file.name,
                index_s,
                os.path.join(report_folder, f"{bench_file.name}_{index_s}.svg"),
            )
            files_reports += f"![{bench_file.name}_{index_s}.svg](./{bench_file.name}_{index_s}.svg)  \n"

    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    template = template.format(
        REV=get_head_rev_name(),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        OS=f"{platform.system()} {platform.machine()}",
        PYTHON=sys.version,
        SIMD_FLAGS=ssrjson.get_current_features(),
        CHIPSET=get_cpu_name(),
        FILES=files_reports,
        MEM=get_mem_total(),
    )
    with open(os.path.join(report_folder, "README.md"), "w") as f:
        f.write(template)


def run_benchmark(process_bytes: int, output: str, is_stdout: bool = False):
    file = get_real_output_file_name(output)
    if os.path.exists(file):
        os.remove(file)
    result: defaultdict[str, defaultdict[str, Any]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for bench_file in get_benchmark_files():
        run_file_benchmark(bench_file, result, process_bytes)
    output_result = json.dumps(result, indent=4)

    with open(f"{file}", "w", encoding="utf-8") as f:
        f.write(output_result)
    if is_stdout:
        print(output_result)
    generate_report(result, output, file)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--file", help="record JSON file", required=False, default=None
    )
    parser.add_argument(
        "-o", "--output", help="Output file", required=False, default=None
    )
    parser.add_argument(
        "--process-bytes",
        help="Total process bytes per test, default 1e8",
        required=False,
        default=100050000,
        type=int,
    )
    parser.add_argument(
        "--stdout", help="Print to stdout", required=False, action="store_true"
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            j = json.load(f)
        generate_report(j, args.output, args.file.split("/")[-1])
    else:
        run_benchmark(args.process_bytes, args.output, args.stdout)


if __name__ == "__main__":
    main()
