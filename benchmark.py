import sys
import os
import gc
import json
from collections import defaultdict
from typing import Any, Callable
import matplotlib.pyplot as plt
import shutil
import time
import platform
import re
import pathlib

import orjson
import pyyjson

CUR_FILE = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(CUR_FILE)
_NS_IN_ONE_S = 1000000000


class ObjectDict(dict):
    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        return self.get(name)


class LibrarySetting(ObjectDict):
    function_catagory: str
    orjson_function: Callable
    pyyjson_function: Callable
    orjson_function_name: str
    pyyjson_function_name: str


def benchmark(repeat_time: int, func, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    # warm up
    pyyjson.run_object_accumulate_benchmark(func, 100, args)
    return pyyjson.run_object_accumulate_benchmark(func, repeat_time, args)


def benchmark_unicode_arg(repeat_time: int, func, unicode: str, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    # warm up
    pyyjson.run_unicode_accumulate_benchmark(func, 100, unicode, args)
    return pyyjson.run_unicode_accumulate_benchmark(func, repeat_time, unicode, args)


def benchmark_use_dump_cache(repeat_time: int, func, raw_bytes: bytes, *args):
    """
    let orjson use utf-8 cache for the same input.
    returns time used (ns).
    """
    new_args = (json.loads(raw_bytes), *args)
    # warm up
    for _ in range(100):
        pyyjson.run_object_benchmark(func, new_args)
    #
    total = 0
    for _ in range(repeat_time):
        total += pyyjson.run_object_benchmark(func, new_args)
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
        pyyjson.run_object_benchmark(func, new_args)
    #
    total = 0
    for _ in range(repeat_time):
        new_args = (json.loads(raw_bytes), *args)
        total += pyyjson.run_object_benchmark(func, new_args)
    return total


def get_benchmark_libraries():
    dumps_setting_to_str = LibrarySetting()
    dumps_setting_to_str.function_catagory = "dumps"
    dumps_setting_to_str.orjson_function = lambda x: orjson.dumps(x).decode("utf-8")
    dumps_setting_to_str.pyyjson_function = pyyjson.dumps
    dumps_setting_to_str.orjson_function_name = "orjson.dumps+decode"
    dumps_setting_to_str.pyyjson_function_name = "pyyjson.dumps"

    dumps_setting_to_bytes = LibrarySetting()
    dumps_setting_to_bytes.function_catagory = "dumps_to_bytes"
    dumps_setting_to_bytes.orjson_function = orjson.dumps
    dumps_setting_to_bytes.pyyjson_function = pyyjson.dumps_to_bytes
    dumps_setting_to_bytes.orjson_function_name = "orjson.dumps"
    dumps_setting_to_bytes.pyyjson_function_name = "pyyjson.dumps_to_bytes"

    loads_setting_str = LibrarySetting()
    loads_setting_str.function_catagory = "loads(str)"
    loads_setting_str.orjson_function = orjson.loads
    loads_setting_str.pyyjson_function = pyyjson.loads
    loads_setting_str.orjson_function_name = "orjson.loads"
    loads_setting_str.pyyjson_function_name = "pyyjson.loads"

    loads_setting_bytes = LibrarySetting()
    loads_setting_bytes.function_catagory = "loads(bytes)"
    loads_setting_bytes.orjson_function = orjson.loads
    loads_setting_bytes.pyyjson_function = pyyjson.loads
    loads_setting_bytes.orjson_function_name = "orjson.loads"
    loads_setting_bytes.pyyjson_function_name = "pyyjson.loads"

    return (
        dumps_setting_to_str,
        dumps_setting_to_bytes,
        loads_setting_str,
        loads_setting_bytes,
    )


def get_benchmark_files() -> list[pathlib.Path]:
    return pathlib.Path(CUR_DIR, "_files").glob("*.json")


def run_dumps_to_str_benchmark(
    curfile_obj: defaultdict[str, Any],
    dumps_setting_to_str: LibrarySetting,
    repeat_times: int,
    raw_bytes_json_encoded: bytes,
):
    cur_obj = curfile_obj[dumps_setting_to_str.function_catagory]
    gc.collect()
    cur_obj[dumps_setting_to_str.orjson_function_name] = orjson_time = (
        benchmark_invalidate_dump_cache(
            repeat_times, dumps_setting_to_str.orjson_function, raw_bytes_json_encoded
        )
    )
    gc.collect()
    cur_obj[dumps_setting_to_str.pyyjson_function_name] = pyyjson_time = (
        benchmark_invalidate_dump_cache(
            repeat_times, dumps_setting_to_str.pyyjson_function, raw_bytes_json_encoded
        )
    )
    _, str_size_dumped, _, _ = pyyjson.inspect_pyunicode(
        pyyjson.dumps(json.loads(raw_bytes_json_encoded))
    )
    cur_obj["ratio"] = pyyjson_time / orjson_time
    cur_obj["pyyjson_bytes_per_sec"] = pyyjson.dumps(
        str_size_dumped * repeat_times / (pyyjson_time / _NS_IN_ONE_S)
    )


def run_dumps_to_bytes_benchmark(
    curfile_obj: defaultdict[str, Any],
    dumps_setting_to_bytes: LibrarySetting,
    repeat_times: int,
    raw_bytes_json_encoded: bytes,
):
    cur_obj = curfile_obj[dumps_setting_to_bytes.function_catagory]
    gc.collect()
    cur_obj[dumps_setting_to_bytes.orjson_function_name] = orjson_time = (
        benchmark_invalidate_dump_cache(
            repeat_times, dumps_setting_to_bytes.orjson_function, raw_bytes_json_encoded
        )
    )
    gc.collect()
    cur_obj[dumps_setting_to_bytes.pyyjson_function_name] = pyyjson_time = (
        benchmark_invalidate_dump_cache(
            repeat_times,
            dumps_setting_to_bytes.pyyjson_function,
            raw_bytes_json_encoded,
        )
    )
    bytes_size_dumped = len(pyyjson.dumps_to_bytes(json.loads(raw_bytes_json_encoded)))
    cur_obj["ratio"] = pyyjson_time / orjson_time
    cur_obj["pyyjson_bytes_per_sec"] = pyyjson.dumps(
        bytes_size_dumped * repeat_times / (pyyjson_time / _NS_IN_ONE_S)
    )


def run_loads_str_benchmark(
    curfile_obj: defaultdict[str, Any],
    loads_setting_str: LibrarySetting,
    repeat_times: int,
    raw_str_json_encoded: str,
):
    cur_obj = curfile_obj[loads_setting_str.function_catagory]
    gc.collect()
    cur_obj[loads_setting_str.orjson_function_name] = orjson_time = (
        benchmark_unicode_arg(
            repeat_times, loads_setting_str.orjson_function, raw_str_json_encoded
        )
    )
    gc.collect()
    cur_obj[loads_setting_str.pyyjson_function_name] = pyyjson_time = (
        benchmark_unicode_arg(
            repeat_times, loads_setting_str.pyyjson_function, raw_str_json_encoded
        )
    )
    _, str_size, _, _ = pyyjson.inspect_pyunicode(raw_str_json_encoded)
    cur_obj["ratio"] = pyyjson_time / orjson_time
    cur_obj["pyyjson_bytes_per_sec"] = pyyjson.dumps(
        str_size * repeat_times / (pyyjson_time / _NS_IN_ONE_S)
    )


def run_loads_bytes_benchmark(
    curfile_obj: defaultdict[str, Any],
    loads_setting_bytes: LibrarySetting,
    repeat_times: int,
    raw_bytes_json_encoded: bytes,
):
    cur_obj = curfile_obj[loads_setting_bytes.function_catagory]
    gc.collect()
    cur_obj[loads_setting_bytes.orjson_function_name] = orjson_time = benchmark(
        repeat_times, loads_setting_bytes.orjson_function, raw_bytes_json_encoded
    )
    gc.collect()
    cur_obj[loads_setting_bytes.pyyjson_function_name] = pyyjson_time = benchmark(
        repeat_times, loads_setting_bytes.pyyjson_function, raw_bytes_json_encoded
    )
    cur_obj["ratio"] = pyyjson_time / orjson_time
    cur_obj["pyyjson_bytes_per_sec"] = pyyjson.dumps(
        len(raw_bytes_json_encoded) * repeat_times / (pyyjson_time / _NS_IN_ONE_S)
    )


def run_file_benchmark(
    file: str, result: defaultdict[str, defaultdict[str, Any]], process_bytes: int
):
    # if not file.endswith("apache.json"):
    #     return
    (
        dumps_setting_to_str,
        dumps_setting_to_bytes,
        loads_setting_str,
        loads_setting_bytes,
    ) = get_benchmark_libraries()
    with open(file, "rb") as f:
        raw_bytes = f.read()
    raw = raw_bytes.decode("utf-8")
    base_file_name = os.path.basename(file)
    curfile_obj = result[base_file_name]
    curfile_obj["byte_size"] = bytes_size = len(raw_bytes)
    kind, str_size, is_ascii, _ = pyyjson.inspect_pyunicode(raw)
    curfile_obj["pyunicode_size"] = str_size
    curfile_obj["pyunicode_kind"] = kind
    curfile_obj["pyunicode_is_ascii"] = is_ascii
    repeat_times = (process_bytes + bytes_size - 1) // bytes_size
    # dumps (to str)
    run_dumps_to_str_benchmark(
        curfile_obj, dumps_setting_to_str, repeat_times, raw_bytes
    )
    # dumps (to bytes)
    run_dumps_to_bytes_benchmark(
        curfile_obj, dumps_setting_to_bytes, repeat_times, raw_bytes
    )
    # loads (str)
    run_loads_str_benchmark(curfile_obj, loads_setting_str, repeat_times, raw)
    # loads (bytes)
    run_loads_bytes_benchmark(curfile_obj, loads_setting_bytes, repeat_times, raw_bytes)


def get_head_rev_name():
    return pyyjson.__version__


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


def generate_report(result: dict[str, dict[str, Any]], output: str, file: str):
    file = file.removesuffix(".json")
    report_folder = f"{file}_report"

    if os.path.exists(report_folder):
        shutil.rmtree(report_folder)
    os.mkdir(report_folder)

    categories = ["dumps", "dumps_to_bytes", "loads(str)", "loads(bytes)"]
    files_report = ""

    for bench_file in get_benchmark_files():
        # Create a 1x4 horizontal layout
        fig, axs = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
        fig.suptitle(f"Benchmark: {bench_file.name}", fontsize=14)

        for idx, category in enumerate(categories):
            curfile_obj = result[bench_file.name][category]
            keys = list(curfile_obj.keys())

            orjson_time = curfile_obj[[k for k in keys if "orjson" in k][0]]
            pyyjson_time = curfile_obj[[k for k in keys if "pyyjson" in k][0]]
            ratio = curfile_obj["ratio"]

            ax = axs[idx]
            bars = ax.bar(
                ["orjson", "pyyjson"],
                [orjson_time, pyyjson_time],
                color=["#6baed6", "#fd8d3c"],  # 更柔和的蓝橙
            )

            # Add bar labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.01,
                    f"{height:.2e}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Padding above bars
            ymax = max(orjson_time, pyyjson_time)
            ax.set_ylim(0, ymax * 1.15)
            ax.plot([0], [1], "^k", transform=ax.transAxes, clip_on=False)

            # Clean visual style: remove axis spines and ticks
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.tick_params(left=False, bottom=False)

            ax.set_title(category, fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["orjson", "pyyjson"])
            if idx == 0:
                ax.set_ylabel("Speed (ns)")

            ax.annotate(
                f"ratio={ratio:.2f}",
                xy=(1.0, 1.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=9,
                style="italic",
            )

        plt.tight_layout(rect=[0, 0.05, 1, 0.9], pad=2.0)

        fig.text(
            0.5,
            0.01,
            "Lower is better ↓",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            color="#555555",
        )
        plt.savefig(os.path.join(report_folder, f"{bench_file.name}.svg"))
        plt.close()
        files_report += f"![{bench_file.name}.svg](./{bench_file.name}.svg)  \n"
    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    template = template.format(
        REV=get_head_rev_name(),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        OS=f"{platform.system()} {platform.machine()}",
        PYTHON=sys.version,
        SIMD_FLAGS=pyyjson.get_current_features(),
        CHIPSET=get_cpu_name(),
        FILES=files_report,
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
        default=100000000,
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
