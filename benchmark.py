import io
import sys
import os
import gc
import json
from collections import defaultdict
from typing import Any, Callable
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from svglib.svglib import svg2rlg
from svglib.fonts import FontMap
from reportlab.graphics import renderPDF
from typing import List
import io
import time
import platform
import re
import pathlib

# import psutil
import math
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"

import orjson
import ssrjson

font_map = FontMap()
font_map.register_default_fonts()

CUR_FILE = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(CUR_FILE)
_NS_IN_ONE_S = 1000000000

PDF_PAGE_SIZE = A4
PDF_HEADING_FONT = "Helvetica-Bold"
# workaround for matplotlib using 700 to represent bold font, but svg2rlg using 700 as normal.
font_map.register_font("Helvetica", weight="700", rlgFontName="Helvetica-Bold")
PDF_TEXT_FONT = "Courier"

# baseline is the first one.
LIBRARIES_COLORS = {"json": "#74c476", "orjson": "#6baed6", "ssrjson": "#fd8d3c"}
LIBRARIES: dict[str, dict[str, Callable[[str | bytes], Any]]] = {
    "dumps": {
        "json.dumps": json.dumps,
        "orjson.dumps+decode": lambda x: orjson.dumps(x).decode("utf-8"),
        "ssrjson.dumps": ssrjson.dumps,
    },
    "dumps(indented2)": {
        "json.dumps": lambda x: json.dumps(x, indent=2),
        "orjson.dumps+decode": lambda x: orjson.dumps(
            x, option=orjson.OPT_INDENT_2
        ).decode("utf-8"),
        "ssrjson.dumps": lambda x: ssrjson.dumps(x, indent=2),
    },
    "dumps_to_bytes": {
        "json.dumps+encode": lambda x: json.dumps(x).encode("utf-8"),
        "orjson.dumps": orjson.dumps,
        "ssrjson.dumps_to_bytes": ssrjson.dumps_to_bytes,
    },
    "dumps_to_bytes(indented2)": {
        "json.dumps+encode": lambda x: json.dumps(x).encode("utf-8"),
        "orjson.dumps": orjson.dumps,
        "ssrjson.dumps_to_bytes": ssrjson.dumps_to_bytes,
    },
    "loads(str)": {
        "json.loads": json.loads,
        "orjson.loads": orjson.loads,
        "ssrjson.loads": ssrjson.loads,
    },
    "loads(bytes)": {
        "json.loads": json.loads,
        "orjson.loads": orjson.loads,
        "ssrjson.loads": ssrjson.loads,
    },
}
CATEGORIES = LIBRARIES.keys()

INDEXES = ["elapsed"]


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
    data_warmup = [json.loads(raw_bytes) for _ in range(10)]
    data = [json.loads(raw_bytes) for _ in range(repeat_time)]
    # warm up
    for i in range(10):
        new_args = (data_warmup[i], *args)
        ssrjson.run_object_benchmark(func, new_args)
    #
    total = 0
    for i in range(repeat_time):
        new_args = (data[i], *args)
        total += ssrjson.run_object_benchmark(func, new_args)
    return total


def get_benchmark_files() -> list[pathlib.Path]:
    return sorted(pathlib.Path(CUR_DIR, "_files").glob("*.json"))


def _run_benchmark(
    curfile_obj: defaultdict[str, Any],
    repeat_times: int,
    input_data: str | bytes,
    mode: str,  # "dumps", etc
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

    # process = psutil.Process()

    for name, func in funcs.items():
        benchmark_func = pick_benchmark_func()
        gc.collect()
        t0 = time.perf_counter()
        # cpu_times_before = process.cpu_times()
        # ctx_before = process.num_ctx_switches()
        # mem_before = process.memory_info().rss

        elapsed = benchmark_func(repeat_times, func, input_data)

        # End measuring
        t1 = time.perf_counter()
        # cpu_times_after = process.cpu_times()
        # ctx_after = process.num_ctx_switches()

        # user_cpu = cpu_times_after.user - cpu_times_before.user
        # system_cpu = cpu_times_after.system - cpu_times_before.system
        # voluntary_ctx = ctx_after.voluntary - ctx_before.voluntary
        # involuntary_ctx = ctx_after.involuntary - ctx_before.involuntary
        # mem_after = process.memory_info().rss

        cur_obj[name] = {
            "elapsed": elapsed,
            # "user_cpu": user_cpu,
            # "system_cpu": system_cpu,
            # "ctx_vol": voluntary_ctx,
            # "ctx_invol": involuntary_ctx,
            # "mem_diff": mem_after - mem_before,
            "wall_time": t1 - t0,
        }

    funcs_iter = iter(funcs.items())
    baseline_name, _ = next(funcs_iter)
    baseline_data = cur_obj[baseline_name]
    for name, func in funcs_iter:
        if name.startswith("ssrjson"):
            # debug use, bytes per sec
            if "dumps" in mode:
                data_obj = json.loads(input_data)
                output = func(data_obj)
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
            cur_obj["ssrjson_bytes_per_sec"] = ssrjson.dumps(
                size * repeat_times / (cur_obj[name]["elapsed"] / _NS_IN_ONE_S)
            )
        for index in INDEXES:
            basename = name.split(".")[0]
            if baseline_data[index] == 0:
                cur_obj[f"{basename}_{index}_ratio"] = math.inf
            else:
                cur_obj[f"{basename}_{index}_ratio"] = (
                    baseline_data[index] / cur_obj[name][index]
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


def get_real_output_file_name():
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
    if ratio < 1:
        return "#d63031"  # red (worse than baseline)
    elif ratio == 1:
        return "black"  # black (baseline)
    elif ratio < 2:
        return "#e67e22"  # orange (similar/slightly better)
    elif ratio < 4:
        return "#f39c12"  # amber (decent improvement)
    elif ratio < 8:
        return "#27ae60"  # green (good)
    elif ratio < 16:
        return "#2980b9"  # blue (great)
    else:
        return "#8e44ad"  # purple (exceptional)


def plot_relative_ops(data: dict, doc_name: str, index_s: str) -> io.BytesIO:
    libs = list(LIBRARIES_COLORS.keys())
    colors = [LIBRARIES_COLORS[n] for n in libs]
    n = len(CATEGORIES)
    bar_width = 0.2
    inner_pad = 0

    fig, axs = plt.subplots(
        1,
        n,
        figsize=(4 * n, 6),
        sharey=False,
        tight_layout=True,
        gridspec_kw={"wspace": 0},
    )

    x_positions = [i * (bar_width + inner_pad) for i in range(len(libs))]

    for ax, cat in zip(axs, CATEGORIES):
        vals = [1.0] + [data[cat][f"{name}_{index_s}_ratio"] for name in libs[1:]]

        for xi, val, col in zip(x_positions, vals, colors):
            ax.bar(xi, val, width=bar_width, color=col)
            ax.text(
                xi,
                val + 0.05,
                f"{val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color=get_ratio_color(val),
            )

        # baseline line
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        # height = 1.1 * max bar height
        ax.set_ylim(0, max(vals + [1.0]) * 1.1)

        # hide all tick
        ax.tick_params(
            axis="both",
            which="both",
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

        # and spine
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)

        ax.set_xlabel(cat, fontsize=10, labelpad=6)

    fig.suptitle(
        doc_name,
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # color legend
    legend_elements = [
        plt.Line2D([0], [0], color=col, lw=4, label=name)
        for name, col in LIBRARIES_COLORS.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.95),
        ncol=len(libs),
        fontsize=14,
        frameon=False,
    )

    fig.text(
        0.5,
        0,
        "Higher is better",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_pdf_report(
    figures: List[List[io.BytesIO]], header_text: str, output_pdf_path: str
) -> str:
    c = canvas.Canvas(output_pdf_path, pagesize=PDF_PAGE_SIZE)
    width, height = PDF_PAGE_SIZE

    # heading info
    c.setFont(PDF_HEADING_FONT, 16)
    text_obj = c.beginText(40, height - 50)
    text_obj.setFont(PDF_TEXT_FONT, 10)
    for line in header_text.splitlines():
        text_obj.textLine(line)
    c.drawText(text_obj)

    header_lines = header_text.count("\n") + 1
    header_height = header_lines * 14 + 10
    # subheading spacing = 30
    y_pos = height - header_height - 30
    bottom_margin = 20
    vertical_gap = 20

    for name, figs in zip(INDEXES, figures):
        text_obj = c.beginText()
        text_obj.setTextOrigin(40, y_pos)
        text_obj.setFont(PDF_HEADING_FONT, 14)
        text_obj.textLine(f"{name}")
        c.drawText(text_obj)
        c.bookmarkHorizontal(name, 0, y_pos + 20)
        c.addOutlineEntry(name, name, level=0)
        y_pos -= 20
        for svg_io in figs:
            svg_io.seek(0)
            drawing = svg2rlg(svg_io, font_map=font_map)

            avail_w = width - 80
            scale = avail_w / drawing.width
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)

            img_h = drawing.height
            # no enough space
            if y_pos - img_h - vertical_gap < bottom_margin:
                c.showPage()
                y_pos = height - bottom_margin

            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.setLineWidth(0.4)
            c.line(40, y_pos, width - 40, y_pos)

            renderPDF.draw(drawing, c, 40, y_pos - img_h)
            y_pos -= img_h + vertical_gap

    c.save()
    return output_pdf_path


def generate_report(result: dict[str, dict[str, Any]], file: str):
    file = file.removesuffix(".json")
    report_name = f"{file}_report.pdf"

    figures = []

    for index_s in INDEXES:
        tmp = []
        for bench_file in get_benchmark_files():
            print(f"Processing {bench_file.name}")
            tmp.append(
                plot_relative_ops(
                    result[bench_file.name],
                    bench_file.name,
                    index_s,
                )
            )
        figures.append(tmp)

    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    template = template.format(
        REV=file.removeprefix("benchmark_result_").removesuffix(".json"),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        OS=f"{platform.system()} {platform.machine()}",
        PYTHON=sys.version,
        ORJSON_VER=orjson.__version__,
        SIMD_FLAGS=ssrjson.get_current_features(),
        CHIPSET=get_cpu_name(),
        MEM=get_mem_total(),
    )
    generate_pdf_report(
        figures,
        header_text=template,
        output_pdf_path=os.path.join(CUR_DIR, report_name),
    )


def generate_report_markdown(result: dict[str, dict[str, Any]], file: str):
    file = file.removesuffix(".json")
    report_name = f"{file}_report.md"
    report_folder = os.path.join(CUR_DIR, f"{file}_report")

    # mkdir
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    template = template.format(
        REV=file.removeprefix("benchmark_result_").removesuffix(".json"),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        OS=f"{platform.system()} {platform.machine()}",
        PYTHON=sys.version,
        ORJSON_VER=orjson.__version__,
        SIMD_FLAGS=ssrjson.get_current_features(),
        CHIPSET=get_cpu_name(),
        MEM=get_mem_total(),
    )

    for index_s in INDEXES:
        template += f"\n\n## {index_s}\n\n"
        for bench_file in get_benchmark_files():
            print(f"Processing {bench_file.name}")
            with open(
                os.path.join(report_folder, bench_file.name + ".svg"), "wb"
            ) as svg_file:
                svg_file.write(
                    plot_relative_ops(
                        result[bench_file.name],
                        bench_file.name,
                        index_s,
                    ).getvalue()
                )
            # add svg
            template += f"![{bench_file.name}](./{bench_file.name}.svg)\n\n"

    with open(os.path.join(report_folder, report_name), "w") as f:
        f.write(template)


def run_benchmark(process_bytes: int):
    file = get_real_output_file_name()
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
    return result, file


def main():
    import argparse

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
        default=100050000,
        type=int,
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
        generate_report_markdown(j, file)
    else:
        generate_report(j, file)


if __name__ == "__main__":
    main()
