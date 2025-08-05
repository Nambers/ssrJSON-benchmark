import importlib
import io
import sys
import os
import gc
import json
from collections import defaultdict
from typing import Any, Callable
from typing import List
import io
import time
import platform
import re
import pathlib
import math
from ssrjson_benchmark import _ssrjson_benchmark
import matplotlib.pyplot as plt
import matplotlib as mpl

import orjson
import ssrjson


mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"


try:
    from svglib.fonts import FontMap

    font_map = FontMap()
    font_map.register_default_fonts()
    # workaround for matplotlib using 700 to represent bold font, but svg2rlg using 700 as normal.
    font_map.register_font("Helvetica", weight="700", rlgFontName="Helvetica-Bold")
except ImportError:
    pass

CUR_FILE = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(CUR_FILE)
CWD = os.getcwd()
_NS_IN_ONE_S = 1000000000

PDF_HEADING_FONT = "Helvetica-Bold"
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
        "json.dumps+encode": lambda x: json.dumps(x, indent=2).encode("utf-8"),
        "orjson.dumps": lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2),
        "ssrjson.dumps_to_bytes": lambda x: ssrjson.dumps_to_bytes(x, indent=2),
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

INDEXES = ["speed"]


def gc_prepare():
    """
    Call collect once, and then disable automatic GC.
    Return True if automatic GC was enabled.
    """
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    return gc_was_enabled


def benchmark(repeat_time: int, func, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    gc_was_enabled = gc_prepare()
    try:
        # warm up
        _ssrjson_benchmark.run_object_accumulate_benchmark(func, 100, args)
        return _ssrjson_benchmark.run_object_accumulate_benchmark(
            func, repeat_time, args
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def benchmark_unicode_arg(repeat_time: int, func, unicode: str, *args):
    """
    Run repeat benchmark, disabling orjson utf-8 cache.
    returns time used (ns).
    """
    gc_was_enabled = gc_prepare()
    try:
        # warm up
        _ssrjson_benchmark.run_unicode_accumulate_benchmark(func, 100, unicode, args)
        return _ssrjson_benchmark.run_unicode_accumulate_benchmark(
            func, repeat_time, unicode, args
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def benchmark_invalidate_dump_cache(repeat_time: int, func, raw_bytes: bytes, *args):
    """
    orjson will use utf-8 cache for the same input,
    so we need to invalidate it.
    returns time used (ns).
    """
    # prepare identical data, without sharing objects
    data_warmup = [json.loads(raw_bytes) for _ in range(10)]
    data = [json.loads(raw_bytes) for _ in range(repeat_time)]
    # disable GC
    gc_was_enabled = gc_prepare()
    try:
        # warm up
        for i in range(10):
            new_args = (data_warmup[i], *args)
            _ssrjson_benchmark.run_object_benchmark(func, new_args)
        #
        total = 0
        for i in range(repeat_time):
            new_args = (data[i], *args)
            total += _ssrjson_benchmark.run_object_benchmark(func, new_args)
        return total
    finally:
        if gc_was_enabled:
            gc.enable()


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

        # t0 = time.perf_counter()
        # cpu_times_before = process.cpu_times()
        # ctx_before = process.num_ctx_switches()
        # mem_before = process.memory_info().rss

        speed = benchmark_func(repeat_times, func, input_data)

        # End measuring
        # t1 = time.perf_counter()
        # cpu_times_after = process.cpu_times()
        # ctx_after = process.num_ctx_switches()

        # user_cpu = cpu_times_after.user - cpu_times_before.user
        # system_cpu = cpu_times_after.system - cpu_times_before.system
        # voluntary_ctx = ctx_after.voluntary - ctx_before.voluntary
        # involuntary_ctx = ctx_after.involuntary - ctx_before.involuntary
        # mem_after = process.memory_info().rss

        cur_obj[name] = {
            "speed": speed,
            # "user_cpu": user_cpu,
            # "system_cpu": system_cpu,
            # "ctx_vol": voluntary_ctx,
            # "ctx_invol": involuntary_ctx,
            # "mem_diff": mem_after - mem_before,
            # "wall_time": t1 - t0,
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
                    _, size, _, _ = _ssrjson_benchmark.inspect_pyunicode(output)
            else:
                size = (
                    len(input_data)
                    if isinstance(input_data, bytes)
                    else _ssrjson_benchmark.inspect_pyunicode(input_data)[1]
                )
            cur_obj["ssrjson_bytes_per_sec"] = (
                size * repeat_times / (cur_obj[name]["speed"] / _NS_IN_ONE_S)
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
    kind, str_size, is_ascii, _ = _ssrjson_benchmark.inspect_pyunicode(raw)
    curfile_obj["pyunicode_size"] = str_size
    curfile_obj["pyunicode_kind"] = kind
    curfile_obj["pyunicode_is_ascii"] = is_ascii
    repeat_times = int((process_bytes + bytes_size - 1) // bytes_size)

    for mode in LIBRARIES.keys():
        _run_benchmark(curfile_obj, repeat_times, raw_bytes, mode)


def get_head_rev_name():
    return (
        getattr(ssrjson, "__version__", None) or getattr(ssrjson, "ssrjson").__version__
    )


def get_real_output_file_name():
    rev = get_head_rev_name()
    if not rev:
        file = "benchmark_result.json"
    else:
        file = f"benchmark_result_{rev}.json"
    return file


def get_cpu_name() -> str:
    cpuinfo_spec = importlib.util.find_spec("cpuinfo")
    if cpuinfo_spec is not None:
        import cpuinfo

        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "UnknownCPU")
    else:
        # fallback
        cpu_name: str = platform.processor()
        if cpu_name.strip() == "":
            # linux fallback
            if os.path.exists("/proc/cpuinfo"):
                with open(file="/proc/cpuinfo", mode="r") as file:
                    cpu_info_lines = file.readlines()
                    for line in cpu_info_lines:
                        if "model name" in line:
                            cpu_name = re.sub(
                                pattern=r"model name\s+:\s+", repl="", string=line
                            )
                            break
            else:
                cpu_name = "UnknownCPU"
    # merge nearby spaces
    return re.sub(pattern=r"\s+", repl=" ", string=cpu_name).strip()


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
    return f"{mem_total / (1024**2):.3f}GiB"


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
        gbps = (data[cat]["ssrjson_bytes_per_sec"] * 8) / (1024**3)

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

        ssrjson_index = libs.index("ssrjson")
        ax.text(
            x_positions[ssrjson_index],
            vals[ssrjson_index] / 2,
            f"{gbps:.2f} Gb/s",
            ha="center",
            va="center",
            fontsize=10,
            color="#2c3e50",
            fontweight="bold",
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


def draw_page_number(c: "canvas.Canvas", page_num: int):
    from reportlab.lib.pagesizes import A4

    width, _ = A4
    c.setFont("Helvetica-Oblique", 8)  # italic
    c.setFillColorRGB(0.5, 0.5, 0.5)  # grey
    c.drawRightString(width - 40, 20, f"{page_num}")


def generate_pdf_report(
    figures: List[List[io.BytesIO]], header_text: str, output_pdf_path: str
) -> str:
    from reportlab.pdfgen import canvas
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg
    from reportlab.lib.pagesizes import A4

    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4

    # heading info
    heading = header_text.splitlines()
    # first line is # header
    header, heading_info = heading[0].removeprefix("#").strip(), heading[1:]
    c.setFont(PDF_HEADING_FONT, 20)
    text_obj = c.beginText(40, height - 50)
    text_obj.textLine(header)
    c.drawText(text_obj)

    # Wrap heading_info lines if overflow
    max_width = width - 80  # 40 margin on both sides
    wrapped_heading_info = []
    for line in heading_info:
        while c.stringWidth(line, PDF_TEXT_FONT, 10) > max_width:
            # Find a split point
            split_idx = int(max_width // c.stringWidth(" ", PDF_TEXT_FONT, 10))
            # Try to split at nearest space before split_idx
            space_idx = line.rfind(" ", 0, split_idx)
            if space_idx == -1:
                space_idx = split_idx
            wrapped_heading_info.append(line[:space_idx])
            # TODO fixed indent
            line = "                " + line[space_idx:].lstrip()
        wrapped_heading_info.append(line)
    heading_info = wrapped_heading_info

    c.setFont(PDF_TEXT_FONT, 10)
    text_obj = c.beginText(40, height - 70)
    for line in heading_info:
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.setFont("Helvetica-Oblique", 8)
    text = "This report was generated by https://github.com/Nambers/ssrJSON-benchmark"
    c.drawString(40, 20, text)
    link_start = 40 + c.stringWidth("This report was generated by ")
    link_end = link_start + c.stringWidth(
        "https://github.com/Nambers/ssrJSON-benchmark"
    )
    text_height = 5  # Adjusted height to better fit the link area
    c.linkURL(
        "https://github.com/Nambers/ssrJSON-benchmark",
        (link_start, 20, link_end, 20 + text_height),
        relative=1,
    )

    header_lines = header_text.count("\n") + 1
    header_height = header_lines * 14 + 10
    # subheading spacing = 30
    y_pos = height - header_height - 30
    bottom_margin = 20
    vertical_gap = 20

    p = 0

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
                draw_page_number(c, p)
                p += 1
                c.showPage()
                y_pos = height - bottom_margin

            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.setLineWidth(0.4)
            c.line(40, y_pos, width - 40, y_pos)

            renderPDF.draw(drawing, c, 40, y_pos - img_h)
            y_pos -= img_h + vertical_gap

    draw_page_number(c, p)
    c.save()
    return output_pdf_path


def generate_report(result: dict[str, dict[str, Any]], file: str, out_dir: str = CWD):
    file = file.removesuffix(".json")
    report_name = f"{file}.pdf"

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
        TIME=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        OS=f"{platform.system()} {platform.machine()} {platform.release()} {platform.version()}",
        PYTHON=sys.version,
        ORJSON_VER=orjson.__version__,
        SIMD_FLAGS=ssrjson.get_current_features(),
        CHIPSET=get_cpu_name(),
        MEM=get_mem_total(),
    )
    out_path = generate_pdf_report(
        figures,
        header_text=template,
        output_pdf_path=os.path.join(out_dir, report_name),
    )
    print(f"Report saved to {out_path}")


def generate_report_markdown(
    result: dict[str, dict[str, Any]], file: str, out_dir: str = CWD
):
    file = file.removesuffix(".json")
    report_name = f"{file}.md"
    report_folder = os.path.join(out_dir, f"{file}_report")

    # mkdir
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    template = template.format(
        REV=file.removeprefix("benchmark_result_").removesuffix(".json"),
        TIME=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        OS=f"{platform.system()} {platform.machine()} {platform.release()} {platform.version()}",
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
    print(f"Report saved to {os.path.join(report_folder, report_name)}")


def run_benchmark(process_bytes: int = 1e8):
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


def run_benchmark_default():
    """
    Run default benchmark with default parameters. Generate report in PDF.
    """
    j, file = run_benchmark()
    file = file.split("/")[-1]
    generate_report(j, file)
