import gc
import io
import json
import math
import multiprocessing
import os
import pathlib
import platform
import re
import sys
import time
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import orjson
import ssrjson
import ujson

from . import _ssrjson_benchmark
from .result_types import BenchmarkFinalResult, BenchmarkResultPerFile

if TYPE_CHECKING:
    from reportlab.pdfgen import canvas

_SSRJSON_HAS_WRITE_UTF8_CACHE = hasattr(ssrjson, "write_utf8_cache")

if _SSRJSON_HAS_WRITE_UTF8_CACHE:
    ssrjson.write_utf8_cache(False)

mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"


CUR_FILE = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(CUR_FILE)
CWD = os.getcwd()
_NS_IN_ONE_S = 1000000000

PDF_HEADING_FONT = "Helvetica-Bold"
PDF_TEXT_FONT = "Courier"

# baseline is the first one.
LIBRARIES_COLORS = {
    "json": "#74c476",
    "ujson": "#c994c7",
    "orjson": "#2c7fb8",
    "ssrjson": "#fd8d3c",
}

MAX_BIN_BYTES_SIZE = 512 * 1024 * 1024  # 512MiB

INDEXED_GROUPS = ["load&dump", "dumps_to_bytes"]
PRINT_INDEX_GROUPS = ["Load & Dump", "Dumps to Bytes"]


class BenchmarkFunction:
    def __init__(self, func: Callable, library_name: str) -> None:
        self.func = func
        self.library_name = library_name


class BenchmarkGroup:
    def __init__(
        self,
        benchmarker: Callable,
        functions: list[BenchmarkFunction],
        index_name: str,
        group_name: str,
        input_preprocessor: Callable[[Any], Any] = lambda x: x,
    ) -> None:
        self.benchmarker = benchmarker
        self.functions = functions
        self.index_name = index_name
        self.group_name = group_name
        self.input_preprocessor = input_preprocessor


# benchmarkers
def _benchmark(repeat_time: int, times_per_bin: int, func, *args):
    """
    Run repeat benchmark, with utf-8 cache.
    returns time used (ns).
    """
    # times_per_bin not used
    # disable automatic GC
    gc_was_enabled = _gc_prepare()
    try:
        # warm up
        _ssrjson_benchmark.run_object_accumulate_benchmark(func, 100, args)
        return _ssrjson_benchmark.run_object_accumulate_benchmark(
            func, repeat_time, args
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_unicode_arg(repeat_time: int, times_per_bin: int, func, unicode: str):
    """
    Run repeat benchmark, disabling utf-8 cache.
    returns time used (ns).
    """
    # disable automatic GC
    gc_was_enabled = _gc_prepare()
    try:
        times_left = repeat_time
        total = 0
        while times_left != 0:
            cur_bin_size = min(times_left, times_per_bin)
            times_left -= cur_bin_size
            # prepare identical data, without sharing objects
            benchmark_data = _ssrjson_benchmark.copy_unicode_list_invalidate_cache(
                unicode, cur_bin_size + 1
            )
            # warm up
            _ssrjson_benchmark.run_object_benchmark(func, (benchmark_data[0],))
            #
            for i in range(1, cur_bin_size + 1):
                total += _ssrjson_benchmark.run_object_benchmark(
                    func, (benchmark_data[i],)
                )
            del benchmark_data
        return total
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_invalidate_dump_cache(
    repeat_time: int, times_per_bin: int, func, raw_bytes: bytes
):
    """
    Invalidate utf-8 cache for the same input.
    returns time used (ns).
    """
    # disable automatic GC
    gc_was_enabled = _gc_prepare()
    try:
        times_left = repeat_time
        total = 0
        while times_left != 0:
            cur_bin_size = min(times_left, times_per_bin)
            times_left -= cur_bin_size
            # prepare identical data, without sharing objects
            benchmark_data = [json.loads(raw_bytes) for _ in range(cur_bin_size + 1)]
            # warm up
            _ssrjson_benchmark.run_object_benchmark(func, (benchmark_data[0],))
            #
            for i in range(1, cur_bin_size + 1):
                total += _ssrjson_benchmark.run_object_benchmark(
                    func, (benchmark_data[i],)
                )
            del benchmark_data
        return total
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_with_dump_cache(
    repeat_time: int, times_per_bin: int, func, raw_bytes: bytes
):
    # disable automatic GC
    gc_was_enabled = _gc_prepare()
    if _SSRJSON_HAS_WRITE_UTF8_CACHE:
        ssrjson.write_utf8_cache(True)
    try:
        total = 0
        data = json.loads(raw_bytes)
        # warm up
        _ssrjson_benchmark.run_object_benchmark(
            func, (data,)
        )  # allow `func`` to cache the UTF-8 source in data, at warm up stage
        # the simplest benchmark way
        for _ in range(repeat_time):
            total += _ssrjson_benchmark.run_object_benchmark(func, (data,))
        return total
    finally:
        if gc_was_enabled:
            gc.enable()
        if _SSRJSON_HAS_WRITE_UTF8_CACHE:
            ssrjson.write_utf8_cache(False)


def _get_benchmark_defs() -> tuple[BenchmarkGroup, ...]:
    return (
        BenchmarkGroup(
            _benchmark_unicode_arg,
            [
                BenchmarkFunction(json.loads, "json"),
                BenchmarkFunction(ujson.loads, "ujson"),
                BenchmarkFunction(orjson.loads, "orjson"),
                BenchmarkFunction(ssrjson.loads, "ssrjson"),
            ],
            INDEXED_GROUPS[0],
            "loads(str)",
            input_preprocessor=lambda x: x.decode("utf-8"),
        ),
        BenchmarkGroup(
            _benchmark,
            [
                BenchmarkFunction(json.loads, "json"),
                BenchmarkFunction(ujson.loads, "ujson"),
                BenchmarkFunction(orjson.loads, "orjson"),
                BenchmarkFunction(ssrjson.loads, "ssrjson"),
            ],
            INDEXED_GROUPS[0],
            "loads(bytes)",
        ),
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            [
                BenchmarkFunction(lambda x: json.dumps(x, ensure_ascii=False), "json"),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, ensure_ascii=False), "ujson"
                ),
                BenchmarkFunction(lambda x: orjson.dumps(x).decode("utf-8"), "orjson"),
                BenchmarkFunction(ssrjson.dumps, "ssrjson"),
            ],
            INDEXED_GROUPS[0],
            "dumps",
        ),
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            [
                BenchmarkFunction(
                    lambda x: json.dumps(x, indent=2, ensure_ascii=False), "json"
                ),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, indent=2, ensure_ascii=False), "ujson"
                ),
                BenchmarkFunction(
                    lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2).decode(
                        "utf-8"
                    ),
                    "orjson",
                ),
                BenchmarkFunction(lambda x: ssrjson.dumps(x, indent=2), "ssrjson"),
            ],
            INDEXED_GROUPS[0],
            "dumps(indented2)",
        ),
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            [
                BenchmarkFunction(
                    lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"), "json"
                ),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                    "ujson",
                ),
                BenchmarkFunction(orjson.dumps, "orjson"),
                BenchmarkFunction(ssrjson.dumps_to_bytes, "ssrjson"),
            ],
            INDEXED_GROUPS[1],
            "dumps_to_bytes",
        ),
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            [
                BenchmarkFunction(
                    lambda x: json.dumps(x, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    "json",
                ),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    "ujson",
                ),
                BenchmarkFunction(
                    lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2), "orjson"
                ),
                BenchmarkFunction(
                    lambda x: ssrjson.dumps_to_bytes(x, indent=2), "ssrjson"
                ),
            ],
            INDEXED_GROUPS[1],
            "dumps_to_bytes(indented2)",
        ),
        BenchmarkGroup(
            _benchmark_with_dump_cache,
            [
                BenchmarkFunction(
                    lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"), "json"
                ),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                    "ujson",
                ),
                BenchmarkFunction(orjson.dumps, "orjson"),
                BenchmarkFunction(ssrjson.dumps_to_bytes, "ssrjson"),
            ],
            INDEXED_GROUPS[1],
            "dumps_to_bytes(cache)",
        ),
        BenchmarkGroup(
            _benchmark_with_dump_cache,
            [
                BenchmarkFunction(
                    lambda x: json.dumps(x, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    "json",
                ),
                BenchmarkFunction(
                    lambda x: ujson.dumps(x, indent=2, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    "ujson",
                ),
                BenchmarkFunction(
                    lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2), "orjson"
                ),
                BenchmarkFunction(
                    lambda x: ssrjson.dumps_to_bytes(x, indent=2), "ssrjson"
                ),
            ],
            INDEXED_GROUPS[1],
            "dumps_to_bytes(cache,indented2)",
        ),
    )


def _get_benchmark_libraries() -> dict[str, BenchmarkGroup]:
    return {x.group_name: x for x in _get_benchmark_defs()}


def _gc_prepare():
    """
    Call collect once, and then disable automatic GC.
    Return True if automatic GC was enabled.
    """
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    return gc_was_enabled


def _get_processed_size(func: Callable, input_data, is_dumps):
    if is_dumps:
        # get output size of dumps
        data_obj = json.loads(input_data)
        output = func(data_obj)
        if isinstance(output, bytes):
            size = len(output)
        else:
            size = _ssrjson_benchmark.inspect_pyunicode(output)[1]
    else:
        # get loads input size
        size = (
            len(input_data)
            if isinstance(input_data, bytes)
            else _ssrjson_benchmark.inspect_pyunicode(input_data)[1]
        )
    return size


def _run_benchmark(
    cur_result_file: BenchmarkResultPerFile,
    repeat_times: int,
    times_per_bin: int,
    input_data: str | bytes,
    benchmark_group: BenchmarkGroup,
):
    group_name = benchmark_group.group_name
    cur_target = cur_result_file[group_name]

    input_data = benchmark_group.input_preprocessor(input_data)

    for benchmark_target in benchmark_group.functions:
        prefix = f"[{benchmark_target.library_name}][{benchmark_group.group_name}]"
        print(
            prefix
            + (" " * max(0, 40 - len(prefix)))
            + f"repeat_times={repeat_times} times_per_bin={times_per_bin}"
        )
        speed = benchmark_group.benchmarker(
            repeat_times, times_per_bin, benchmark_target.func, input_data
        )
        cur_lib = cur_target[benchmark_target.library_name]
        cur_lib.speed = speed

    baseline_name = "json"
    baseline_data = cur_target[baseline_name]
    for benchmark_target in benchmark_group.functions:
        cur_lib = cur_target[benchmark_target.library_name]
        if benchmark_target.library_name == "ssrjson":
            # calculate bytes per sec for ssrJSON
            size = _get_processed_size(
                benchmark_target.func, input_data, "dumps" in group_name
            )
            cur_target.ssrjson_bytes_per_sec = (
                size * repeat_times / (cur_lib.speed / _NS_IN_ONE_S)
            )

        cur_lib.ratio = (
            math.inf
            if baseline_data.speed == 0
            else (baseline_data.speed / cur_lib.speed)
        )


def _run_file_benchmark(
    benchmark_libraries: dict[str, BenchmarkGroup],
    file: pathlib.Path,
    process_bytes: int,
    bin_process_bytes: int,
    index_s: str,
):
    print(f"Running benchmark for {file.name}, index group: {index_s}")
    with open(file, "rb") as f:
        raw_bytes = f.read()
    raw = raw_bytes.decode("utf-8")
    base_file_name = os.path.basename(file)
    cur_result_file = BenchmarkResultPerFile()
    cur_result_file.byte_size = bytes_size = len(raw_bytes)
    if bytes_size == 0:
        raise RuntimeError(f"File {file} is empty.")
    kind, str_size, is_ascii, _ = _ssrjson_benchmark.inspect_pyunicode(raw)
    cur_result_file.pyunicode_size = str_size
    cur_result_file.pyunicode_kind = kind
    cur_result_file.pyunicode_is_ascii = is_ascii
    repeat_times = int((process_bytes + bytes_size - 1) // bytes_size)
    times_per_bin = max(1, bin_process_bytes // bytes_size)

    for benchmark_group in benchmark_libraries.values():
        if benchmark_group.index_name == index_s:
            _run_benchmark(
                cur_result_file, repeat_times, times_per_bin, raw_bytes, benchmark_group
            )
    return base_file_name, cur_result_file


def _get_head_rev_name():
    return (
        getattr(ssrjson, "__version__", None) or getattr(ssrjson, "ssrjson").__version__
    )


def _get_real_output_file_name():
    rev = _get_head_rev_name()
    if not rev:
        file = "benchmark_result.json"
    else:
        file = f"benchmark_result_{rev}.json"
    return file


def _get_cpu_name() -> str:
    cpuinfo_spec = find_spec("cpuinfo")
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


def _get_mem_total() -> str:
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

        mem_total = psutil.virtual_memory().total // 1024  # in KB
    return f"{mem_total / (1024**2):.3f}GiB"


def _get_ratio_color(ratio: float) -> str:
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


def _plot_relative_ops(categories: list[str], data: dict, doc_name: str) -> io.BytesIO:
    libs = list(LIBRARIES_COLORS.keys())
    colors = [LIBRARIES_COLORS[n] for n in libs]
    n = len(categories)
    bar_width = 0.2
    inner_pad = 0

    fig, axs = plt.subplots(
        1,
        n,
        figsize=(3 * n, 4),
        sharey=False,
        tight_layout=True,
        gridspec_kw={"wspace": 0},
    )

    x_positions = [i * (bar_width + inner_pad) for i in range(len(libs))]

    for ax, cat in zip(axs, categories):
        vals = [1.0] + [data[cat][name]["ratio"] for name in libs[1:]]
        gbps = (data[cat]["ssrjson_bytes_per_sec"]) / (1024**3)

        for xi, val, col in zip(x_positions, vals, colors):
            ax.bar(xi, val, width=bar_width, color=col)
            ax.text(
                xi,
                val + 0.05,
                f"{val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color=_get_ratio_color(val),
            )

        ssrjson_index = libs.index("ssrjson")
        ax.text(
            x_positions[ssrjson_index],
            vals[ssrjson_index] / 2,
            f"{gbps:.2f} GB/s",
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


def _draw_page_number(c: "canvas.Canvas", page_num: int):
    from reportlab.lib.pagesizes import A4

    width, _ = A4
    c.setFont("Helvetica-Oblique", 8)  # italic
    c.setFillColorRGB(0.5, 0.5, 0.5)  # grey
    c.drawRightString(width - 40, 20, f"{page_num}")


def _generate_pdf_report(
    figures: List[List[io.BytesIO]], header_text: str, output_pdf_path: str
) -> str:
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from svglib.svglib import svg2rlg

    try:
        from svglib.fonts import FontMap

        font_map = FontMap()
        font_map.register_default_fonts()
        # workaround for matplotlib using 700 to represent bold font, but svg2rlg using 700 as normal.
        font_map.register_font("Helvetica", weight="700", rlgFontName="Helvetica-Bold")
    except ImportError:
        font_map = None

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

    for i in range(len(INDEXED_GROUPS)):
        name = PRINT_INDEX_GROUPS[i]
        figs = figures[i]

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
                _draw_page_number(c, p)
                p += 1
                c.showPage()
                y_pos = height - bottom_margin

            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.setLineWidth(0.4)
            c.line(40, y_pos, width - 40, y_pos)

            renderPDF.draw(drawing, c, 40, y_pos - img_h)
            y_pos -= img_h + vertical_gap

    _draw_page_number(c, p)
    c.save()
    return output_pdf_path


def _fetch_header(rev) -> str:
    with open(os.path.join(CUR_DIR, "template.md"), "r") as f:
        template = f.read()
    return template.format(
        REV=rev,
        TIME=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        OS=f"{platform.system()} {platform.machine()} {platform.release()} {platform.version()}",
        PYTHON=sys.version,
        ORJSON_VER=orjson.__version__,
        UJSON_VER=ujson.__version__,
        SIMD_FLAGS=ssrjson.get_current_features(),
        CHIPSET=_get_cpu_name(),
        MEM=_get_mem_total(),
    )


def generate_report_pdf(result: BenchmarkFinalResult, file: str, out_dir: str = CWD):
    """
    Generate PDF report, using `result`.
    """
    file = file.removesuffix(".json")
    report_name = f"{file}.pdf"

    figures = [[] for _ in range(len(INDEXED_GROUPS))]
    benchmark_groups = _get_benchmark_libraries()
    cats = [
        [a for a in result.categories if benchmark_groups[a].index_name == index_name]
        for index_name in INDEXED_GROUPS
    ]

    for i in range(len(INDEXED_GROUPS)):
        for bench_filename in result.filenames:
            print(f"Processing {bench_filename} [{INDEXED_GROUPS[i]}](PDF)")
            figures[i].append(
                _plot_relative_ops(
                    cats[i],
                    result.results[INDEXED_GROUPS[i]][bench_filename],
                    bench_filename,
                )
            )

    template = _fetch_header(
        file.removeprefix("benchmark_result_").removesuffix(".json")
    )
    out_path = _generate_pdf_report(
        figures,
        header_text=template,
        output_pdf_path=os.path.join(out_dir, report_name),
    )
    print(f"Report saved to {out_path}")
    return out_path


def generate_report_markdown(
    result: BenchmarkFinalResult, file: str, out_dir: str = CWD
):
    """
    Generate Markdown report, using `result`.
    """
    file = file.removesuffix(".json")
    report_name = f"{file}.md"
    report_folder = os.path.join(out_dir, f"{file}_report")

    # mkdir
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    template = _fetch_header(
        file.removeprefix("benchmark_result_").removesuffix(".json")
    )

    benchmark_groups = _get_benchmark_libraries()
    cats = [
        [a for a in result.categories if benchmark_groups[a].index_name == index_name]
        for index_name in INDEXED_GROUPS
    ]

    for i in range(len(INDEXED_GROUPS)):
        template += f"\n\n## {PRINT_INDEX_GROUPS[i]}\n\n"
        for bench_filename in result.filenames:
            print(f"Processing {bench_filename} [{INDEXED_GROUPS[i]}](Markdown)")
            with open(
                os.path.join(
                    report_folder, f"{bench_filename}_{INDEXED_GROUPS[i]}.svg"
                ),
                "wb",
            ) as svg_file:
                svg_file.write(
                    _plot_relative_ops(
                        cats[i],
                        result.results[INDEXED_GROUPS[i]][bench_filename],
                        bench_filename,
                    ).getvalue()
                )
            # add svg
            template += f"![{bench_filename}_{INDEXED_GROUPS[i]}](./{bench_filename}_{INDEXED_GROUPS[i]}.svg)\n\n"

    ret = os.path.join(report_folder, report_name)
    with open(ret, "w") as f:
        f.write(template)
    print(f"Report saved to {ret}")
    return ret


def parse_file_result(j):
    return BenchmarkFinalResult.parse(j)


def is_unix_except_macos():
    system = platform.system()
    return system in ("Linux", "AIX", "FreeBSD")


def _set_multiprocessing_start_method():
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise


def run_benchmark(
    files: list[pathlib.Path],
    process_bytes: int,
    bin_process_bytes: int,
):
    """
    Generate a JSON result of benchmark.
    Also returns a result object.
    """
    # Set multiprocessing start method to fork, if Python version is 3.14+ on Unix
    if sys.version_info >= (3, 14) and is_unix_except_macos():
        _set_multiprocessing_start_method()

    file = _get_real_output_file_name()

    result = BenchmarkFinalResult()
    result.results = dict()

    benchmark_libraries = _get_benchmark_libraries()

    result.categories = list(benchmark_libraries.keys())
    result.filenames = [files[i].name for i in range(len(files))]

    for index_s in INDEXED_GROUPS:
        result.results[index_s] = dict()
        for bench_file in files:
            k, v = _run_file_benchmark(
                benchmark_libraries,
                bench_file,
                process_bytes,
                bin_process_bytes,
                index_s,
            )
            result.results[index_s][k] = v
    output_result = result.dumps()

    if os.path.exists(file):
        os.remove(file)

    with open(f"{file}", "w", encoding="utf-8") as f:
        f.write(output_result)
    return result, file
