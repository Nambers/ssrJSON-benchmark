import io
import math
import os
from typing import TYPE_CHECKING

from .benchmark import (
    BASE_INDEX_GROUPS,
    fetch_header,
    print_index_name,
    split_index_name,
)
from .result_types import BenchmarkFinalResult, BenchmarkResultPerFileTarget

if TYPE_CHECKING:
    from reportlab.pdfgen import canvas

_PDF_HEADING_FONT = "Helvetica-Bold"
_PDF_TEXT_FONT = "Courier"

# ssrjson's color is always fixed
_SSRJSON_COLOR = "#fd8d3c"

# Canonical library display order – libraries not in this list are appended at
# the end in their original discovery order.
_CANONICAL_LIB_ORDER = [
    "json",
    "ujson",
    "pydantic_core",
    "msgspec",
    "orjson",
    "ssrjson",
]

# Color palette derived from the original _LIBRARIES_COLORS in reverse order
# (excluding ssrjson). Colors are assigned sequentially to libraries enumerated
# from bottom-to-top (i.e. reversed non-ssrjson library list).
# One additional color is appended for when more libraries are present.
_OTHER_COLORS_PALETTE = [
    "#2c7fb8",  # blue   (was orjson)
    "#8856a7",  # purple (was msgspec)
    "#c994c7",  # light purple (was ujson)
    "#74c476",  # green  (was json)
    "#41b6c4",  # teal   (new extra color)
]


# ---------------------------------------------------------------------------
# Dynamic color assignment
# ---------------------------------------------------------------------------


def assign_colors(libraries: list[str]) -> dict[str, str]:
    """Assign colors to libraries dynamically.

    ssrjson always gets _SSRJSON_COLOR.
    Other libraries are enumerated bottom-to-top (reversed order from the
    libraries list), and each gets the next color from _OTHER_COLORS_PALETTE
    sequentially starting at index 0.

    Example: libraries = [json, msgspec, ssrjson]
      non-ssrjson bottom-to-top: [msgspec, json]
      msgspec -> palette[0] = #2c7fb8
      json    -> palette[1] = #8856a7
    """
    colors = {}
    non_ssrjson = [lib for lib in libraries if lib != "ssrjson"]
    # Bottom-to-top = reversed order
    for palette_idx, lib in enumerate(reversed(non_ssrjson)):
        colors[lib] = _OTHER_COLORS_PALETTE[palette_idx % len(_OTHER_COLORS_PALETTE)]
    if "ssrjson" in libraries:
        colors["ssrjson"] = _SSRJSON_COLOR
    return colors


# ---------------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------------


class PlotConfig:
    def __init__(
        self,
        bar_width: float = 0.2,
        fig_width_per_cat: float = 3,
        fig_height: float = 4,
        show_std_dev: bool = True,
        title_fontsize: int = 20,
        ratio_fontsize: int = 9,
        gbps_fontsize: int = 10,
        wspace: float = 0.0,
    ):
        self.bar_width = bar_width
        self.fig_width_per_cat = fig_width_per_cat
        self.fig_height = fig_height
        self.show_std_dev = show_std_dev
        self.title_fontsize = title_fontsize
        self.ratio_fontsize = ratio_fontsize
        self.gbps_fontsize = gbps_fontsize
        self.wspace = wspace


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_prepare():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"


def _get_ratio_color(ratio: float) -> str:
    if ratio < 1:
        return "#d63031"
    elif ratio == 1:
        return "black"
    elif ratio < 2:
        return "#e67e22"
    elif ratio < 4:
        return "#f39c12"
    elif ratio < 8:
        return "#27ae60"
    elif ratio < 16:
        return "#2980b9"
    else:
        return "#8e44ad"


# ---------------------------------------------------------------------------
# Single benchmark SVG plot
# ---------------------------------------------------------------------------


def plot_benchmark_svg(
    categories: list[str],
    data: dict[str, BenchmarkResultPerFileTarget],
    doc_name: str,
    config: PlotConfig | None = None,
) -> io.BytesIO:
    """Generate an SVG bar chart for a single file's benchmark results.

    Args:
        categories: list of benchmark group names to show
        data: mapping from group_name -> BenchmarkResultPerFileTarget
            (this is typically the .targets dict of a BenchmarkResultPerFile,
             or the BenchmarkResultPerFile itself accessed by group_name)
        doc_name: title for the chart
        config: optional PlotConfig
    Returns:
        BytesIO containing the SVG

    Groups that were skipped for this file (e.g. the cached-dump groups on an
    ASCII document) simply have no entry in *data* and their subplot is turned
    off, so no separate mask is needed.
    """
    import matplotlib.pyplot as plt

    if config is None:
        # Narrow sections (dumps to str, loads) hold two charts per row and can
        # afford room between them; dumps to bytes packs four across and has to
        # stay tight to fit the page width.
        config = (
            PlotConfig(fig_width_per_cat=5.0, wspace=0.3)
            if len(categories) <= 2
            else PlotConfig()
        )

    # Determine libraries as the union across all visible categories,
    # then sort according to the canonical display order.
    libs_seen: set[str] = set()
    for cat in categories:
        if cat in data:
            target = (
                data[cat]
                if isinstance(data[cat], BenchmarkResultPerFileTarget)
                else data[cat]
            )
            for lib in target.libraries:
                libs_seen.add(lib)
    # Sort by canonical order; unknown libs go to the end.
    _order_map = {name: idx for idx, name in enumerate(_CANONICAL_LIB_ORDER)}
    libs = sorted(libs_seen, key=lambda n: _order_map.get(n, len(_CANONICAL_LIB_ORDER)))
    lib_colors = assign_colors(libs) if libs else {}

    if not libs:
        # Fallback: empty plot
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    n = len(categories)
    bar_width = config.bar_width
    # Nominal canvas width. The saved SVG is padded up to this below: without
    # it each figure is cropped to its own content, and since the PDF stretches
    # every image to the page width, two sections with the same column count
    # come out at different sizes purely because their tick labels differ in
    # length.
    target_w_in = config.fig_width_per_cat * n

    fig, axs = plt.subplots(
        1,
        n,
        figsize=(config.fig_width_per_cat * n, config.fig_height),
        sharey=False,
        tight_layout=True,
        gridspec_kw={"wspace": config.wspace},
    )
    if n == 1:
        axs = [axs]

    for ax, cat in zip(axs, categories):
        if cat not in data:
            ax.axis("off")
            continue

        target = data[cat]
        # Determine which libraries actually participated in this category
        cat_libs_set = set(
            target.libraries
            if isinstance(target, BenchmarkResultPerFileTarget)
            else target.keys()
        )
        # Only keep libs that actually participated, preserving canonical order
        cat_libs = [name for name in libs if name in cat_libs_set]
        cat_colors = [lib_colors.get(n, "#999999") for n in cat_libs]
        cat_x_positions = [j * bar_width for j in range(len(cat_libs))]

        if isinstance(target, dict):
            # backward compat: dict access
            vals = []
            std_devs = []
            for name in cat_libs:
                lib_data = target.get(name, {})
                if isinstance(lib_data, dict):
                    vals.append(lib_data.get("ratio", 1.0))
                    std_devs.append(lib_data.get("std_dev", 0.0))
                else:
                    vals.append(lib_data.ratio)
                    std_devs.append(lib_data.std_dev)
            ssrjson_bps = target.get("ssrjson_bytes_per_sec", 0.0)
            if isinstance(ssrjson_bps, (dict,)):
                ssrjson_bps = 0.0
        else:
            vals = []
            std_devs = []
            for name in cat_libs:
                if name not in target:
                    vals.append(None)
                    std_devs.append(0.0)
                    continue
                lib_result = target[name]
                vals.append(lib_result.ratio)
                std_devs.append(lib_result.std_dev)
            ssrjson_bps = target.ssrjson_bytes_per_sec

        gbps = ssrjson_bps / (1024**3) if ssrjson_bps else 0.0

        # Error bars come from the recorded confidence interval of the summary
        # statistic (or the run-to-run range when --runs > 1). The old bar
        # plotted the standard deviation of a *single iteration*, which
        # describes the latency distribution rather than the uncertainty of the
        # charted number and overstates it by roughly sqrt(n).
        ratio_errors: list[tuple[float, float]] = []
        mismatched: list[bool] = []
        for j_idx, name in enumerate(cat_libs):
            val = vals[j_idx]
            if val is None or isinstance(target, dict) or name not in target:
                ratio_errors.append((0.0, 0.0))
                mismatched.append(False)
                continue
            lib_result = target[name]
            mismatched.append(not lib_result.output_ok)
            lo, hi = lib_result.ratio_lo, lib_result.ratio_hi
            if lo > 0 and hi > 0 and hi >= lo:
                ratio_errors.append((max(0.0, val - lo), max(0.0, hi - val)))
            elif (
                lib_result.std_dev > 0
                and lib_result.speed > 0
                and lib_result.repeat_count > 1
            ):
                # Pre-CI result files: derive a standard error of the mean so
                # old reports do not keep showing the overstated bar.
                cv = lib_result.std_dev * lib_result.repeat_count / lib_result.speed
                half = val * cv / math.sqrt(lib_result.repeat_count)
                ratio_errors.append((half, half))
            else:
                ratio_errors.append((0.0, 0.0))

        for xi, val, col, err, bad in zip(
            cat_x_positions, vals, cat_colors, ratio_errors, mismatched
        ):
            if val is None:
                continue

            ax.bar(
                xi,
                val,
                width=bar_width,
                color=col,
                hatch="//" if bad else None,
                edgecolor="#d63031" if bad else None,
            )

            err_lo, err_hi = err
            if config.show_std_dev and (err_lo > 0 or err_hi > 0):
                ax.errorbar(
                    xi,
                    val,
                    yerr=[[err_lo], [err_hi]],
                    fmt="none",
                    ecolor="#333333",
                    capsize=2,
                    capthick=0.8,
                    linewidth=0.8,
                )

            # Ratio label; "!" marks output that did not round-trip
            label_y = val + max(err_hi, 0) + 0.05
            ax.text(
                xi,
                label_y,
                f"{val:.2f}x" + ("!" if bad else ""),
                ha="center",
                va="bottom",
                fontsize=config.ratio_fontsize,
                color="#d63031" if bad else _get_ratio_color(val),
            )
            # Relative half-width of the interval below the ratio label
            if config.show_std_dev and (err_lo > 0 or err_hi > 0) and val > 0:
                pct = (err_lo + err_hi) / 2 / val * 100
                ax.text(
                    xi,
                    label_y - 0.02,
                    f"\u00b1{pct:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=config.ratio_fontsize - 2,
                    color="#888888",
                )

        if "ssrjson" in cat_libs and gbps > 0:
            ssrjson_index = cat_libs.index("ssrjson")
            if vals[ssrjson_index] is not None:
                ax.text(
                    cat_x_positions[ssrjson_index],
                    vals[ssrjson_index] / 2,
                    f"{gbps:.2f} GB/s",
                    ha="center",
                    va="center",
                    fontsize=config.gbps_fontsize,
                    color="#2c3e50",
                    fontweight="bold",
                )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        max_val_with_err = max(
            (v + e[1] for v, e in zip(vals, ratio_errors) if v is not None),
            default=1.0,
        )
        ax.set_ylim(0, max(max_val_with_err, 1.0) * 1.1)

        ax.tick_params(
            axis="both",
            which="both",
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)

        ax.set_xlabel(cat, fontsize=10, labelpad=6)

    fig.suptitle(
        doc_name,
        fontsize=config.title_fontsize,
        fontweight="bold",
        y=0.98,
    )

    legend_elements = [
        plt.Line2D([0], [0], color=lib_colors.get(name, "#999999"), lw=4, label=name)
        for name in libs
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
    from matplotlib.transforms import Bbox

    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    if tb.width < target_w_in:
        pad = (target_w_in - tb.width) / 2.0
        tb = Bbox.from_extents(tb.x0 - pad, tb.y0, tb.x1 + pad, tb.y1)
    plt.savefig(buf, format="svg", bbox_inches=tb)
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------


def plot_distribution_svg(
    ratio_distr: list[list[float]],
    lib_names: list[str],
    title: str = "Speed Ratio Distribution per Library",
) -> io.BytesIO:
    """Generate a box-plot SVG of speed ratio distributions per library.

    Colors follow the same dynamic assignment rules.
    """
    import matplotlib.pyplot as plt

    lib_colors = assign_colors(lib_names)

    fig, ax = plt.subplots(1, 1, figsize=(3 * len(lib_names), 4), tight_layout=True)

    bplot = ax.boxplot(
        ratio_distr,
        vert=True,
        patch_artist=True,
        showfliers=False,
    )

    for median in bplot["medians"]:
        median.set_color("red")
        median.set_linewidth(2)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.text(
        0.5,
        1.02,
        "Baseline (json)",
        ha="left",
        va="bottom",
        fontsize=10,
        color="gray",
    )
    ax.set_xticklabels(lib_names)
    ax.set_ylabel("Speed Ratio to json")
    ax.yaxis.set_major_formatter("{x:.1f}x")
    ax.set_title(title)

    for patch, name in zip(bplot["boxes"], lib_names):
        patch.set_facecolor(lib_colors.get(name, "#999999"))

    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Helpers for report layout
# ---------------------------------------------------------------------------


def _get_index_names(result: BenchmarkFinalResult) -> list[str]:
    """Index groups present in the result, in the order they were measured.

    Derived from the data rather than a module constant so that both the
    locality-split layout and pre-split result files render, and so that
    re-rendering never depends on which JSON libraries happen to be installed.
    """
    names = list(result.categories.keys())
    names += [name for name in result.results if name not in names]
    # Keep locality blocks together and the base groups in their canonical
    # order within each block, independent of dict ordering in the JSON.
    base_order = {base: i for i, base in enumerate(BASE_INDEX_GROUPS)}
    localities: list[str] = []
    for name in names:
        locality = split_index_name(name)[1]
        if locality not in localities:
            localities.append(locality)
    # Base-major: the report is three parts (dumps to bytes, dumps to str,
    # loads), each holding its localities, rather than one block per locality.
    return sorted(
        names,
        key=lambda n: (
            base_order.get(split_index_name(n)[0], len(base_order)),
            localities.index(split_index_name(n)[1]),
        ),
    )


def _get_cats(result: BenchmarkFinalResult) -> dict[str, list[str]]:
    """Ordered group names per index group."""
    cats: dict[str, list[str]] = {}
    for index_name in _get_index_names(result):
        ordered = list(result.categories.get(index_name, []))
        for file_result in result.results.get(index_name, {}).values():
            for group_name in file_result.targets:
                if group_name not in ordered:
                    ordered.append(group_name)
        cats[index_name] = ordered
    return cats


def _group_by_locality(result: BenchmarkFinalResult) -> dict[str, list[str]]:
    """{locality: [index_name, ...]}. Pre-split result files land under ''."""
    grouped: dict[str, list[str]] = {}
    for index_name in _get_index_names(result):
        _, locality = split_index_name(index_name)
        grouped.setdefault(locality, []).append(index_name)
    return grouped


def _get_non_baseline_libs(result: BenchmarkFinalResult) -> list[str]:
    """Get the union of non-baseline library names from the result data, sorted by canonical order."""
    libs_seen: set[str] = set()
    for files_dict in result.results.values():
        for filename in result.filenames:
            file_result = files_dict.get(filename)
            if file_result is None:
                continue
            for target in file_result.targets.values():
                for lib in target.libraries:
                    if lib != "json":
                        libs_seen.add(lib)
    _order_map = {name: idx for idx, name in enumerate(_CANONICAL_LIB_ORDER)}
    return sorted(libs_seen, key=lambda n: _order_map.get(n, len(_CANONICAL_LIB_ORDER)))


def _collect_ratios(result: BenchmarkFinalResult, cats, index_names):
    """Collect ratio distributions for non-baseline libraries over *index_names*.

    Groups that were skipped for a file simply have no target entry, so no
    separate ASCII mask is needed.
    """
    non_baseline_libs = _get_non_baseline_libs(result)
    ratios = {lib: [] for lib in non_baseline_libs}

    for indexed_group in index_names:
        files = result.results.get(indexed_group, {})
        for bench_filename in result.filenames:
            file_result = files.get(bench_filename)
            if file_result is None:
                continue
            for cat in cats.get(indexed_group, []):
                target = file_result.targets.get(cat)
                if target is None:
                    continue
                # Groups whose comparison is structurally unfair for this file
                # (ssrJSON skipping a UTF-8 cache write that orjson cannot skip)
                # keep their own chart but must not colour the headline summary.
                if not target.in_summary:
                    continue
                for lib_name in non_baseline_libs:
                    if lib_name in target:
                        ratios[lib_name].append(target[lib_name].ratio)
    return non_baseline_libs, [ratios[lib] for lib in non_baseline_libs]


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _draw_page_number(c: "canvas.Canvas", page_num: int):
    from reportlab.lib.pagesizes import A4

    width, _ = A4
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawRightString(width - 40, 20, f"{page_num}")


def _generate_pdf_report(
    figures: list[list[io.BytesIO]],
    section_names: list[str],
    header_text: str,
    output_pdf_path: str,
    distribution_svgs: list[io.BytesIO],
    summary_note: str | list[str] = "",
) -> str:
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from svglib.svglib import svg2rlg

    try:
        from svglib.fonts import FontMap

        font_map = FontMap()
        font_map.register_default_fonts()
        font_map.register_font("Helvetica", weight="700", rlgFontName="Helvetica-Bold")
    except ImportError:
        font_map = None

    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4

    heading = header_text.splitlines()
    header, heading_info = heading[0].removeprefix("#").strip(), heading[1:]
    c.setFont(_PDF_HEADING_FONT, 20)
    text_obj = c.beginText(40, height - 50)
    text_obj.textLine(header)
    c.drawText(text_obj)

    max_width = width - 80
    wrapped_heading_info = []
    for line in heading_info:
        while c.stringWidth(line, _PDF_TEXT_FONT, 10) > max_width:
            split_idx = int(max_width // c.stringWidth(" ", _PDF_TEXT_FONT, 10))
            space_idx = line.rfind(" ", 0, split_idx)
            if space_idx == -1:
                space_idx = split_idx
            wrapped_heading_info.append(line[:space_idx])
            line = "                " + line[space_idx:].lstrip()
        wrapped_heading_info.append(line)
    heading_info = wrapped_heading_info

    c.setFont(_PDF_TEXT_FONT, 10)
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
    c.linkURL(
        "https://github.com/Nambers/ssrJSON-benchmark",
        (link_start, 20, link_end, 25),
        relative=1,
    )

    header_lines = header_text.count("\n") + 1
    header_height = header_lines * 14 + 10
    y_pos = height - header_height - 40
    bottom_margin = 20
    vertical_gap = 20
    p = 0

    # TL;DR distribution plot
    text_obj = c.beginText()
    text_obj.setTextOrigin(40, y_pos)
    text_obj.setFont(_PDF_HEADING_FONT, 14)
    text_obj.textLine("TL;DR")
    c.drawText(text_obj)
    c.bookmarkHorizontal("TL;DR", 0, y_pos + 20)
    c.addOutlineEntry("TL;DR", "TL;DR", level=0)
    y_pos -= 20

    for distribution_svg in distribution_svgs:
        distribution_svg.seek(0)
        drawing = svg2rlg(distribution_svg, font_map=font_map)
        avail_w = width - 80
        scale = avail_w / drawing.width
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        img_h = drawing.height
        if y_pos - img_h - vertical_gap < bottom_margin:
            _draw_page_number(c, p)
            p += 1
            c.showPage()
            y_pos = height - bottom_margin
        renderPDF.draw(drawing, c, 40, y_pos - img_h)
        y_pos -= img_h + vertical_gap

    paragraphs = [summary_note] if isinstance(summary_note, str) else summary_note
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColorRGB(0.35, 0.35, 0.35)
        max_chars = 150
        for para in paragraphs:
            words, line = para.split(), ""
            note_lines = []
            for word in words:
                if len(line) + len(word) + 1 > max_chars:
                    note_lines.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            note_lines.append(line)
            for note_line in note_lines:
                c.drawString(40, y_pos, note_line)
                y_pos -= 9
            y_pos -= 4
        c.setFillColorRGB(0, 0, 0)
        y_pos -= 8

    for name, figs in zip(section_names, figures):
        text_obj = c.beginText()
        text_obj.setTextOrigin(40, y_pos)
        text_obj.setFont(_PDF_HEADING_FONT, 14)
        text_obj.textLine(name)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_figures(result: BenchmarkFinalResult, cats, index_names, fmt: str):
    """One SVG per (index group, file), in report order."""
    figures = []
    for indexed_group in index_names:
        files = result.results.get(indexed_group, {})
        group_figures = []
        for bench_filename in result.filenames:
            file_result = files.get(bench_filename)
            if file_result is None:
                continue
            print(f"Processing {bench_filename} [{indexed_group}]({fmt})")
            group_figures.append(
                plot_benchmark_svg(
                    cats[indexed_group], file_result.targets, bench_filename
                )
            )
        figures.append(group_figures)
    return figures


def _summary_note(result: BenchmarkFinalResult) -> str:
    """Explain which groups the aggregate leaves out, and why.

    Dropping them silently would be its own kind of dishonesty: the charts
    still show them, so the reader needs to know the box plot does not.
    """
    excluded: list[str] = []
    for files_dict in result.results.values():
        for file_result in files_dict.values():
            for group_name, target in file_result.targets.items():
                if not target.in_summary and group_name not in excluded:
                    excluded.append(group_name)
    if not excluded:
        return ""
    return (
        "Every library in this summary runs in the configuration its wheel "
        "ships with, including ssrJSON's UTF-8 cache writing (on by default, "
        "and not switchable at all in orjson, msgspec or pydantic_core). "
        "'dumps to bytes' is therefore the serialize-once cost and "
        "'dumps to bytes (cached)' the steady-state repeat-dump cost; real "
        "workloads sit between them depending on how often the caller "
        "re-serializes the same object. Excluded from this summary (still "
        "charted below): "
        + ", ".join(f"'{name}'" for name in excluded)
        + " -- ssrJSON is forced out of its default there to isolate what the "
        "cache write costs, which is an engine diagnostic rather than a "
        "package comparison."
    )


def _breakeven_rows(result: BenchmarkFinalResult):
    """Derive, per file, how many dumps of one object it takes for the UTF-8
    cache write to pay for itself.

    Pure arithmetic over numbers already measured, no extra benchmarking:
        N * t_nowrite  ==  t_write + (N - 1) * t_cached

    Two guards, both learned the hard way. Only the hot locality is used: the
    cached group re-dumps one live object, so pairing it with a cold
    first-dump would mix two memory regimes in one equation. And the write
    cost must clear its own confidence interval -- on files with few non-ASCII
    strings it is under the noise floor, and dividing two indistinguishable
    numbers produced break-evens below 1, which is not a physical answer. Such
    files are reported as "write cost not measurable" instead, which is the
    honest and also more useful statement: leave the default alone.
    """
    resolved: list[tuple[str, float]] = []
    negligible: list[str] = []
    for index_name, files_dict in result.results.items():
        if split_index_name(index_name)[1] != "hot":
            continue
        for file_name, file_result in files_dict.items():

            def ssr(group_name, _fr=file_result):
                target = _fr.targets.get(group_name)
                if target is None:
                    return None
                return _fr.targets[group_name].lib_results.get("ssrjson")

            w = ssr("dumps to bytes")
            nw = ssr("dumps to bytes (no cache write)")
            c = ssr("dumps to bytes (cached)")
            if not (w and nw and c):
                continue
            # Is writing measurably slower than not writing?
            if not (w.stat > nw.stat and w.stat_lo > nw.stat_hi):
                negligible.append(file_name)
                continue
            # Does reading the cache measurably help? Without that the write
            # can never amortise and there is no crossover.
            if not (nw.stat > c.stat and nw.stat_lo > c.stat_hi):
                negligible.append(file_name)
                continue
            resolved.append((file_name, (w.stat - c.stat) / (nw.stat - c.stat)))
    resolved.sort()
    return resolved, sorted(set(negligible))


def _breakeven_note(result: BenchmarkFinalResult) -> str:
    resolved, negligible = _breakeven_rows(result)
    if not resolved and not negligible:
        return ""
    parts = []
    if resolved:
        parts.append(
            "UTF-8 cache break-even for ssrJSON -- how many times one object "
            "must be serialized before writing the cache pays for itself: "
            + "; ".join(f"{name} N={n:.1f}" for name, n in resolved)
            + ". Below that, ssrjson.write_utf8_cache(False) is faster; above "
            "it, the default wins."
        )
    if negligible:
        parts.append(
            "On " + ", ".join(negligible) + " the cache write costs less than "
            "the measurement can resolve, so there is no crossover to report "
            "and the default needs no thought."
        )
    return " ".join(parts)


def _build_distributions(
    result: BenchmarkFinalResult, cats
) -> list[tuple[str, io.BytesIO]]:
    """One ratio box plot per locality, so the reader can see directly how much
    of a library's advantage comes from the cold regime."""
    distributions = []
    for locality, index_names in _group_by_locality(result).items():
        libs, ratio_lists = _collect_ratios(result, cats, index_names)
        title = "Speed Ratio Distribution per Library"
        if locality:
            title += f" ({locality})"
        distributions.append(
            (locality, plot_distribution_svg(ratio_lists, libs, title))
        )
    return distributions


def generate_report_pdf(
    result: BenchmarkFinalResult, file: str, out_dir: str | None = None
) -> str:
    _plot_prepare()

    if out_dir is None:
        out_dir = os.getcwd()

    file = file.removesuffix(".json")
    report_name = f"{file}.pdf"

    cats = _get_cats(result)
    index_names = _get_index_names(result)
    figures = _build_figures(result, cats, index_names, "PDF")
    distributions = _build_distributions(result, cats)

    template = fetch_header(result)
    out_path = _generate_pdf_report(
        figures,
        section_names=[print_index_name(name) for name in index_names],
        header_text=template,
        output_pdf_path=os.path.join(out_dir, report_name),
        distribution_svgs=[svg for _, svg in distributions],
        summary_note=[_summary_note(result), _breakeven_note(result)],
    )
    return out_path


def _safe_slug(name: str) -> str:
    """Index names contain '|', which is awkward in filenames and markdown links."""
    return name.replace("|", "_")


def generate_report_markdown(
    result: BenchmarkFinalResult, file: str, out_dir: str | None = None
) -> str:
    _plot_prepare()

    if out_dir is None:
        out_dir = os.getcwd()

    file = file.removesuffix(".json")
    report_name = f"{file}.md"
    report_folder = os.path.join(out_dir, f"{file}_report")

    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    template = fetch_header(result)
    template += "\n\n## TL;DR\n\nTLDRIMGPLACEHOLDER\n\n"

    cats = _get_cats(result)

    for indexed_group in _get_index_names(result):
        template += f"\n\n## {print_index_name(indexed_group)}\n\n"
        files = result.results.get(indexed_group, {})
        slug = _safe_slug(indexed_group)
        for bench_filename in result.filenames:
            file_result = files.get(bench_filename)
            if file_result is None:
                continue
            print(f"Processing {bench_filename} [{indexed_group}](Markdown)")
            svg_buf = plot_benchmark_svg(
                cats[indexed_group], file_result.targets, bench_filename
            )
            svg_name = f"{bench_filename}_{slug}.svg"
            with open(os.path.join(report_folder, svg_name), "wb") as svg_file:
                svg_file.write(svg_buf.getvalue())
            template += f"![{bench_filename}_{slug}](./{svg_name})\n\n"

    tldr = ""
    for locality, dist_svg in _build_distributions(result, cats):
        svg_name = (
            f"ratio_distribution_{locality}.svg"
            if locality
            else "ratio_distribution.svg"
        )
        with open(os.path.join(report_folder, svg_name), "wb") as svg_file:
            svg_file.write(dist_svg.getvalue())
        tldr += f"![ratio_distribution]({'./' + svg_name})\n\n"
    for note in (_summary_note(result), _breakeven_note(result)):
        if note:
            tldr += f"> {note}\n\n"
    template = template.replace("TLDRIMGPLACEHOLDER", tldr.strip())

    ret = os.path.join(report_folder, report_name)
    with open(ret, "w") as f:
        f.write(template)
    return ret
