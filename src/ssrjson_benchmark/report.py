import io
import itertools
import math
import os
import statistics
import textwrap
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

_PAGE_MARGIN = 40
_REPO_URL = "https://github.com/Nambers/ssrJSON-benchmark"
# Charts are drawn at exactly the width they are printed on A4 (595.276pt)
# inside the margins: font sizes below are real point sizes, and every chart
# in the report shares one scale.
_FIG_WIDTH_IN = (595.276 - 2 * _PAGE_MARGIN) / 72

# All text is one dark ink; color is reserved for the data.
_INK = "#1a1a1a"
_AXIS = "#333333"
_BASELINE = "#666666"
_MISMATCH = "#d63031"
_MEDIAN = "#e31a1c"

# One color per library name, so a library keeps its color in every chart
# whichever other libraries happen to take part (pydantic_core has no loads,
# msgspec no indented dumps). These are the hues earlier reports used, in
# canonical display order.
_LIB_COLORS = {
    "json": "#41b6c4",
    "ujson": "#74c476",
    "pydantic_core": "#c994c7",
    "msgspec": "#8856a7",
    "orjson": "#2c7fb8",
    "ssrjson": "#fd8d3c",
}
# For libraries this module does not know yet, handed out in order of first
# appearance so they never take a known library's color.
_EXTRA_COLORS = ["#969696", "#e377c2", "#bcbd22", "#8c564b"]


def assign_colors(libraries: list[str]) -> dict[str, str]:
    """Color per library, independent of which other libraries are present."""
    extra = itertools.cycle(_EXTRA_COLORS)
    return {lib: _LIB_COLORS.get(lib) or next(extra) for lib in libraries}


def _order_libs(libs) -> list[str]:
    """Canonical order; libraries this module does not know go last, by name."""
    order = {name: idx for idx, name in enumerate(_LIB_COLORS)}
    return sorted(set(libs), key=lambda n: (order.get(n, len(order)), n))


def _fmt_ratio(value: float) -> str:
    return f"{value:.1f}x" if value >= 10 else f"{value:.2f}x"


# Per-document chart geometry, in inches: every benchmark group gets a column
# a quarter of the page wide, filled from the left, so bars have the same width
# in every chart and columns line up down the whole report.
_COL_W = _FIG_WIDTH_IN / 4
_PLOT_H = 1.12


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_prepare():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["hatch.linewidth"] = 0.6


def _bar_entry(target: BenchmarkResultPerFileTarget, lib: str):
    """(ratio, err_lo, err_hi, output_ok) for one library, or None if absent.

    Error bars come from the recorded confidence interval of the summary
    statistic (or the run-to-run range when --runs > 1). Pre-CI result files
    fall back to a standard error of the mean derived from the per-iteration
    standard deviation, which would otherwise overstate it by about sqrt(n).
    """
    if lib not in target:
        return None
    r = target[lib]
    val = r.ratio
    lo, hi = r.ratio_lo, r.ratio_hi
    if lo > 0 and hi > 0 and hi >= lo:
        err = (max(0.0, val - lo), max(0.0, hi - val))
    elif r.std_dev > 0 and r.speed > 0 and r.repeat_count > 1:
        cv = r.std_dev * r.repeat_count / r.speed
        half = val * cv / math.sqrt(r.repeat_count)
        err = (half, half)
    else:
        err = (0.0, 0.0)
    return val, err[0], err[1], r.output_ok


def _legend_row(fig, handles, y: float):
    """One centred row of legend entries directly under a chart's title."""
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.6,
        borderaxespad=0,
        borderpad=0,
        labelcolor=_INK,
    )


def _save_svg(fig) -> io.BytesIO:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    # No tight bbox: every chart keeps its nominal size so the PDF can place
    # them all at the same scale.
    fig.savefig(buf, format="svg", transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Single benchmark SVG plot
# ---------------------------------------------------------------------------


def plot_benchmark_svg(
    categories: list[str],
    data: dict[str, BenchmarkResultPerFileTarget],
    doc_name: str,
    *,
    note: str = "",
) -> io.BytesIO:
    """Speed ratio to json for one document, one bar group per benchmark group.

    Every group in *categories* keeps its column, filled from the left on the
    report-wide grid, so a group sits in the same place in every chart. Groups
    that were skipped for this file (e.g. the cached-dump groups on an ASCII
    document) have no entry in *data* and show "N/A" in their column. Groups
    share one y scale so they compare by height.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    present = [cat for cat in categories if cat in data]
    libs = _order_libs(lib for cat in present for lib in data[cat].libraries)
    lib_colors = assign_colors(libs)

    W = _FIG_WIDTH_IN
    top_pad, title_h, legend_h = 0.03, 0.23, 0.21
    note_h = 0.17 if note else 0.0
    bottom_h = 0.34
    head_h = top_pad + title_h + legend_h + note_h
    H = head_h + _PLOT_H + bottom_h
    fig = plt.figure(figsize=(W, H))

    fig.text(
        0.5,
        1 - top_pad / H,
        doc_name,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=_INK,
    )
    if note:
        fig.text(
            0.5,
            1 - (top_pad + title_h + legend_h) / H,
            note,
            ha="center",
            va="top",
            fontsize=7.5,
            color=_INK,
        )
    if not libs:
        return _save_svg(fig)
    _legend_row(
        fig,
        [Line2D([0], [0], color=lib_colors[lib], lw=5, label=lib) for lib in libs],
        1 - (top_pad + title_h) / H,
    )

    entries = {cat: [_bar_entry(data[cat], lib) for lib in libs] for cat in present}
    y_end = max(
        [1.0] + [e[0] + e[2] for rows in entries.values() for e in rows if e is not None]
    )
    # Headroom above the tallest bar for its value label.
    y_max = y_end * _PLOT_H / (_PLOT_H - 12 / 72)

    # More groups than the grid has columns: narrow them to still fit.
    col_w = min(_COL_W, W / len(categories))
    pad = 0.07
    for i, cat in enumerate(categories):
        left = i * col_w + pad
        ax = fig.add_axes((left / W, bottom_h / H, (col_w - 2 * pad) / W, _PLOT_H / H))
        ax.set_facecolor("none")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.55, len(libs) - 0.45)
        ax.set_ylim(0, y_max)
        ax.annotate(
            cat,
            xy=(0.5, 0),
            xycoords="axes fraction",
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color=_INK,
        )
        if cat not in entries:
            ax.text(
                0.5,
                0.35,
                "N/A",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=_INK,
            )
            continue
        ax.axhline(1.0, color=_BASELINE, linestyle="--", linewidth=0.8, zorder=4)

        for x, (lib, entry) in enumerate(zip(libs, entries[cat])):
            if entry is None:
                continue
            val, err_lo, err_hi, ok = entry
            ax.bar(
                x,
                val,
                width=0.88,
                color=lib_colors[lib],
                hatch=None if ok else "////",
                edgecolor=_MISMATCH if not ok else "none",
                linewidth=0,
            )
            if err_lo > 0 or err_hi > 0:
                ax.errorbar(
                    x,
                    val,
                    yerr=[[err_lo], [err_hi]],
                    fmt="none",
                    ecolor=_AXIS,
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                )
            # "!" marks output that did not round-trip
            ax.annotate(
                _fmt_ratio(val) + ("" if ok else "!"),
                xy=(x, val + err_hi),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=_INK if ok else _MISMATCH,
                fontweight="bold" if lib == "ssrjson" else "normal",
            )

        # Only set when ssrjson was measured in this group.
        bps = data[cat].ssrjson_bytes_per_sec
        if bps:
            ax.annotate(
                f"ssrjson {bps / 1024**3:.2f} GB/s",
                xy=(0.5, 0),
                xycoords="axes fraction",
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                color=_INK,
            )

    return _save_svg(fig)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------


def _lighten(hex_color: str, amount: float) -> str:
    """Blend toward white. Used instead of alpha, which svglib drops."""
    rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def plot_summary_svg(
    panels: list[tuple[str, dict[str, tuple[list[str], list[list[float]]]]]],
) -> io.BytesIO:
    """Per-document speed ratios to json as box plots, one framed panel per
    operation, stacked under one shared legend.

    *panels* is ``[(title, {locality: (lib_names, ratios_per_lib)})]``, one
    ratio per document. Each library gets one box per locality side by side,
    the first in full color and the rest lighter. Whiskers reach the slowest
    and fastest document, so no document is hidden as an outlier. Each panel
    keeps its own y scale: loads and dumps ratios live on very different
    ranges.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    libs = _order_libs(
        lib for _, by_loc in panels for names, _ in by_loc.values() for lib in names
    )
    lib_colors = assign_colors(libs)
    locs = list(dict.fromkeys(loc for _, by_loc in panels for loc in by_loc))
    n_loc = len(locs)
    shades = [0.0, 0.55, 0.75][:n_loc] + [0.85] * max(0, n_loc - 3)

    W = _FIG_WIDTH_IN
    top_pad, legend_h = 0.04, 0.26
    title_h, plot_h, ticks_h, gap = 0.24, 1.15, 0.22, 0.12
    band = title_h + plot_h + ticks_h
    H = top_pad + legend_h + len(panels) * band + (len(panels) - 1) * gap
    left_w = 0.55
    fig = plt.figure(figsize=(W, H))

    handles = [
        Patch(
            facecolor=_lighten("#7a7a7a", shade),
            edgecolor=_AXIS,
            linewidth=0.6,
            label=f"{loc} cache",
        )
        for loc, shade in zip(locs, shades)
        if loc
    ]
    handles.append(Line2D([0], [0], color=_MEDIAN, lw=2, label="median"))
    handles.append(
        Line2D([0], [0], color=_BASELINE, lw=0.8, ls="--", label="json (1x)")
    )
    _legend_row(fig, handles, 1 - top_pad / H)

    box_w = 0.62 / n_loc
    for p, (title, by_loc) in enumerate(panels):
        band_top = H - top_pad - legend_h - p * (band + gap)
        fig.text(
            0.5,
            band_top / H,
            title,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=_INK,
        )
        ax = fig.add_axes(
            (
                left_w / W,
                (band_top - title_h - plot_h) / H,
                (W - left_w - 0.05) / W,
                plot_h / H,
            )
        )
        for spine in ax.spines.values():
            spine.set_color(_AXIS)
            spine.set_linewidth(0.8)
        ax.set_facecolor("none")

        y_hi = 1.0
        for k, loc in enumerate(locs):
            names, ratio_lists = by_loc.get(loc, ([], []))
            for lib, values in zip(names, ratio_lists):
                x = libs.index(lib) + (k - (n_loc - 1) / 2) * (box_w + 0.03)
                y_hi = max(y_hi, max(values))
                ax.boxplot(
                    [values],
                    positions=[x],
                    widths=box_w,
                    whis=(0, 100),
                    showfliers=False,
                    patch_artist=True,
                    manage_ticks=False,
                    boxprops=dict(
                        facecolor=_lighten(lib_colors[lib], shades[k]),
                        edgecolor=_AXIS,
                        linewidth=0.7,
                    ),
                    whiskerprops=dict(color=_AXIS, linewidth=0.8),
                    capprops=dict(color=_AXIS, linewidth=0.8),
                    medianprops=dict(color=_MEDIAN, linewidth=2),
                )

        ax.set_xlim(-0.6, len(libs) - 0.4)
        ax.set_xticks(range(len(libs)))
        ax.set_xticklabels(libs, fontsize=8.5, color=_INK)
        ax.set_ylim(0, y_hi * 1.06)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter("{x:g}x")
        ax.tick_params(axis="both", labelsize=8, colors=_INK, length=3, width=0.6)
        ax.set_ylabel("Speed ratio to json", fontsize=7.5, color=_INK)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(1.0, color=_BASELINE, linestyle="--", linewidth=0.8, zorder=1)

    return _save_svg(fig)


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


def _collect_ratios(result: BenchmarkFinalResult, cats, index_name: str):
    """Per-document speed ratios to json for one summary panel.

    Each document is one point, the geometric mean of its ratios over the
    panel's groups, so a document with more groups gets no more votes.
    """
    files = result.results.get(index_name, {})
    per_lib: dict[str, list[float]] = {}

    for bench_filename in result.filenames:
        file_result = files.get(bench_filename)
        if file_result is None:
            continue
        doc_ratios: dict[str, list[float]] = {}
        for cat in cats.get(index_name, []):
            target = file_result.targets.get(cat)
            # Groups kept out of the summary (the no-cache-write diagnostic,
            # loads of a document whose compact twin stands in for it) keep
            # their own chart but must not colour the headline summary.
            if target is None or not target.in_summary:
                continue
            for lib_name, lib_result in target.lib_results.items():
                ratio = lib_result.ratio
                if lib_name != "json" and ratio > 0 and math.isfinite(ratio):
                    doc_ratios.setdefault(lib_name, []).append(ratio)
        for lib_name, ratios in doc_ratios.items():
            per_lib.setdefault(lib_name, []).append(statistics.geometric_mean(ratios))

    libs = _order_libs(per_lib)
    return libs, [per_lib[lib] for lib in libs]


def _parse_header(header_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the markdown header into its title and (key, value) rows,
    dropping the markdown code backticks."""
    lines = header_text.splitlines()
    title = lines[0].removeprefix("#").strip()
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        rows.append((key.strip(), value.strip().strip("`").strip()))
    return title, rows


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _header_lines(rows: list[tuple[str, str]], max_chars: int) -> list[str]:
    """Monospace "Key:  value" lines, long values wrapped under the value
    column."""
    key_w = max((len(key) for key, _ in rows), default=0) + 2
    lines = []
    for key, value in rows:
        prefix = f"{key}:".ljust(key_w)
        lines += textwrap.wrap(
            value,
            max_chars,
            initial_indent=prefix,
            subsequent_indent=" " * key_w,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [prefix.rstrip()]
    return lines


def _draw_footer(c: "canvas.Canvas"):
    from reportlab.lib.pagesizes import A4

    page_num = c.getPageNumber()
    c.setFillColor(_INK)
    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(A4[0] - _PAGE_MARGIN, 20, f"{page_num}")
    if page_num == 1:
        prefix = "This report was generated by "
        c.drawString(_PAGE_MARGIN, 20, prefix + _REPO_URL)
        start = _PAGE_MARGIN + c.stringWidth(prefix)
        end = start + c.stringWidth(_REPO_URL)
        c.linkURL(_REPO_URL, (start, 18, end, 27), relative=1)


def _generate_pdf_report(
    sections: list[tuple[str, list[io.BytesIO]]],
    header_text: str,
    output_pdf_path: str,
    summary_svg: io.BytesIO | None,
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

    title, rows = _parse_header(header_text)
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    c.setTitle(title)
    width, height = A4
    avail_w = width - 2 * _PAGE_MARGIN
    top, bottom = height - _PAGE_MARGIN, 36

    def new_page():
        _draw_footer(c)
        c.showPage()
        return top

    def load(svg_io: io.BytesIO):
        svg_io.seek(0)
        drawing = svg2rlg(svg_io, font_map=font_map)
        # Charts are drawn at print size; only shrink something too wide.
        scale = min(1.0, avail_w / drawing.width)
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        return drawing

    def heading(y: float, text: str, next_h: float) -> float:
        """Draw a section heading, starting a new page first if the figure
        *next_h* tall that follows it would not fit under it."""
        if y - 26 - next_h < bottom:
            y = new_page()
        c.setFillColor(_INK)
        c.setFont(_PDF_HEADING_FONT, 14)
        c.drawString(_PAGE_MARGIN, y - 14, text)
        c.setFillColorRGB(0, 0, 0)
        c.bookmarkHorizontal(text, 0, y + 6)
        c.addOutlineEntry(text, text, level=0)
        return y - 14 - 12

    def place(y: float, drawing, rule: bool) -> float:
        if y - drawing.height < bottom:
            y = new_page()
        if rule:
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.setLineWidth(0.5)
            c.line(_PAGE_MARGIN, y + 4, width - _PAGE_MARGIN, y + 4)
        renderPDF.draw(drawing, c, _PAGE_MARGIN, y - drawing.height)
        return y - drawing.height - 12

    # Title and the run's environment, as a monospace block.
    c.setFillColor(_INK)
    c.setFont(_PDF_HEADING_FONT, 20)
    c.drawString(_PAGE_MARGIN, top - 20, title)
    size = 9
    max_chars = int(avail_w // c.stringWidth(" ", _PDF_TEXT_FONT, size))
    text_obj = c.beginText(_PAGE_MARGIN, top - 20 - 24)
    text_obj.setFont(_PDF_TEXT_FONT, size)
    text_obj.setLeading(11.5)
    for line in _header_lines(rows, max_chars):
        text_obj.textLine(line)
    c.drawText(text_obj)
    y = text_obj.getY() - 10
    c.setFillColorRGB(0, 0, 0)

    if summary_svg is not None:
        drawing = load(summary_svg)
        y = heading(y, "TL;DR", drawing.height)
        y = place(y, drawing, rule=False) - 6

    for name, figs in sections:
        drawings = [load(svg) for svg in figs]
        y = heading(y, name, drawings[0].height if drawings else 0)
        for index, drawing in enumerate(drawings):
            y = place(y, drawing, rule=index > 0)
        y -= 8

    _draw_footer(c)
    c.save()
    return output_pdf_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_figures(result: BenchmarkFinalResult, cats, fmt: str):
    """{index name: [(file name, SVG), ...]}, in report order."""
    breakeven = _breakeven(result)
    figures = {}
    for indexed_group, group_cats in cats.items():
        files = result.results.get(indexed_group, {})
        group_figures = figures[indexed_group] = []
        for bench_filename in result.filenames:
            file_result = files.get(bench_filename)
            if file_result is None:
                continue
            print(f"Processing {bench_filename} [{indexed_group}]({fmt})")
            n = breakeven.get((indexed_group, bench_filename))
            note = f"UTF-8 cache break-even: {n:.1f} dumps of one object" if n else ""
            svg = plot_benchmark_svg(
                group_cats, file_result.targets, bench_filename, note=note
            )
            group_figures.append((bench_filename, svg))
    return figures


def _breakeven(result: BenchmarkFinalResult) -> dict[tuple[str, str], float]:
    """Derive, per file, how many dumps of one object it takes for the UTF-8
    cache write to pay for itself.

    Pure arithmetic over numbers already measured, no extra benchmarking:
        N * t_nowrite  ==  t_write + (N - 1) * t_cached

    Two guards. Only the hot locality is used: the cached group re-dumps one
    live object, so pairing it with a cold first-dump would mix two memory
    regimes in one equation. And the write cost must clear its own confidence
    interval -- on files with few non-ASCII strings it is under the noise
    floor, and dividing two indistinguishable numbers gives break-evens below
    1, which is not a physical answer. Such files get no break-even.

    Keyed by (index name, file name), the chart the number belongs on.
    """
    resolved: dict[tuple[str, str], float] = {}
    for index_name, files_dict in result.results.items():
        if split_index_name(index_name)[1] != "hot":
            continue
        for file_name, file_result in files_dict.items():
            ssr = {g: t.lib_results.get("ssrjson") for g, t in file_result.targets.items()}
            w = ssr.get("dumps to bytes")
            nw = ssr.get("dumps to bytes (no cache write)")
            c = ssr.get("dumps to bytes (cached)")
            if not (w and nw and c):
                continue
            # Is writing measurably slower than not writing?
            if not (w.stat > nw.stat and w.stat_lo > nw.stat_hi):
                continue
            # Does reading the cache measurably help? Without that the write
            # can never amortise and there is no crossover.
            if not (nw.stat > c.stat and nw.stat_lo > c.stat_hi):
                continue
            resolved[index_name, file_name] = (w.stat - c.stat) / (nw.stat - c.stat)
    return resolved


def _build_summary(result: BenchmarkFinalResult, cats) -> io.BytesIO | None:
    """The TL;DR: one box-plot panel per operation, localities side by side.

    Operations are not pooled: their ratios to json live on very different
    scales, so a pooled box mostly reports how many points each operation
    happened to contribute. Each document counts once per panel (see
    _collect_ratios).
    """
    panels: dict[str, dict] = {}
    for index_name in cats:
        base, locality = split_index_name(index_name)
        panels.setdefault(base, {})[locality] = _collect_ratios(result, cats, index_name)
    if not panels:
        return None
    return plot_summary_svg(
        [(print_index_name(base), per_loc) for base, per_loc in panels.items()]
    )


def generate_report_pdf(
    result: BenchmarkFinalResult, file: str, out_dir: str | None = None
) -> str:
    _plot_prepare()

    if out_dir is None:
        out_dir = os.getcwd()

    file = file.removesuffix(".json")
    report_name = f"{file}.pdf"

    cats = _get_cats(result)
    figures = _build_figures(result, cats, "PDF")

    return _generate_pdf_report(
        [
            (print_index_name(name), [svg for _, svg in group])
            for name, group in figures.items()
        ],
        header_text=fetch_header(result),
        output_pdf_path=os.path.join(out_dir, report_name),
        summary_svg=_build_summary(result, cats),
    )


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

    cats = _get_cats(result)

    summary = _build_summary(result, cats)
    if summary is not None:
        with open(os.path.join(report_folder, "summary.svg"), "wb") as svg_file:
            svg_file.write(summary.getvalue())
        template += "\n\n## TL;DR\n\n![summary](./summary.svg)\n"

    figures = _build_figures(result, cats, "Markdown")
    for indexed_group, group_figures in figures.items():
        template += f"\n\n## {print_index_name(indexed_group)}\n\n"
        slug = _safe_slug(indexed_group)
        for bench_filename, svg_buf in group_figures:
            svg_name = f"{bench_filename}_{slug}.svg"
            with open(os.path.join(report_folder, svg_name), "wb") as svg_file:
                svg_file.write(svg_buf.getvalue())
            template += f"![{bench_filename}_{slug}](./{svg_name})\n\n"

    ret = os.path.join(report_folder, report_name)
    with open(ret, "w") as f:
        f.write(template)
    return ret
