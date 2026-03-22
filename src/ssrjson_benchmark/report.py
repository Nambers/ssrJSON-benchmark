import io
import os
from typing import TYPE_CHECKING

from .benchmark import (
    INDEXED_GROUPS,
    PRINT_INDEX_GROUPS,
    _get_benchmark_libraries,
    _NAME_DUMPSTOBYTES,
    fetch_header,
)
from .result_types import BenchmarkFinalResult, BenchmarkResultPerFileTarget

if TYPE_CHECKING:
    from reportlab.pdfgen import canvas

_PDF_HEADING_FONT = "Helvetica-Bold"
_PDF_TEXT_FONT = "Courier"

# ssrjson's color is always fixed
_SSRJSON_COLOR = "#fd8d3c"

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
    ):
        self.bar_width = bar_width
        self.fig_width_per_cat = fig_width_per_cat
        self.fig_height = fig_height
        self.show_std_dev = show_std_dev
        self.title_fontsize = title_fontsize
        self.ratio_fontsize = ratio_fontsize
        self.gbps_fontsize = gbps_fontsize


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
    mask: list[bool] | None = None,
    config: PlotConfig | None = None,
) -> io.BytesIO:
    """Generate an SVG bar chart for a single file's benchmark results.

    Args:
        categories: list of benchmark group names to show
        data: mapping from group_name -> BenchmarkResultPerFileTarget
            (this is typically the .targets dict of a BenchmarkResultPerFile,
             or the BenchmarkResultPerFile itself accessed by group_name)
        doc_name: title for the chart
        mask: optional boolean mask for categories
        config: optional PlotConfig
    Returns:
        BytesIO containing the SVG
    """
    import matplotlib.pyplot as plt

    if config is None:
        config = PlotConfig()
    if mask is None:
        mask = [True] * len(categories)

    # Determine libraries from first visible category
    libs = []
    lib_colors = {}
    for i, cat in enumerate(categories):
        if mask[i] and cat in data:
            target = (
                data[cat]
                if isinstance(data[cat], BenchmarkResultPerFileTarget)
                else data[cat]
            )
            libs = target.libraries
            lib_colors = assign_colors(libs)
            break

    if not libs:
        # Fallback: empty plot
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    colors = [lib_colors.get(n, "#999999") for n in libs]
    n = len(categories)
    bar_width = config.bar_width

    fig, axs = plt.subplots(
        1,
        n,
        figsize=(config.fig_width_per_cat * n, config.fig_height),
        sharey=False,
        tight_layout=True,
        gridspec_kw={"wspace": 0},
    )
    if n == 1:
        axs = [axs]

    x_positions = [i * (bar_width) for i in range(len(libs))]

    for i, (ax, cat) in enumerate(zip(axs, categories)):
        if not mask[i] or cat not in data:
            ax.axis("off")
            continue

        target = data[cat]
        if isinstance(target, dict):
            # backward compat: dict access
            vals = []
            std_devs = []
            for name in libs:
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
            for name in libs:
                if name in target:
                    lib_result = target[name]
                    vals.append(lib_result.ratio)
                    # Convert std_dev from ns to ratio scale for display
                    # std_dev is in ns, ratio = baseline_speed / this_speed
                    # We show the coefficient of variation on the bar
                    std_devs.append(lib_result.std_dev)
                else:
                    vals.append(1.0)
                    std_devs.append(0.0)
            ssrjson_bps = target.ssrjson_bytes_per_sec

        gbps = ssrjson_bps / (1024**3) if ssrjson_bps else 0.0

        # Compute ratio std dev for display (CV * ratio)
        # CV = std_dev / mean_per_iteration = std_dev * repeat_count / speed
        ratio_errors = []
        for j_idx, name in enumerate(libs):
            if name in target and not isinstance(target, dict):
                lib_result = target[name]
                if (
                    lib_result.speed > 0
                    and lib_result.std_dev > 0
                    and lib_result.repeat_count > 0
                ):
                    cv = lib_result.std_dev * lib_result.repeat_count / lib_result.speed
                    ratio_errors.append(vals[j_idx] * cv)
                else:
                    ratio_errors.append(0)
            else:
                ratio_errors.append(0)

        for xi, val, col, err in zip(x_positions, vals, colors, ratio_errors):
            ax.bar(xi, val, width=bar_width, color=col)

            # Show std dev whisker if enabled and significant
            if config.show_std_dev and err > 0:
                ax.errorbar(
                    xi,
                    val,
                    yerr=err,
                    fmt="none",
                    ecolor="#333333",
                    capsize=2,
                    capthick=0.8,
                    linewidth=0.8,
                )

            # Ratio label
            label_y = val + max(err, 0) + 0.05
            ax.text(
                xi,
                label_y,
                f"{val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=config.ratio_fontsize,
                color=_get_ratio_color(val),
            )
            # CV percentage (±X.X%) below the ratio label
            if config.show_std_dev and err > 0 and val > 0:
                cv_pct = err / val * 100
                ax.text(
                    xi,
                    label_y - 0.02,
                    f"\u00b1{cv_pct:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=config.ratio_fontsize - 2,
                    color="#888888",
                )

        if "ssrjson" in libs and gbps > 0:
            ssrjson_index = libs.index("ssrjson")
            ax.text(
                x_positions[ssrjson_index],
                vals[ssrjson_index] / 2,
                f"{gbps:.2f} GB/s",
                ha="center",
                va="center",
                fontsize=config.gbps_fontsize,
                color="#2c3e50",
                fontweight="bold",
            )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        max_val_with_err = max(v + e for v, e in zip(vals, ratio_errors))
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
    plt.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------


def plot_distribution_svg(
    ratio_distr: list[list[float]],
    lib_names: list[str],
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
    ax.set_title("Speed Ratio Distribution per Library")

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


def _get_cats_and_masks(result: BenchmarkFinalResult):
    """Compute categories per index group and ASCII masks."""
    benchmark_groups = _get_benchmark_libraries()
    cats = [
        [a for a in result.categories if benchmark_groups[a].index_name == index_name]
        for index_name in INDEXED_GROUPS
    ]
    masks = [
        [
            benchmark_groups[a].index_name != _NAME_DUMPSTOBYTES
            or not benchmark_groups[a].skip_when_ascii
            for a in cats[i]
        ]
        for i in range(len(INDEXED_GROUPS))
    ]
    return cats, masks, benchmark_groups


def _get_non_baseline_libs(result: BenchmarkFinalResult) -> list[str]:
    """Get the list of non-baseline library names from the result data."""
    for index_name in INDEXED_GROUPS:
        if index_name in result.results:
            for filename in result.filenames:
                if filename in result.results[index_name]:
                    file_result = result.results[index_name][filename]
                    for target in file_result.targets.values():
                        non_baseline = [l for l in target.libraries if l != "json"]
                        if non_baseline:
                            return non_baseline
    return []


def _collect_ratios(result: BenchmarkFinalResult, cats, masks):
    """Collect ratio distributions for non-baseline libraries."""
    non_baseline_libs = _get_non_baseline_libs(result)
    ratios = {lib: [] for lib in non_baseline_libs}

    for i, indexed_group in enumerate(INDEXED_GROUPS):
        if indexed_group not in result.results:
            continue
        for bench_filename in result.filenames:
            if bench_filename not in result.results[indexed_group]:
                continue
            file_result = result.results[indexed_group][bench_filename]
            is_ascii = file_result.pyunicode_is_ascii
            mask = masks[i] if is_ascii else [True] * len(cats[i])
            for j, cat in enumerate(cats[i]):
                if not mask[j]:
                    continue
                if cat not in file_result.targets:
                    continue
                target = file_result.targets[cat]
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
    header_text: str,
    output_pdf_path: str,
    distribution_svg: io.BytesIO,
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

    for i in range(len(INDEXED_GROUPS)):
        name = PRINT_INDEX_GROUPS[i]
        figs = figures[i]

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


def generate_report_pdf(
    result: BenchmarkFinalResult, file: str, out_dir: str | None = None
) -> str:
    _plot_prepare()

    if out_dir is None:
        out_dir = os.getcwd()

    file = file.removesuffix(".json")
    report_name = f"{file}.pdf"

    figures = [[] for _ in range(len(INDEXED_GROUPS))]
    cats, masks, benchmark_groups = _get_cats_and_masks(result)

    for i, indexed_group in enumerate(INDEXED_GROUPS):
        if indexed_group not in result.results:
            continue
        for bench_filename in result.filenames:
            if bench_filename not in result.results[indexed_group]:
                continue
            print(f"Processing {bench_filename} [{indexed_group}](PDF)")
            file_result = result.results[indexed_group][bench_filename]
            if file_result.pyunicode_is_ascii:
                mask = masks[i]
            else:
                mask = [True] * len(cats[i])
            figures[i].append(
                plot_benchmark_svg(cats[i], file_result.targets, bench_filename, mask)
            )

    non_baseline_libs, ratio_lists = _collect_ratios(result, cats, masks)
    dist_svg = plot_distribution_svg(ratio_lists, non_baseline_libs)

    template = fetch_header(result)
    out_path = _generate_pdf_report(
        figures,
        header_text=template,
        output_pdf_path=os.path.join(out_dir, report_name),
        distribution_svg=dist_svg,
    )
    return out_path


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

    cats, masks, benchmark_groups = _get_cats_and_masks(result)

    for i, indexed_group in enumerate(INDEXED_GROUPS):
        template += f"\n\n## {PRINT_INDEX_GROUPS[i]}\n\n"
        if indexed_group not in result.results:
            continue
        for bench_filename in result.filenames:
            if bench_filename not in result.results[indexed_group]:
                continue
            print(f"Processing {bench_filename} [{indexed_group}](Markdown)")
            file_result = result.results[indexed_group][bench_filename]
            if file_result.pyunicode_is_ascii:
                mask = masks[i]
            else:
                mask = [True] * len(cats[i])
            svg_buf = plot_benchmark_svg(
                cats[i], file_result.targets, bench_filename, mask
            )
            svg_path = os.path.join(
                report_folder, f"{bench_filename}_{indexed_group}.svg"
            )
            with open(svg_path, "wb") as svg_file:
                svg_file.write(svg_buf.getvalue())
            template += f"![{bench_filename}_{indexed_group}](./{bench_filename}_{indexed_group}.svg)\n\n"

    non_baseline_libs, ratio_lists = _collect_ratios(result, cats, masks)
    dist_svg = plot_distribution_svg(ratio_lists, non_baseline_libs)
    with open(os.path.join(report_folder, "ratio_distribution.svg"), "wb") as svg_file:
        svg_file.write(dist_svg.getvalue())
    template = template.replace(
        "TLDRIMGPLACEHOLDER", "![ratio_distribution](./ratio_distribution.svg)"
    )

    ret = os.path.join(report_folder, report_name)
    with open(ret, "w") as f:
        f.write(template)
    return ret
