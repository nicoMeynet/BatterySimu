#!/usr/bin/env python3
"""
Build a PDF report from exported notebook graphs.

Example:
  python generate_pdf_report.py \
    --global out/graphs/global/*.png \
    --monthly out/graphs/monthly/*.png \
    --seasonal out/graphs/seasonal/*.png \
    --output out/battery_graph_report.pdf \
    --title "Battery Simulation Graph Report"
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DOC_DEFAULTS_DIR = Path(__file__).resolve().parent / "doc" / "pdf_report"


def load_required_text_from_doc(filename: str) -> str:
    path = DOC_DEFAULTS_DIR / filename
    try:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"Missing required documentation text file: {path}. "
                "Files under doc/pdf_report are mandatory."
            )
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read required documentation text file: {path}") from exc
    if not text:
        raise ValueError(f"Required documentation text file is empty: {path}")
    return text


DEFAULT_INTRO = load_required_text_from_doc("default_intro.md")
DEFAULT_METHODOLOGY = load_required_text_from_doc("default_methodology.md")
DEFAULT_SCOPE = load_required_text_from_doc("default_scope.md")
DEFAULT_DATA_REQUIREMENTS = load_required_text_from_doc("default_data_requirements.md")
COPYRIGHT_SHORT = load_required_text_from_doc("copyright_short.txt")
LICENSE_SHORT = load_required_text_from_doc("license_short.txt")


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_caption(path: Path) -> str:
    name = path.stem
    # Drop numeric ordering prefix (e.g. "01_")
    name = name.lstrip("0123456789").lstrip("_- ")
    return name.replace("_", " ").replace("-", " ").strip().title()


def graph_context_notes(image_path: Path) -> tuple[str, str, str]:
    parent = image_path.parent.name.lower()
    if "global" in parent:
        return (
            "Global (full-year aggregate) view.",
            "Read across battery scenarios to compare yearly totals; when a right axis is present, use it as a percentage reference and not as CHF.",
            "Source: battery_comparison_global.ipynb + out/simulation_json/config_*.json",
        )
    if "month" in parent:
        return (
            "Monthly view (Jan-Dec) across analyzed battery scenarios.",
            "Read month-to-month seasonality first, then compare scenario levels/shapes to see where the battery improves performance and where gains flatten.",
            "Source: battery_comparison_month.ipynb + out/simulation_json/config_*.json",
        )
    if "season" in parent:
        return (
            "Seasonal sizing view across battery capacities.",
            "X-axis is total battery size (kWh across phases) and each line represents a season; look for diminishing returns (knee points) as size increases.",
            "Source: battery_comparison_season.ipynb + out/simulation_json/config_*.json",
        )
    return (
        "Comparison view across simulated battery scenarios.",
        "Compare relative levels and trends between scenarios using the axis units shown in the chart.",
        "Source: notebook export + out/simulation_json/",
    )


def graph_description(image_path: Path, *, compact: bool = False) -> str:
    key = image_path.stem.lower()
    key = key.lstrip("0123456789").lstrip("_- ")
    view_text, read_text, source_text = graph_context_notes(image_path)

    metric_text = "This chart compares the selected metric across simulated battery scenarios."
    if "energy financial impact" in key or "financial impact" in key:
        metric_text = (
            "This chart shows financial impact split by energy channel (import/consumption-side and export/injection-side) in CHF, with a percentage axis when present."
        )
    elif "rentability overview" in key:
        metric_text = (
            "This chart summarizes high-level financial KPIs (gain, bill impact, payback/rentability indicators) and is useful as a first screening before reading detailed charts."
        )
    elif "battery utilization" in key:
        metric_text = (
            "This chart summarizes yearly battery usage intensity (for example throughput, cycling and activity) to show how strongly each configuration is used."
        )
    elif "status heatmap" in key or "heatmap" in key:
        metric_text = (
            "This heatmap shows relative metric intensity across scenarios/periods using color scale encoding; compare cells by color first, then confirm exact labels/axes."
        )
    elif "energy reduction" in key:
        metric_text = (
            "This chart shows yearly energy reduction effects in kWh versus the no-battery baseline (and may include a right-axis percentage of total consumption when available)."
        )
    elif "net financial gain" in key:
        metric_text = "Higher CHF values indicate larger net financial gain versus the no-battery baseline."
    elif "grid import reduction" in key:
        metric_text = "Higher kWh values indicate more grid import avoided thanks to battery charging/discharging behavior."
    elif "grid export reduction" in key:
        metric_text = "Higher kWh values indicate more surplus energy retained/shifted instead of being exported to the grid."
    elif "energy shifting" in key or "energy flow" in key:
        metric_text = (
            "This chart compares battery charging/discharging or import/export reduction channels to show how energy is shifted through storage."
        )
    elif "throughput" in key:
        metric_text = (
            "Higher throughput means more energy cycled through the battery; this often improves savings but also indicates higher utilization."
        )
    elif "equivalent full battery cycles" in key or "equivalent full cycles" in key:
        metric_text = "Equivalent full cycles estimate how intensively the battery is used relative to its capacity."
    elif "full saturation" in key or "battery full" in key or "full soc" in key:
        metric_text = (
            "Higher values mean the battery is full more often, which can indicate oversizing if this remains high across many periods."
        )
    elif "empty limitation" in key or "battery empty" in key or "empty soc" in key:
        metric_text = (
            "Higher values mean the battery is empty more often, which is a typical undersizing indicator during demand peaks."
        )
    elif "structural energy undersizing" in key:
        metric_text = (
            "Higher percentages indicate more days where battery energy capacity was insufficient to cover the intended shifting/coverage objective."
        )
    elif "evening energy undersizing" in key:
        metric_text = (
            "Higher percentages indicate more peak/evening periods with insufficient stored energy, which is critical for comfort and tariff optimization."
        )
    elif "power saturation at max limit" in key or "active power saturation" in key:
        metric_text = (
            "Higher percentages indicate more time constrained by maximum charge/discharge power, pointing to power-limit bottlenecks rather than energy-capacity limits."
        )
    elif "idle power limited" in key or "idle missed opportunities" in key:
        metric_text = (
            "Higher percentages indicate more time where the system was idle but could have acted if charge/discharge power limits were higher."
        )
    elif "power state distribution" in key:
        metric_text = (
            "This chart shows the share of time spent charging, discharging and idle states to understand operating balance over the analyzed period."
        )
    elif "activity duration" in key:
        metric_text = (
            "This chart compares charging vs discharging duration to show how often the battery is active and whether usage is balanced."
        )
    elif "undersizing" in key:
        metric_text = "Higher values indicate more structural undersizing events (battery energy capacity insufficient for the demand pattern)."
    elif "power saturation" in key:
        metric_text = "Higher values indicate more time limited by charge/discharge power constraints."

    if compact:
        return f"{metric_text} {read_text}"
    return f"{view_text} {metric_text} {read_text}"


def graph_source_text(image_path: Path) -> str:
    _, _, source_text = graph_context_notes(image_path)
    return source_text


def normalize_power(values: list[int | float] | None) -> str:
    if not values:
        return "-"
    if all(v == values[0] for v in values):
        return str(values[0])
    return "/".join(str(v) for v in values)


def extract_config_payload(data: dict) -> dict:
    if "battery" in data and "tariff" in data:
        return data
    if "battery" in data:
        return data
    if "configuration" in data and "battery" in data["configuration"] and "tariff" in data["configuration"]:
        return data["configuration"]
    raise ValueError("Unsupported config schema (expected config JSON or simulation output JSON).")


def load_battery_configuration_rows(config_paths: list[Path]) -> list[dict]:
    rows = []
    for path in sorted(config_paths):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = extract_config_payload(raw)

        battery = cfg["battery"]
        capacity_per_phase = battery.get("capacity_Wh_per_phase", [0, 0, 0])
        total_kwh = sum(capacity_per_phase) / 1000 if capacity_per_phase else 0

        rows.append(
            {
                "scenario": path.stem.replace("config_", ""),
                "fields": [
                    ("Capacity / phase (Wh)", "/".join(str(x) for x in capacity_per_phase)),
                    ("Total capacity (kWh)", f"{total_kwh:.2f}"),
                    ("Battery cost (CHF)", str(battery.get("cost_chf", "-"))),
                    (
                        "Charge / discharge power (W)",
                        (
                            f"{normalize_power(battery.get('max_charge_power_watts'))} / "
                            f"{normalize_power(battery.get('max_discharge_power_watts'))}"
                        ),
                    ),
                    (
                        "Charge / discharge efficiency",
                        (
                            f"{battery.get('charge_efficiency', '-')} / "
                            f"{battery.get('discharge_efficiency', '-')}"
                        ),
                    ),
                    ("SOC min / max (%)", f"{battery.get('soc_min_pct', '-')} / {battery.get('soc_max_pct', '-')}"),
                    ("Max cycles", str(battery.get("max_cycles", "-"))),
                ],
            }
        )
    return rows


def load_energy_tariff_configuration_rows(config_paths: list[Path]) -> list[dict]:
    if not config_paths:
        return []

    first_path = sorted(config_paths)[0]
    with open(first_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = extract_config_payload(raw)

    tariff = cfg.get("tariff")
    if tariff is None:
        tariff_path = first_path.parent / "energy_tariff.json"
        with open(tariff_path, "r", encoding="utf-8") as f:
            tariff_wrapper = json.load(f)
        tariff = tariff_wrapper.get("tariff", tariff_wrapper)

    peak = tariff.get("peak", {})
    off_peak = tariff.get("off_peak", {})
    return [
        {
            "scenario": "energy_tariff",
            "fields": [
                ("Peak consume tariff", str(peak.get("tariff_consume", "-"))),
                ("Peak inject tariff", str(peak.get("tariff_inject", "-"))),
                ("Peak days", "/".join(str(x) for x in peak.get("days", [])) or "-"),
                ("Peak hours", "/".join(str(x) for x in peak.get("hours", [])) or "-"),
                ("Off-peak consume tariff", str(off_peak.get("tariff_consume", "-"))),
                ("Off-peak inject tariff", str(off_peak.get("tariff_inject", "-"))),
            ],
        }
    ]


def load_input_data_summary(simulation_paths: list[Path], configuration_count: int) -> dict | None:
    if not simulation_paths:
        return None

    starts = []
    ends = []
    total_points = 0
    phase_count = None

    for path in sorted(simulation_paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        battery_cfg = data.get("configuration", {}).get("battery", {})
        cap = battery_cfg.get("capacity_Wh_per_phase")
        if isinstance(cap, list) and phase_count is None:
            phase_count = len(cap)

        for month in data.get("months", []):
            r = month.get("range", {})
            start = r.get("start_date")
            end = r.get("end_date")
            samples = r.get("samples_analyzed")

            if start:
                starts.append(start)
            if end:
                ends.append(end)
            if isinstance(samples, (int, float)):
                total_points += int(samples)

    if not starts or not ends:
        return None

    return {
        "number_of_phases": phase_count if phase_count is not None else 3,
        "date_from": min(starts),
        "date_to": max(ends),
        "number_of_points": total_points,
        "number_of_configurations": configuration_count,
    }


def draw_cover(pdf: PdfPages, title: str, subtitle: str | None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f6f7fb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.5,
        0.75,
        title,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#1f2937",
    )
    if subtitle:
        ax.text(
            0.5,
            0.69,
            subtitle,
            ha="center",
            va="center",
            fontsize=13,
            color="#4b5563",
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(
        0.5,
        0.63,
        f"Generated on {generated}",
        ha="center",
        va="center",
        fontsize=11,
        color="#6b7280",
    )

    ax.add_patch(
        plt.Rectangle(
            (0.15, 0.61),
            0.70,
            0.002,
            transform=ax.transAxes,
            color="#d1d5db",
        )
    )
    ax.text(0.5, 0.06, COPYRIGHT_SHORT, ha="center", va="center", fontsize=9, color="#6b7280")
    ax.text(0.5, 0.04, LICENSE_SHORT, ha="center", va="center", fontsize=8.5, color="#9ca3af")

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def draw_toc_page(
    pdf: PdfPages,
    entries: list[tuple[str, int]],
    page_index: int,
    total_pages: int,
) -> None:
    """Render a table of contents page with chapter start pages."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.06,
        0.965,
        "Table of Contents",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#111827",
        transform=ax.transAxes,
    )
    ax.text(
        0.94,
        0.965,
        f"Page {page_index}/{total_pages}",
        ha="right",
        va="top",
        fontsize=10,
        color="#6b7280",
        transform=ax.transAxes,
    )

    y = 0.90
    if not entries:
        ax.text(
            0.08,
            y,
            "No chapters available.",
            ha="left",
            va="top",
            fontsize=12,
            color="#6b7280",
            transform=ax.transAxes,
        )
        y -= 0.06
    for idx, (title, start_page) in enumerate(entries, start=1):
        left = f"{idx}. {title}"
        right = str(start_page)
        ax.text(0.08, y, left, ha="left", va="top", fontsize=12, color="#374151", transform=ax.transAxes)
        ax.text(0.92, y, right, ha="right", va="top", fontsize=12, color="#111827", fontweight="bold", transform=ax.transAxes)
        ax.text(0.08, y - 0.012, "·" * 120, ha="left", va="top", fontsize=8, color="#e5e7eb", transform=ax.transAxes)
        y -= 0.06
        if y < 0.12:
            break

    ax.text(0.06, 0.025, COPYRIGHT_SHORT, ha="left", va="bottom", fontsize=8.5, color="#9ca3af", transform=ax.transAxes)
    ax.text(0.94, 0.025, "License: CC BY-NC 4.0", ha="right", va="bottom", fontsize=8.5, color="#9ca3af", transform=ax.transAxes)

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def draw_structured_text_page(
    pdf: PdfPages,
    title: str,
    page_lines: list[tuple[str, str]],
    is_continuation: bool,
    page_index: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.06,
        0.965,
        f"{title} (cont.)" if is_continuation else title,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.94,
        0.965,
        f"Page {page_index}/{total_pages}",
        ha="right",
        va="top",
        fontsize=10,
        color="#6b7280",
    )

    x = 0.08
    y = 0.88
    line_h = 0.022

    for kind, text in page_lines:
        if kind == "blank":
            y -= 0.012
            continue
        if kind == "h2":
            ax.text(
                x,
                y,
                text,
                ha="left",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                color="#111827",
            )
            y -= line_h * 1.2
            continue
        if kind == "h3":
            ax.text(
                x,
                y,
                text,
                ha="left",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                color="#1f2937",
            )
            y -= line_h * 1.1
            continue
        if kind == "bullet":
            ax.text(x, y, f"• {text}", ha="left", va="top", fontsize=11, color="#374151")
            y -= line_h
            continue
        if kind == "bullet_cont":
            ax.text(x + 0.02, y, text, ha="left", va="top", fontsize=11, color="#374151")
            y -= line_h
            continue
        ax.text(x, y, text, ha="left", va="top", fontsize=11, color="#374151")
        y -= line_h

    ax.text(0.06, 0.025, COPYRIGHT_SHORT, ha="left", va="bottom", fontsize=8.5, color="#9ca3af")
    ax.text(0.94, 0.025, "License: CC BY-NC 4.0", ha="right", va="bottom", fontsize=8.5, color="#9ca3af")

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def parse_structured_text(body_text: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for raw_line in body_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(("blank", ""))
            continue
        if line.startswith("## "):
            lines.append(("h2", line[3:].strip()))
            continue
        if line.startswith("### "):
            lines.append(("h3", line[4:].strip()))
            continue
        if line.startswith("- "):
            wrapped = textwrap.wrap(line[2:].strip(), width=92)
            if wrapped:
                lines.append(("bullet", wrapped[0]))
                for cont in wrapped[1:]:
                    lines.append(("bullet_cont", cont))
            continue
        for wline in textwrap.wrap(line, width=95):
            lines.append(("text", wline))
    return lines


def line_height_units(kind: str) -> float:
    if kind == "blank":
        return 0.012
    if kind == "h2":
        return 0.022 * 1.2
    if kind == "h3":
        return 0.022 * 1.1
    return 0.022


def paginate_structured_lines(
    structured_lines: list[tuple[str, str]],
    *,
    start_y: float = 0.88,
    min_y: float = 0.08,
) -> list[list[tuple[str, str]]]:
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    y = start_y

    for entry in structured_lines:
        h = line_height_units(entry[0])
        if y - h < min_y and current:
            pages.append(current)
            current = []
            y = start_y
        current.append(entry)
        y -= h

    if current:
        pages.append(current)
    return pages


def draw_config_cards_page(
    pdf: PdfPages,
    section_title: str,
    rows: list[dict],
    page_index: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.04,
        0.95,
        section_title,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.96,
        0.95,
        f"Page {page_index}/{total_pages}",
        ha="right",
        va="top",
        fontsize=10,
        color="#6b7280",
    )

    # Up to 2 readable cards per page.
    slots = [(0.05, 0.53, 0.90, 0.37), (0.05, 0.10, 0.90, 0.37)]
    for slot, row in zip(slots, rows):
        x, y, w, h = slot

        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.0,
                edgecolor="#d1d5db",
                facecolor="#f9fafb",
                transform=ax.transAxes,
            )
        )

        ax.text(
            x + 0.02,
            y + h - 0.04,
            row["scenario"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#111827",
        )

        row_y = y + h - 0.085
        for label, value in row["fields"]:
            ax.text(
                x + 0.02,
                row_y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#374151",
                fontweight="bold",
            )
            ax.text(
                x + 0.48,
                row_y,
                str(value),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#111827",
            )
            row_y -= 0.03

    ax.text(0.06, 0.025, COPYRIGHT_SHORT, ha="left", va="bottom", fontsize=8.5, color="#9ca3af")
    ax.text(0.94, 0.025, "License: CC BY-NC 4.0", ha="right", va="bottom", fontsize=8.5, color="#9ca3af")

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def draw_input_data_page(
    pdf: PdfPages,
    summary: dict,
    page_index: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.04,
        0.95,
        "Input Data Ingested",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.96,
        0.95,
        f"Page {page_index}/{total_pages}",
        ha="right",
        va="top",
        fontsize=10,
        color="#6b7280",
    )

    ax.add_patch(
        plt.Rectangle(
            (0.08, 0.58),
            0.84,
            0.26,
            linewidth=1.0,
            edgecolor="#d1d5db",
            facecolor="#f9fafb",
            transform=ax.transAxes,
        )
    )

    rows = [
        ("Number of phases", str(summary["number_of_phases"])),
        ("Date range (from)", str(summary["date_from"])),
        ("Date range (to)", str(summary["date_to"])),
        ("Number of points", str(summary["number_of_points"])),
        ("Number of configurations", str(summary["number_of_configurations"])),
    ]

    y = 0.80
    for label, value in rows:
        ax.text(0.11, y, label, ha="left", va="top", fontsize=11, color="#374151", fontweight="bold")
        ax.text(0.52, y, value, ha="left", va="top", fontsize=11, color="#111827")
        y -= 0.045

    ax.text(0.06, 0.025, COPYRIGHT_SHORT, ha="left", va="bottom", fontsize=8.5, color="#9ca3af")
    ax.text(0.94, 0.025, "License: CC BY-NC 4.0", ha="right", va="bottom", fontsize=8.5, color="#9ca3af")

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def draw_image_page(
    pdf: PdfPages,
    section_title: str,
    images: list[Path],
    page_index: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis("off")
    fig_w, fig_h = fig.get_size_inches()

    ax_bg.text(
        0.06,
        0.965,
        section_title,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax_bg.text(
        0.94,
        0.965,
        f"Page {page_index}/{total_pages}",
        ha="right",
        va="top",
        fontsize=10,
        color="#6b7280",
    )

    if len(images) == 1:
        block_slots = [(0.06, 0.09, 0.88, 0.82)]
    else:
        block_slots = [(0.06, 0.52, 0.88, 0.43), (0.06, 0.05, 0.88, 0.43)]

    for slot, image_path in zip(block_slots, images):
        left, bottom, width, height = slot
        caption_left = left + 0.006
        caption_width = max(0.1, width - 0.012)
        title_y = bottom + height - 0.010

        title_text = clean_caption(image_path)
        source_text = graph_source_text(image_path)

        is_single = len(images) == 1
        desc_text = graph_description(image_path, compact=not is_single)
        if is_single:
            title_fontsize = 12.8
            desc_fontsize = 10.2
            source_fontsize = 9.1
        else:
            title_fontsize = 11.7
            desc_fontsize = 9.2
            source_fontsize = 8.4

        # Wrap caption text so longer explanations remain readable without overlapping the image.
        if is_single:
            title_wrap_width = 68
            desc_wrap_width = 100
            source_wrap_width = 104
        else:
            title_wrap_width = 72
            desc_wrap_width = 104
            source_wrap_width = 108
        title_lines = textwrap.wrap(title_text, width=title_wrap_width) or [title_text]
        desc_lines = textwrap.wrap(desc_text, width=desc_wrap_width) or [desc_text]
        source_lines = textwrap.wrap(source_text, width=source_wrap_width) or [source_text]
        title_render = "\n".join(title_lines)
        desc_render = "\n".join(desc_lines)
        source_render = "\n".join(source_lines)

        if is_single:
            title_line_h = 0.018
            desc_line_h = 0.0138
            source_line_h = 0.0128
        else:
            title_line_h = 0.0172
            desc_line_h = 0.0132
            source_line_h = 0.0120
        title_block_h = title_line_h * len(title_lines)
        desc_top_y = title_y - title_block_h - 0.006
        desc_block_h = desc_line_h * len(desc_lines)
        source_top_y = desc_top_y - desc_block_h - 0.004
        source_block_h = source_line_h * len(source_lines)

        # Draw title/description outside of the image area to avoid overlap.
        ax_bg.text(
            caption_left,
            title_y,
            title_render,
            ha="left",
            va="top",
            fontsize=title_fontsize,
            fontweight="bold",
            color="#374151",
            linespacing=1.15,
            clip_on=True,
        )
        ax_bg.text(
            caption_left,
            desc_top_y,
            desc_render,
            ha="left",
            va="top",
            fontsize=desc_fontsize,
            color="#6b7280",
            linespacing=1.18,
            clip_on=True,
        )
        ax_bg.text(
            caption_left,
            source_top_y,
            source_render,
            ha="left",
            va="top",
            fontsize=source_fontsize,
            color="#9ca3af",
            linespacing=1.15,
            clip_on=True,
        )

        img = mpimg.imread(image_path)
        img_h = int(img.shape[0]) if hasattr(img, "shape") and len(img.shape) >= 2 else 1
        img_w = int(img.shape[1]) if hasattr(img, "shape") and len(img.shape) >= 2 else 1
        img_ratio = img_w / max(img_h, 1)  # width / height

        # Available area under description text.
        available_top = source_top_y - source_block_h - 0.010
        available_bottom = bottom
        max_width = caption_width
        max_height = max(0.05, available_top - available_bottom)

        # Size axes so image keeps its original aspect ratio without added top whitespace.
        height_if_full_width = max_width * fig_w / (img_ratio * fig_h)
        if height_if_full_width <= max_height:
            ax_width = max_width
            ax_height = height_if_full_width
        else:
            ax_height = max_height
            ax_width = ax_height * img_ratio * fig_h / fig_w

        ax_left = caption_left + (max_width - ax_width) / 2
        ax_bottom = available_top - ax_height  # top-align image under text

        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        ax.axis("off")
        ax.imshow(img, aspect="auto")

    ax_bg.text(
        0.06,
        0.03,
        f"Battery simulation notebook export | {COPYRIGHT_SHORT}",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#9ca3af",
    )
    ax_bg.text(
        0.94,
        0.03,
        "License: CC BY-NC 4.0",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#9ca3af",
    )

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


def validate_images(paths: list[str], section_name: str) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"{section_name} image not found: {raw}")
        if not p.is_file():
            raise ValueError(f"{section_name} entry is not a file: {raw}")
        resolved.append(p)
    return sorted(resolved)


def validate_json_paths(paths: list[str], section_name: str) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"{section_name} file not found: {raw}")
        if not p.is_file():
            raise ValueError(f"{section_name} entry is not a file: {raw}")
        resolved.append(p)
    return sorted(resolved)


def build_pdf(
    output_pdf: Path,
    title: str,
    subtitle: str | None,
    intro_text: str,
    methodology_text: str,
    scope_text: str,
    data_requirements_text: str,
    recommendation_text: str,
    license_text: str,
    input_data_summary: dict | None,
    battery_configuration_rows: list[dict],
    energy_tariff_configuration_rows: list[dict],
    global_images: list[Path],
    monthly_images: list[Path],
    seasonal_images: list[Path],
) -> None:
    section_specs = [
        ("Global Graphs", global_images, 2),
        ("Seasonal Graphs", seasonal_images, 2),
        ("Monthly Graphs", monthly_images, 2),
    ]
    intro_pages = paginate_structured_lines(parse_structured_text(intro_text)) if intro_text.strip() else []
    methodology_pages = paginate_structured_lines(parse_structured_text(methodology_text)) if methodology_text.strip() else []
    scope_pages = paginate_structured_lines(parse_structured_text(scope_text)) if scope_text.strip() else []
    data_requirements_pages = (
        paginate_structured_lines(parse_structured_text(data_requirements_text))
        if data_requirements_text.strip()
        else []
    )
    recommendation_pages = (
        paginate_structured_lines(parse_structured_text(recommendation_text))
        if recommendation_text.strip()
        else []
    )
    license_pages = (
        paginate_structured_lines(parse_structured_text(license_text))
        if license_text.strip()
        else []
    )
    battery_config_page_count = len(list(chunked(battery_configuration_rows, 2))) if battery_configuration_rows else 0
    energy_tariff_page_count = (
        len(list(chunked(energy_tariff_configuration_rows, 2))) if energy_tariff_configuration_rows else 0
    )
    input_data_page_count = 1 if input_data_summary else 0
    section_page_counts = [len(list(chunked(images, per_page))) for _, images, per_page in section_specs]
    toc_page_count = 1
    chapter_counts = [
        ("Introduction", len(intro_pages)),
        ("Report Scope", len(scope_pages)),
        ("AI Recommendation", len(recommendation_pages)),
        ("Input Data Ingested", input_data_page_count),
        *[(section_title, count) for (section_title, _, _), count in zip(section_specs, section_page_counts)],
        ("Simulation Methodology", len(methodology_pages)),
        ("Data Requirements", len(data_requirements_pages)),
        ("Battery Configuration Used", battery_config_page_count),
        ("Energy Tariff Configuration Used", energy_tariff_page_count),
        ("License", len(license_pages)),
    ]

    total_pages = 1 + toc_page_count + sum(count for _, count in chapter_counts)

    toc_entries: list[tuple[str, int]] = []
    next_page = 1 + toc_page_count + 1  # Cover + TOC + first chapter page
    for chapter_title, count in chapter_counts:
        if count > 0:
            toc_entries.append((chapter_title, next_page))
            next_page += count

    page_index = 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        draw_cover(pdf, title, subtitle)
        page_index += 1
        draw_toc_page(pdf, toc_entries, page_index, total_pages)
        page_index += 1

        for i, page_lines in enumerate(intro_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="Introduction",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for i, page_lines in enumerate(scope_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="Report Scope",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for i, page_lines in enumerate(recommendation_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="AI Recommendation",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        if input_data_summary:
            draw_input_data_page(
                pdf=pdf,
                summary=input_data_summary,
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for (section_title, images, per_page), section_page_count in zip(section_specs, section_page_counts):
            if section_page_count == 0:
                continue
            for page_images in chunked(images, per_page):
                draw_image_page(
                    pdf=pdf,
                    section_title=section_title,
                    images=page_images,
                    page_index=page_index,
                    total_pages=total_pages,
                )
                page_index += 1

        for i, page_lines in enumerate(methodology_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="Simulation Methodology",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for i, page_lines in enumerate(data_requirements_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="Data Requirements",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for config_chunk in chunked(battery_configuration_rows, 2):
            draw_config_cards_page(
                pdf=pdf,
                section_title="Battery Configuration Used",
                rows=config_chunk,
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for config_chunk in chunked(energy_tariff_configuration_rows, 2):
            draw_config_cards_page(
                pdf=pdf,
                section_title="Energy Tariff Configuration Used",
                rows=config_chunk,
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for i, page_lines in enumerate(license_pages):
            draw_structured_text_page(
                pdf=pdf,
                title="License",
                page_lines=page_lines,
                is_continuation=(i > 0),
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PDF report from global, monthly and seasonal graph images."
    )
    parser.add_argument(
        "--global",
        dest="global_images",
        nargs="*",
        default=[],
        help="Optional global graph image files (PNG/JPG/SVG). Shell globs are supported.",
    )
    parser.add_argument(
        "--monthly",
        nargs="+",
        required=True,
        help="Monthly graph image files (PNG/JPG/SVG). Shell globs are supported.",
    )
    parser.add_argument(
        "--seasonal",
        nargs="+",
        required=True,
        help="Seasonal graph image files (PNG/JPG/SVG). Shell globs are supported.",
    )
    parser.add_argument(
        "--output",
        default="out/battery_graph_report.pdf",
        help="Output PDF file path (default: out/battery_graph_report.pdf).",
    )
    parser.add_argument(
        "--title",
        default="Battery Simulation Graph Report",
        help="Report title on the cover page.",
    )
    parser.add_argument(
        "--subtitle",
        default="Monthly and seasonal comparison charts",
        help="Optional subtitle on the cover page.",
    )
    parser.add_argument(
        "--intro",
        default=DEFAULT_INTRO,
        help="Introduction text shown at the beginning of the PDF report.",
    )
    parser.add_argument(
        "--methodology",
        default=DEFAULT_METHODOLOGY,
        help="Simulation methodology text shown after the introduction page.",
    )
    parser.add_argument(
        "--scope",
        default=DEFAULT_SCOPE,
        help="Report scope text shown after the methodology section.",
    )
    parser.add_argument(
        "--data-requirements",
        default=DEFAULT_DATA_REQUIREMENTS,
        help="Data requirements text shown after the report scope section.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help=(
            "Configuration JSON files used in the analysis. "
            "Supports both raw config files and simulation output JSON files."
        ),
    )
    parser.add_argument(
        "--simulation-jsons",
        nargs="*",
        default=[],
        help="Simulation output JSON files used to summarize input data ingested.",
    )
    parser.add_argument(
        "--recommendation-file",
        default=None,
        help=(
            "Optional text/markdown file containing AI recommendation to embed in the PDF "
            "(for example out/simulation_llm_recommendation/recommendation_ollama.md)."
        ),
    )
    parser.add_argument(
        "--license-file",
        default="LICENSE",
        help=(
            "Optional text file containing the project license to embed in the PDF "
            "(default: LICENSE)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global_images = validate_images(args.global_images, "global") if args.global_images else []
    monthly_images = validate_images(args.monthly, "monthly")
    seasonal_images = validate_images(args.seasonal, "seasonal")
    config_paths = validate_json_paths(args.configs, "config") if args.configs else []
    simulation_paths = (
        validate_json_paths(args.simulation_jsons, "simulation")
        if args.simulation_jsons
        else []
    )
    battery_configuration_rows = load_battery_configuration_rows(config_paths) if config_paths else []
    energy_tariff_configuration_rows = load_energy_tariff_configuration_rows(config_paths) if config_paths else []
    input_data_summary = load_input_data_summary(simulation_paths, len(config_paths))
    recommendation_text = ""
    if args.recommendation_file:
        rec_path = Path(args.recommendation_file)
        if rec_path.exists() and rec_path.is_file():
            recommendation_text = rec_path.read_text(encoding="utf-8").strip()
            if recommendation_text:
                print(f"Loaded AI recommendation: {rec_path}")
            else:
                print(f"AI recommendation file is empty, skipping: {rec_path}")
        else:
            print(f"AI recommendation file not found, skipping: {rec_path}")
    license_text = ""
    if args.license_file:
        license_path = Path(args.license_file)
        if license_path.exists() and license_path.is_file():
            license_text = license_path.read_text(encoding="utf-8").strip()
            if license_text:
                print(f"Loaded license text: {license_path}")
            else:
                print(f"License file is empty, skipping: {license_path}")
        else:
            print(f"License file not found, skipping: {license_path}")
    output_pdf = Path(args.output)

    build_pdf(
        output_pdf=output_pdf,
        title=args.title,
        subtitle=args.subtitle,
        intro_text=args.intro,
        methodology_text=args.methodology,
        scope_text=args.scope,
        data_requirements_text=args.data_requirements,
        recommendation_text=recommendation_text,
        license_text=license_text,
        input_data_summary=input_data_summary,
        battery_configuration_rows=battery_configuration_rows,
        energy_tariff_configuration_rows=energy_tariff_configuration_rows,
        global_images=global_images,
        monthly_images=monthly_images,
        seasonal_images=seasonal_images,
    )
    print(f"PDF report generated: {output_pdf}")


if __name__ == "__main__":
    main()
