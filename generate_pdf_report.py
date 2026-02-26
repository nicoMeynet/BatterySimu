#!/usr/bin/env python3
"""
Build a PDF report from exported notebook graphs.

Example:
  python generate_pdf_report.py \
    --global out/graphs/global/*.png \
    --monthly out/graphs/monthly/*.png \
    --seasonal out/graphs/seasonal/*.png \
    --output out/battery_graph_report.pdf \
    --title "Residential Battery Sizing & Performance Analysis"
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DOC_DEFAULTS_DIR = Path(__file__).resolve().parent / "doc" / "pdf_report"
TOC_ENTRIES_PER_PAGE = 14


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


def build_kpi_summary_overview_text(kpi_summary_markdown_text: str) -> str:
    """Build a compact, reader-friendly summary from the full KPI markdown.

    This keeps the main report concise while the full markdown is included later
    in the KPI appendix for traceability.
    """
    if not kpi_summary_markdown_text.strip():
        return ""

    lines = [ln.rstrip() for ln in kpi_summary_markdown_text.splitlines()]

    metadata_lines: list[str] = []
    graph_sections: list[dict[str, str]] = []
    llm_notes: list[str] = []

    current: dict[str, str] | None = None
    in_llm_notes = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("## Notes for LLM Recommendation"):
            in_llm_notes = True
            current = None
            continue

        if in_llm_notes:
            if line.startswith("- "):
                llm_notes.append(line[2:].strip())
            continue

        if line.startswith("## Graph KPI: "):
            current = {
                "title": line[len("## Graph KPI: ") :].strip(),
                "threshold": "",
                "selection_rule": "",
                "recommendation": "",
                "rationale": "",
            }
            graph_sections.append(current)
            continue

        if current is None:
            if line.startswith("- "):
                metadata_lines.append(line)
            continue

        if line.startswith("- Threshold:"):
            current["threshold"] = line[len("- Threshold:") :].strip()
        elif line.startswith("- Selection rule:"):
            current["selection_rule"] = line[len("- Selection rule:") :].strip()
        elif line.startswith("- Recommendation:"):
            current["recommendation"] = line[len("- Recommendation:") :].strip()
        elif line.startswith("- Rationale:"):
            current["rationale"] = line[len("- Rationale:") :].strip()

    parts: list[str] = []
    parts.append("## KPI Summary")
    parts.append("")
    parts.append(
        "This section condenses the 7 graph-based sizing KPIs used to guide the battery recommendation. "
        "A full traceable export of the KPI markdown (including detailed tables) is included later in the KPI Appendix."
    )
    parts.append("")

    if metadata_lines:
        parts.append("### Build Metadata")
        parts.extend(metadata_lines[:8])  # keep top metadata compact
        parts.append("")

    if graph_sections:
        parts.append("### KPI Recommendations At A Glance")
        for item in graph_sections:
            title = item.get("title", "Graph KPI")
            rec = item.get("recommendation") or "not found"
            threshold = item.get("threshold") or "not specified"
            rule = item.get("selection_rule") or ""
            rationale = item.get("rationale") or ""

            parts.append(f"## {title}")
            parts.append(f"- Recommendation: {rec}")
            parts.append(f"- Threshold: {threshold}")
            if rule:
                parts.append(f"- Rule: {rule}")
            if rationale:
                # Keep rationale short in the main body
                wrapped = textwrap.shorten(rationale, width=240, placeholder="...")
                parts.append(f"- Rationale: {wrapped}")
            parts.append("")

    if llm_notes:
        parts.append("### Interpretation Notes")
        for note in llm_notes:
            parts.append(f"- {note}")
        parts.append("")

    parts.append("### Full Details")
    parts.append(
        "The complete KPI markdown (all thresholds, candidate tables and rationales) is included in the KPI Appendix section of this PDF."
    )

    return "\n".join(parts).strip()


def _infer_kpi_type_label(title: str, selection_rule: str) -> str:
    t = (title or "").lower()
    r = (selection_rule or "").lower()
    if "step delta" in t or "incremental" in r:
        return "Step delta"
    if "cap" in t or "threshold" in r or "excluded" in r:
        return "Constraint cap"
    return "Graph KPI"


def parse_kpi_summary_compact_data(kpi_summary_markdown_text: str) -> dict:
    """Extract compact KPI summary rows from kpi_summary.md for one-page PDF rendering."""
    result = {
        "metadata": {},
        "rows": [],
        "row_count": 0,
    }
    if not kpi_summary_markdown_text.strip():
        return result

    lines = [ln.rstrip() for ln in kpi_summary_markdown_text.splitlines()]
    current: dict[str, str] | None = None
    graph_sections: list[dict[str, str]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("## Notes for LLM Recommendation"):
            current = None
            break

        if line.startswith("## Graph KPI: "):
            current = {
                "title": _clean_inline_markdown_for_pdf(line[len("## Graph KPI: ") :].strip()),
                "threshold": "",
                "selection_rule": "",
                "recommendation": "",
            }
            graph_sections.append(current)
            continue

        if current is None:
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                result["metadata"][key.strip().lower()] = _clean_inline_markdown_for_pdf(value.strip())
            continue

        if line.startswith("- Threshold:"):
            current["threshold"] = _clean_inline_markdown_for_pdf(line[len("- Threshold:") :].strip())
        elif line.startswith("- Selection rule:"):
            current["selection_rule"] = _clean_inline_markdown_for_pdf(line[len("- Selection rule:") :].strip())
        elif line.startswith("- Recommendation:"):
            current["recommendation"] = _clean_inline_markdown_for_pdf(line[len("- Recommendation:") :].strip())

    compact_rows: list[dict[str, str]] = []
    for item in graph_sections:
        title = item.get("title", "").strip()
        threshold = item.get("threshold", "").strip()
        selection_rule = item.get("selection_rule", "").strip()
        recommendation = item.get("recommendation", "").strip()

        # Extract battery size and scenario from recommendation line if possible.
        size_display = "-"
        scenario_display = "-"
        rec_display = recommendation or "-"

        size_match = re.search(r"(\d+(?:\.\d+)?)\s*kWh", recommendation, flags=re.IGNORECASE)
        if size_match:
            size_display = f"{float(size_match.group(1)):.2f} kWh"

        scenario_match = re.search(r"\(([^)]+)\)", recommendation)
        if scenario_match:
            scenario_display = scenario_match.group(1).strip()

        # If parsing worked, keep recommendation compact; otherwise preserve a shortened line.
        if size_display != "-" or scenario_display != "-":
            rec_display = size_display
        else:
            rec_display = textwrap.shorten(recommendation or "-", width=52, placeholder="...")

        compact_rows.append(
            {
                "kpi": title,
                "type": _infer_kpi_type_label(title, selection_rule),
                "recommended": rec_display,
                "scenario": scenario_display,
                "threshold": textwrap.shorten(threshold or "-", width=84, placeholder="..."),
            }
        )

    result["rows"] = compact_rows
    result["row_count"] = len(compact_rows)
    return result


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
    sortable_rows: list[tuple[tuple[int, float, str], dict]] = []
    for path in sorted(config_paths):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = extract_config_payload(raw)

        battery = cfg["battery"]
        capacity_per_phase = battery.get("capacity_Wh_per_phase", [0, 0, 0])
        total_kwh = sum(capacity_per_phase) / 1000 if capacity_per_phase else 0
        scenario = path.stem.replace("config_", "")
        scenario_lc = scenario.lower()
        is_no_battery = "nobattery" in scenario_lc or total_kwh == 0

        row = {
            "scenario": scenario,
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
        sort_key = (0 if is_no_battery else 1, float(total_kwh), scenario_lc)
        sortable_rows.append((sort_key, row))
    return [row for _sort_key, row in sorted(sortable_rows, key=lambda item: item[0])]


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

    cover_title = title
    if title.strip() == "Residential Battery Sizing & Performance Analysis":
        cover_title = "Residential Battery Sizing\n&\nPerformance Analysis"

    title_line_count = cover_title.count("\n") + 1
    multiline_title = title_line_count > 1
    title_y = 0.79 if multiline_title else 0.75
    subtitle_y = 0.64 if multiline_title else 0.69
    generated_y = 0.58 if multiline_title else 0.63
    divider_y = 0.56 if multiline_title else 0.61

    ax.text(
        0.5,
        title_y,
        cover_title,
        ha="center",
        va="center",
        fontsize=26 if title_line_count >= 3 else 28,
        fontweight="bold",
        color="#1f2937",
        linespacing=1.15,
    )
    if subtitle:
        ax.text(
            0.5,
            subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=13,
            color="#4b5563",
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(
        0.5,
        generated_y,
        f"Generated on {generated}",
        ha="center",
        va="center",
        fontsize=11,
        color="#6b7280",
    )

    ax.add_patch(
        plt.Rectangle(
            (0.15, divider_y),
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
    is_continuation: bool = False,
) -> None:
    """Render a table of contents page with chapter start pages."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.06,
        0.965,
        "Table of Contents (cont.)" if is_continuation else "Table of Contents",
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
        if line.startswith("# "):
            lines.append(("h2", line[2:].strip()))
            continue
        if line.startswith("## "):
            lines.append(("h2", line[3:].strip()))
            continue
        if line.startswith("### "):
            lines.append(("h3", line[4:].strip()))
            continue
        if line.startswith("- "):
            wrapped = textwrap.wrap(line[2:].strip(), width=88)
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


def _split_markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _clean_inline_markdown_for_pdf(text: str) -> str:
    """Normalize inline markdown syntax for PDF body rendering."""
    if not text:
        return ""
    cleaned = str(text).replace("`", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_table_cells(line)
    if len(cells) < 2:
        return False
    for cell in cells:
        token = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", token):
            return False
    return True


def _parse_markdown_table_alignments(separator_line: str, column_count: int) -> list[str]:
    aligns: list[str] = []
    cells = _split_markdown_table_cells(separator_line)
    cells = cells[:column_count] + ["---"] * max(0, column_count - len(cells))
    for cell in cells:
        token = cell.replace(" ", "")
        if token.startswith(":") and token.endswith(":"):
            aligns.append("center")
        elif token.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def parse_markdown_blocks_with_tables(body_text: str) -> list[dict]:
    """Parse markdown-ish text into renderable blocks, including table blocks."""
    blocks: list[dict] = []
    raw_lines = body_text.strip().splitlines()
    i = 0
    while i < len(raw_lines):
        raw_line = raw_lines[i]
        line = raw_line.strip()

        if not line:
            blocks.append({"kind": "blank"})
            i += 1
            continue

        # Markdown table block: header row + separator row + data rows.
        if (
            line.startswith("|")
            and i + 1 < len(raw_lines)
            and _is_markdown_table_separator(raw_lines[i + 1])
        ):
            header_cells = [_clean_inline_markdown_for_pdf(c) for c in _split_markdown_table_cells(raw_lines[i])]
            alignments = _parse_markdown_table_alignments(raw_lines[i + 1], len(header_cells))
            rows: list[list[str]] = []
            i += 2
            while i < len(raw_lines):
                candidate_raw = raw_lines[i]
                candidate = candidate_raw.strip()
                if not candidate or not candidate.startswith("|") or not candidate.endswith("|"):
                    break
                row_cells = [_clean_inline_markdown_for_pdf(c) for c in _split_markdown_table_cells(candidate_raw)]
                if len(row_cells) < len(header_cells):
                    row_cells = row_cells + [""] * (len(header_cells) - len(row_cells))
                elif len(row_cells) > len(header_cells):
                    # Merge overflow cells into the last column to stay robust.
                    row_cells = row_cells[: len(header_cells) - 1] + [" | ".join(row_cells[len(header_cells) - 1 :])]
                rows.append(row_cells)
                i += 1
            blocks.append(
                {
                    "kind": "table",
                    "headers": header_cells,
                    "alignments": alignments,
                    "rows": rows,
                }
            )
            continue

        if line.startswith("# "):
            blocks.append({"kind": "h2", "text": _clean_inline_markdown_for_pdf(line[2:].strip())})
            i += 1
            continue
        if line.startswith("## "):
            blocks.append({"kind": "h2", "text": _clean_inline_markdown_for_pdf(line[3:].strip())})
            i += 1
            continue
        if line.startswith("### "):
            blocks.append({"kind": "h3", "text": _clean_inline_markdown_for_pdf(line[4:].strip())})
            i += 1
            continue
        if line.startswith("- "):
            bullet_text = _clean_inline_markdown_for_pdf(line[2:].strip())
            wrapped = textwrap.wrap(
                bullet_text,
                width=82,
                break_long_words=True,
                break_on_hyphens=True,
            )
            if wrapped:
                blocks.append({"kind": "bullet", "text": wrapped[0]})
                for cont in wrapped[1:]:
                    blocks.append({"kind": "bullet_cont", "text": cont})
            i += 1
            continue

        plain_text = _clean_inline_markdown_for_pdf(line)
        for wline in textwrap.wrap(
            plain_text,
            width=84,
            break_long_words=True,
            break_on_hyphens=True,
        ):
            blocks.append({"kind": "text", "text": wline})
        i += 1

    return blocks


def _table_column_width_fractions(headers: list[str], rows: list[list[str]]) -> list[float]:
    if not headers:
        return []

    n = len(headers)
    header_lc = [h.lower() for h in headers]
    weights: list[float] = []

    sample_rows = rows[:8]
    for idx, header in enumerate(headers):
        values = [header]
        for row in sample_rows:
            if idx < len(row):
                values.append(row[idx])
        lengths = [len(v.strip()) for v in values if isinstance(v, str)]
        max_len = max(lengths or [8])
        avg_len = (sum(lengths) / len(lengths)) if lengths else 8

        w = max(8.0, min(max_len, 34) * 0.70 + min(avg_len, 24) * 0.30)
        if "violating months" in header_lc[idx]:
            w *= 2.4
        elif "scenario" in header_lc[idx]:
            w *= 1.35
        elif any(tok in header_lc[idx] for tok in ("status", "months >", "worst month", "empty", "full")):
            w *= 1.10
        elif any(tok in header_lc[idx] for tok in ("(%)", "kwh", "years", "chf", "size")):
            w *= 0.95
        weights.append(w)

    total = sum(weights) or float(n)
    fracs = [w / total for w in weights]

    # Soft min/max to avoid unusable thin columns and over-dominant long-text columns.
    min_frac = 0.06 if n <= 6 else 0.05
    max_frac = 0.42 if n <= 5 else 0.36 if n <= 6 else 0.32

    for _ in range(6):
        below = [i for i, f in enumerate(fracs) if f < min_frac]
        above = [i for i, f in enumerate(fracs) if f > max_frac]
        if not below and not above:
            break

        for i in below:
            deficit = min_frac - fracs[i]
            fracs[i] = min_frac
            donors = [j for j in range(n) if j != i and fracs[j] > min_frac]
            donor_total = sum(fracs[j] - min_frac for j in donors)
            if donor_total > 0:
                for j in donors:
                    take = deficit * ((fracs[j] - min_frac) / donor_total)
                    fracs[j] = max(min_frac, fracs[j] - take)

        for i in above:
            excess = fracs[i] - max_frac
            fracs[i] = max_frac
            receivers = [j for j in range(n) if j != i and fracs[j] < max_frac]
            recv_total = sum(max_frac - fracs[j] for j in receivers)
            if recv_total > 0:
                for j in receivers:
                    give = excess * ((max_frac - fracs[j]) / recv_total)
                    fracs[j] = min(max_frac, fracs[j] + give)

    # Renormalize for numerical drift.
    total = sum(fracs) or float(n)
    return [f / total for f in fracs]


def _wrap_table_cell_lines(text: str, char_cap: int) -> list[str]:
    value = "" if text is None else str(text)
    if not value:
        return [""]
    wrapped: list[str] = []
    for part in value.splitlines() or [""]:
        segs = textwrap.wrap(
            part,
            width=max(4, char_cap),
            break_long_words=True,
            break_on_hyphens=True,
        )
        wrapped.extend(segs or [""])
    return wrapped or [""]


def _measure_markdown_table_block(block: dict, total_width: float = 0.84) -> dict:
    headers = [str(x) for x in block.get("headers", [])]
    rows = [[str(c) for c in row] for row in (block.get("rows") or [])]
    alignments = [str(a) for a in (block.get("alignments") or [])]
    if not headers:
        return {
            "total_height": 0.0,
            "column_fracs": [],
            "header_lines": [],
            "row_lines": [],
            "row_heights": [],
            "header_height": 0.0,
            "alignments": alignments,
        }

    column_fracs = _table_column_width_fractions(headers, rows)
    # Approx char capacity for wrapped text, tuned for A4 page width and font sizes used below.
    # Use conservative wrap widths so text stays inside cells on the PDF page.
    char_caps = [max(5, int(frac * 96)) for frac in column_fracs]

    header_lines = [_wrap_table_cell_lines(headers[i], char_caps[i]) for i in range(len(headers))]
    row_lines = [
        [_wrap_table_cell_lines((row[i] if i < len(row) else ""), char_caps[i]) for i in range(len(headers))]
        for row in rows
    ]

    header_line_h = 0.0138
    row_line_h = 0.0129
    cell_pad_y = 0.0042

    header_height = max((len(lines) for lines in header_lines), default=1) * header_line_h + 2 * cell_pad_y
    row_heights = [
        max((len(lines) for lines in row_cells), default=1) * row_line_h + 2 * cell_pad_y
        for row_cells in row_lines
    ]
    total_height = header_height + sum(row_heights)

    # Keep a small breathing room after tables when used in flow layout.
    return {
        "total_height": total_height,
        "column_fracs": column_fracs,
        "header_lines": header_lines,
        "row_lines": row_lines,
        "row_heights": row_heights,
        "header_height": header_height,
        "alignments": alignments[: len(headers)] + ["left"] * max(0, len(headers) - len(alignments)),
        "header_line_h": header_line_h,
        "row_line_h": row_line_h,
        "cell_pad_y": cell_pad_y,
        "total_width": total_width,
    }


def _appendix_heading_lines(text: str, kind: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return [""]
    wrap_width = 62 if kind == "h2" else 70
    return textwrap.wrap(
        raw,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [raw]


def markdown_block_height_units(block: dict) -> float:
    kind = str(block.get("kind", "text"))
    if kind == "table":
        layout = _measure_markdown_table_block(block)
        return layout["total_height"] + 0.010

    # KPI Appendix uses a dedicated renderer with tighter typography than the generic
    # structured-text pages, so use matching heights for pagination.
    appendix_line_h = 0.019
    if kind == "blank":
        return 0.012
    if kind == "h2":
        n = len(_appendix_heading_lines(str(block.get("text", "")), "h2"))
        return appendix_line_h * (1.25 + max(0, n - 1) * 1.05)
    if kind == "h3":
        n = len(_appendix_heading_lines(str(block.get("text", "")), "h3"))
        return appendix_line_h * (1.15 + max(0, n - 1) * 1.0)
    return appendix_line_h


def _split_markdown_table_block_rows(block: dict, max_height: float) -> list[dict]:
    if block.get("kind") != "table":
        return [block]
    rows = list(block.get("rows") or [])
    if not rows:
        return [block]

    headers = list(block.get("headers") or [])
    alignments = list(block.get("alignments") or [])
    chunks: list[dict] = []
    current_rows: list[list[str]] = []

    def _make(rows_subset: list[list[str]]) -> dict:
        return {"kind": "table", "headers": headers, "alignments": alignments, "rows": rows_subset}

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_block = _make(candidate_rows)
        h = markdown_block_height_units(candidate_block)
        if current_rows and h > max_height:
            chunks.append(_make(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append(_make(current_rows))

    return chunks or [block]


def paginate_markdown_blocks(
    blocks: list[dict],
    *,
    start_y: float = 0.88,
    min_y: float = 0.08,
) -> list[list[dict]]:
    pages: list[list[dict]] = []
    current: list[dict] = []
    y = start_y
    page_capacity = start_y - min_y

    for block in blocks:
        block_items = [block]
        if block.get("kind") == "table":
            h = markdown_block_height_units(block)
            if h > page_capacity:
                block_items = _split_markdown_table_block_rows(block, page_capacity)

        for item in block_items:
            h = markdown_block_height_units(item)
            if y - h < min_y and current:
                pages.append(current)
                current = []
                y = start_y
            current.append(item)
            y -= h

    if current:
        pages.append(current)
    return pages


def draw_markdown_table_block(
    ax,
    table_block: dict,
    *,
    x: float,
    y_top: float,
    width: float = 0.84,
) -> float:
    """Draw a markdown table as a real bordered table. Returns vertical height used."""
    layout = _measure_markdown_table_block(table_block, total_width=width)
    total_height = float(layout["total_height"])
    if total_height <= 0:
        return 0.0

    headers: list[str] = list(table_block.get("headers") or [])
    rows: list[list[str]] = list(table_block.get("rows") or [])
    column_fracs: list[float] = list(layout["column_fracs"])
    header_lines: list[list[str]] = list(layout["header_lines"])
    row_lines: list[list[list[str]]] = list(layout["row_lines"])
    row_heights: list[float] = list(layout["row_heights"])
    alignments: list[str] = list(layout["alignments"])
    header_height = float(layout["header_height"])
    header_line_h = float(layout["header_line_h"])
    row_line_h = float(layout["row_line_h"])
    cell_pad_y = float(layout["cell_pad_y"])

    edge_color = "#cbd5e1"
    header_bg = "#e5e7eb"
    alt_bg = "#f8fafc"
    text_color = "#111827"
    sub_text_color = "#374151"

    # Outer border
    ax.add_patch(
        plt.Rectangle(
            (x, y_top - total_height),
            width,
            total_height,
            linewidth=1.0,
            edgecolor=edge_color,
            facecolor="white",
            transform=ax.transAxes,
            zorder=0,
        )
    )

    # Column edges (absolute x coordinates)
    col_xs = [x]
    running = x
    for frac in column_fracs:
        running += width * frac
        col_xs.append(running)

    # Header row background and border
    header_bottom = y_top - header_height
    ax.add_patch(
        plt.Rectangle(
            (x, header_bottom),
            width,
            header_height,
            linewidth=0.8,
            edgecolor=edge_color,
            facecolor=header_bg,
            transform=ax.transAxes,
            zorder=0,
        )
    )

    # Draw vertical lines across entire table
    for cx in col_xs[1:-1]:
        ax.plot([cx, cx], [y_top - total_height, y_top], transform=ax.transAxes, color=edge_color, linewidth=0.8, zorder=1)

    # Header text
    for col_idx, header in enumerate(headers):
        cell_left = col_xs[col_idx]
        cell_right = col_xs[col_idx + 1]
        pad_x = 0.006
        halign = "center" if alignments[col_idx] == "center" else ("right" if alignments[col_idx] == "right" else "left")
        text_x = (cell_left + cell_right) / 2 if halign == "center" else (cell_right - pad_x if halign == "right" else cell_left + pad_x)
        ax.text(
            text_x,
            y_top - cell_pad_y,
            "\n".join(header_lines[col_idx]),
            ha=halign,
            va="top",
            fontsize=8.7,
            fontweight="bold",
            color=text_color,
            linespacing=1.12,
            transform=ax.transAxes,
            clip_on=True,
        )

    # Body rows
    y_cursor = header_bottom
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_bottom = y_cursor - row_h
        if row_idx % 2 == 1:
            ax.add_patch(
                plt.Rectangle(
                    (x, row_bottom),
                    width,
                    row_h,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=alt_bg,
                    transform=ax.transAxes,
                    zorder=0,
                )
            )

        # horizontal line
        ax.plot([x, x + width], [row_bottom, row_bottom], transform=ax.transAxes, color=edge_color, linewidth=0.8, zorder=1)

        for col_idx in range(len(headers)):
            cell_left = col_xs[col_idx]
            cell_right = col_xs[col_idx + 1]
            pad_x = 0.006
            align = alignments[col_idx]
            halign = "center" if align == "center" else ("right" if align == "right" else "left")
            text_x = (cell_left + cell_right) / 2 if halign == "center" else (cell_right - pad_x if halign == "right" else cell_left + pad_x)
            lines = row_lines[row_idx][col_idx]
            ax.text(
                text_x,
                y_cursor - cell_pad_y,
                "\n".join(lines),
                ha=halign,
                va="top",
                fontsize=8.4,
                color=sub_text_color,
                linespacing=1.12,
                transform=ax.transAxes,
                clip_on=True,
            )

        y_cursor = row_bottom

    # Top and header divider lines
    ax.plot([x, x + width], [y_top, y_top], transform=ax.transAxes, color=edge_color, linewidth=0.8, zorder=1)
    ax.plot([x, x + width], [header_bottom, header_bottom], transform=ax.transAxes, color=edge_color, linewidth=0.8, zorder=1)

    return total_height + 0.010


def draw_markdown_blocks_page(
    pdf: PdfPages,
    title: str,
    page_blocks: list[dict],
    is_continuation: bool,
    page_index: int,
    total_pages: int,
) -> None:
    """Structured text page renderer with markdown-table support (used for KPI appendix)."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.06,
        0.965,
        f"{title} (cont.)" if is_continuation else title,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
        transform=ax.transAxes,
        clip_on=True,
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
        clip_on=True,
    )

    x = 0.08
    y = 0.885
    line_h = 0.019
    h2_fontsize = 11.8
    h3_fontsize = 10.7
    body_fontsize = 9.8
    bullet_indent = 0.024

    for block in page_blocks:
        kind = str(block.get("kind", "text"))
        if kind == "table":
            used = draw_markdown_table_block(ax, block, x=x, y_top=y, width=0.84)
            y -= used
            continue
        if kind == "blank":
            y -= 0.012
            continue
        text = str(block.get("text", ""))
        if kind == "h2":
            h2_lines = _appendix_heading_lines(text, "h2")
            ax.text(
                x,
                y,
                "\n".join(h2_lines),
                ha="left",
                va="top",
                fontsize=h2_fontsize,
                fontweight="bold",
                color="#111827",
                transform=ax.transAxes,
                clip_on=True,
                linespacing=1.15,
            )
            y -= line_h * (1.25 + max(0, len(h2_lines) - 1) * 1.05)
            continue
        if kind == "h3":
            h3_lines = _appendix_heading_lines(text, "h3")
            ax.text(
                x,
                y,
                "\n".join(h3_lines),
                ha="left",
                va="top",
                fontsize=h3_fontsize,
                fontweight="bold",
                color="#1f2937",
                transform=ax.transAxes,
                clip_on=True,
                linespacing=1.15,
            )
            y -= line_h * (1.15 + max(0, len(h3_lines) - 1) * 1.0)
            continue
        if kind == "bullet":
            ax.text(
                x,
                y,
                f"• {text}",
                ha="left",
                va="top",
                fontsize=body_fontsize,
                color="#374151",
                transform=ax.transAxes,
                clip_on=True,
                linespacing=1.18,
            )
            y -= line_h
            continue
        if kind == "bullet_cont":
            ax.text(
                x + bullet_indent,
                y,
                text,
                ha="left",
                va="top",
                fontsize=body_fontsize,
                color="#374151",
                transform=ax.transAxes,
                clip_on=True,
                linespacing=1.18,
            )
            y -= line_h
            continue
        ax.text(
            x,
            y,
            text,
            ha="left",
            va="top",
            fontsize=body_fontsize,
            color="#374151",
            transform=ax.transAxes,
            clip_on=True,
            linespacing=1.18,
        )
        y -= line_h

    ax.text(
        0.06,
        0.025,
        COPYRIGHT_SHORT,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#9ca3af",
        transform=ax.transAxes,
        clip_on=True,
    )
    ax.text(
        0.94,
        0.025,
        "License: CC BY-NC 4.0",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#9ca3af",
        transform=ax.transAxes,
        clip_on=True,
    )

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


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


def draw_kpi_summary_compact_page(
    pdf: PdfPages,
    kpi_summary_text: str,
    page_index: int,
    total_pages: int,
) -> None:
    """Render a single compact KPI Summary page (one row per graph KPI)."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.06,
        0.965,
        "KPI Summary",
        ha="left",
        va="top",
        fontsize=18,
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

    parsed = parse_kpi_summary_compact_data(kpi_summary_text)
    rows = list(parsed.get("rows") or [])
    metadata = dict(parsed.get("metadata") or {})

    ax.text(
        0.06,
        0.915,
        (
            "Single-page overview of the 7 graph-based sizing KPIs used for battery selection. "
            "See KPI Appendix for full candidate tables and detailed rationales."
        ),
        ha="left",
        va="top",
        fontsize=10.2,
        color="#4b5563",
        transform=ax.transAxes,
        wrap=True,
    )

    # Metadata strip (compact)
    meta_items = []
    for label_key, display_label in [
        ("generated at", "Generated"),
        ("simulation files analyzed", "Simulation files"),
        ("graph kpi sections included", "Graph KPIs"),
        ("kpi config source", "KPI config"),
    ]:
        value = metadata.get(label_key)
        if value:
            meta_items.append((display_label, value))

    if meta_items:
        ax.add_patch(
            plt.Rectangle(
                (0.06, 0.815),
                0.88,
                0.075,
                linewidth=1.0,
                edgecolor="#d1d5db",
                facecolor="#f9fafb",
                transform=ax.transAxes,
            )
        )
        mx = 0.075
        my = 0.867
        for idx, (label, value) in enumerate(meta_items[:4]):
            col = idx % 2
            row = idx // 2
            x = mx + col * 0.43
            y = my - row * 0.031
            ax.text(
                x,
                y,
                f"{label}:",
                ha="left",
                va="center",
                fontsize=9.1,
                fontweight="bold",
                color="#374151",
                transform=ax.transAxes,
            )
            ax.text(
                x + 0.11,
                y,
                textwrap.shorten(str(value), width=44, placeholder="..."),
                ha="left",
                va="center",
                fontsize=9.1,
                color="#111827",
                transform=ax.transAxes,
            )

    if rows:
        table_rows = [
            [
                r.get("kpi", "-"),
                r.get("type", "-"),
                r.get("recommended", "-"),
                r.get("scenario", "-"),
                r.get("threshold", "-"),
            ]
            for r in rows
        ]
        # Shorten a few cells to keep this one-page summary readable.
        for row in table_rows:
            row[0] = textwrap.shorten(row[0], width=58, placeholder="...")
            row[3] = textwrap.shorten(row[3], width=34, placeholder="...")
            row[4] = textwrap.shorten(row[4], width=88, placeholder="...")

        table_block = {
            "kind": "table",
            "headers": ["Graph KPI", "Type", "Recommended", "Scenario", "Threshold"],
            "alignments": ["left", "center", "center", "left", "left"],
            "rows": table_rows,
        }

        table_top = 0.79 if meta_items else 0.86
        used = draw_markdown_table_block(ax, table_block, x=0.06, y_top=table_top, width=0.88)
        footer_note_y = max(0.055, table_top - used - 0.012)
        ax.text(
            0.06,
            footer_note_y,
            "Interpretation: 'Step delta' KPIs identify diminishing-return sizing points; 'Constraint cap' KPIs exclude non-compliant sizes first.",
            ha="left",
            va="top",
            fontsize=9.2,
            color="#4b5563",
            transform=ax.transAxes,
            wrap=True,
        )
    else:
        ax.add_patch(
            plt.Rectangle(
                (0.06, 0.50),
                0.88,
                0.22,
                linewidth=1.0,
                edgecolor="#d1d5db",
                facecolor="#f9fafb",
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.50,
            0.61,
            "No graph KPI rows could be parsed from kpi_summary.md",
            ha="center",
            va="center",
            fontsize=12,
            color="#374151",
            transform=ax.transAxes,
        )
        ax.text(
            0.50,
            0.565,
            "Check the KPI summary markdown format or see the KPI Appendix for raw content.",
            ha="center",
            va="center",
            fontsize=10,
            color="#6b7280",
            transform=ax.transAxes,
        )

    ax.text(0.06, 0.025, COPYRIGHT_SHORT, ha="left", va="bottom", fontsize=8.5, color="#9ca3af", transform=ax.transAxes)
    ax.text(0.94, 0.025, "License: CC BY-NC 4.0", ha="right", va="bottom", fontsize=8.5, color="#9ca3af", transform=ax.transAxes)

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
    recommendation_graph_images: list[Path],
    kpi_summary_text: str,
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
    recommendation_graph_page_count = len(list(chunked(recommendation_graph_images, 1))) if recommendation_graph_images else 0
    kpi_summary_page_count = 1 if kpi_summary_text.strip() else 0
    kpi_appendix_blocks = parse_markdown_blocks_with_tables(kpi_summary_text) if kpi_summary_text.strip() else []
    kpi_appendix_pages = paginate_markdown_blocks(kpi_appendix_blocks) if kpi_appendix_blocks else []
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
    chapter_counts = [
        ("Introduction", len(intro_pages)),
        ("Report Scope", len(scope_pages)),
        ("AI Recommendation", len(recommendation_pages) + recommendation_graph_page_count),
        ("KPI Summary", kpi_summary_page_count),
        ("Input Data Ingested", input_data_page_count),
        *[(section_title, count) for (section_title, _, _), count in zip(section_specs, section_page_counts)],
        ("Simulation Methodology", len(methodology_pages)),
        ("Data Requirements", len(data_requirements_pages)),
        ("Battery Configuration Used", battery_config_page_count),
        ("Energy Tariff Configuration Used", energy_tariff_page_count),
        ("KPI Appendix", len(kpi_appendix_pages)),
        ("License", len(license_pages)),
    ]

    nonzero_chapters = [(title, count) for title, count in chapter_counts if count > 0]
    toc_page_count = max(1, (len(nonzero_chapters) + TOC_ENTRIES_PER_PAGE - 1) // TOC_ENTRIES_PER_PAGE)
    total_pages = 1 + toc_page_count + sum(count for _, count in chapter_counts)

    toc_entries: list[tuple[str, int]] = []
    next_page = 1 + toc_page_count + 1  # Cover + TOC + first chapter page
    for chapter_title, count in nonzero_chapters:
        toc_entries.append((chapter_title, next_page))
        next_page += count

    toc_entry_pages: list[list[tuple[str, int]]] = (
        [toc_entries[i : i + TOC_ENTRIES_PER_PAGE] for i in range(0, len(toc_entries), TOC_ENTRIES_PER_PAGE)]
        if toc_entries
        else [[]]
    )

    page_index = 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        draw_cover(pdf, title, subtitle)
        page_index += 1
        for toc_idx, toc_page_entries in enumerate(toc_entry_pages):
            draw_toc_page(
                pdf,
                toc_page_entries,
                page_index,
                total_pages,
                is_continuation=(toc_idx > 0),
            )
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

        for page_images in chunked(recommendation_graph_images, 1):
            draw_image_page(
                pdf=pdf,
                section_title="AI Recommendation",
                images=page_images,
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        if kpi_summary_page_count > 0:
            draw_kpi_summary_compact_page(
                pdf=pdf,
                kpi_summary_text=kpi_summary_text,
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

        for i, page_lines in enumerate(kpi_appendix_pages):
            draw_markdown_blocks_page(
                pdf=pdf,
                title="KPI Appendix",
                page_blocks=page_lines,
                is_continuation=(i > 0),
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
        default="Residential Battery Sizing & Performance Analysis",
        help="Report title on the cover page.",
    )
    parser.add_argument(
        "--subtitle",
        default="Data-Driven Financial, Structural and Seasonal Evaluation",
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
        "--kpi-summary-file",
        default=None,
        help=(
            "Optional KPI summary markdown file to embed in the PDF as a compact KPI Summary "
            "section plus a full KPI Appendix (for example out/kpi_summary/kpi_summary.md)."
        ),
    )
    parser.add_argument(
        "--recommendation-graph-image",
        default="out/kpi_images/01_graph_kpi_consensus_best_battery.png",
        help=(
            "Optional KPI recommendation graph image to embed in the AI Recommendation "
            "section (default: out/kpi_images/01_graph_kpi_consensus_best_battery.png)."
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
    recommendation_graph_images: list[Path] = []
    if args.recommendation_graph_image:
        rec_graph_path = Path(args.recommendation_graph_image)
        if rec_graph_path.exists() and rec_graph_path.is_file():
            recommendation_graph_images = validate_images([str(rec_graph_path)], "AI recommendation graph")
            print(f"Loaded AI recommendation graph: {rec_graph_path}")
        else:
            print(f"AI recommendation graph not found, skipping: {rec_graph_path}")
    kpi_summary_text = ""
    if args.kpi_summary_file:
        kpi_summary_path = Path(args.kpi_summary_file)
        if kpi_summary_path.exists() and kpi_summary_path.is_file():
            kpi_summary_text = kpi_summary_path.read_text(encoding="utf-8").strip()
            if kpi_summary_text:
                print(f"Loaded KPI summary markdown: {kpi_summary_path}")
            else:
                print(f"KPI summary markdown file is empty, skipping: {kpi_summary_path}")
        else:
            print(f"KPI summary markdown file not found, skipping: {kpi_summary_path}")
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
        recommendation_graph_images=recommendation_graph_images,
        kpi_summary_text=kpi_summary_text,
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
