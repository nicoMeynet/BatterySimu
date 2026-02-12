#!/usr/bin/env python3
"""
Build a PDF report from exported notebook graphs.

Example:
  python generate_pdf_report.py \
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

DEFAULT_INTRO = """This report evaluates multiple residential battery configurations to determine the optimal storage size for the household. The analysis is based on real 3-phase grid measurements collected prior to battery installation, ensuring that all results reflect actual consumption and injection behavior.

Using timestamp-level import and export data, the simulator models tariff-aware charge and discharge decisions under defined battery constraints. Each configuration is evaluated against an identical no-battery baseline, enabling a fair and consistent comparison.

To generate meaningful results, the user must provide:
- Historical grid import and export measurements (without battery)
- 3-phase power data with adequate time resolution
- The battery configurations to be tested
- Applicable tariff parameters

Measurements can be obtained using a 3-phase energy meter (for example Shelly 3EM) integrated with Home Assistant for data collection and export.

The objective is to support a data-driven investment decision by balancing financial return, energy adequacy, and power limitations, selecting the smallest configuration that delivers robust and sustainable performance across the full year."""

DEFAULT_METHODOLOGY = """## 1. Input Data
- Real 3-phase household consumption data (Phase A, B, C)
- Timestamp-level power measurements
- Historical seasonal load patterns
- Exported data from Home Assistant
Data granularity ensures realistic modeling of daily and seasonal variability.

## 2. Simulation Engine
- Timestamp-by-timestamp battery charge/discharge decisions
- SOC (State of Charge) tracking with min/max constraints
- Power limitation enforcement (charge and discharge limits)
- Efficiency modeling (charge and discharge losses)
- Priority rules for self-consumption optimization
Battery behavior is evaluated under physical and electrical constraints.

## 3. Tariff Model
- Peak and off-peak consumption tariffs
- Injection tariffs for exported energy
- Net financial gain calculation versus no battery
- Cost comparison per scenario
All financial gains are calculated relative to the no-battery baseline.

## 4. Comparison Baseline
- Identical household load data
- Same solar production profile
- No-battery reference case
- Identical tariff assumptions
This ensures consistent and fair scenario comparison.

## 5. Key Performance Indicators (KPIs)
### Financial Indicators
- Monthly and seasonal net financial gain
- Grid import reduction
- Grid export reduction
### Energy Indicators
- Battery energy throughput
- Equivalent full cycles
- Energy shifting volume
### Structural Sizing Indicators
- Battery full SOC share (oversizing indicator)
- Battery empty SOC share (undersizing indicator)
- Daily structural undersizing
- Evening structural undersizing
### Power Indicators
- Active power saturation
- Idle missed opportunities
- Power state distribution
These indicators together provide a multi-dimensional sizing assessment."""

DEFAULT_SCOPE = """## 1. Objective
This report evaluates multiple residential battery storage configurations in order to:
- Identify the optimal storage capacity for the household
- Quantify financial impact versus a no-battery baseline
- Detect structural undersizing or oversizing conditions
- Assess seasonal robustness and power limitations
- Support an investment decision based on measurable indicators
The goal is to select the smallest battery configuration that provides strong financial and operational performance without structural limitations.

## 2. What This Report Covers
This report includes:
- Monthly and seasonal energy behavior analysis
- Financial gain comparison versus no battery
- Grid import and export reduction metrics
- Battery utilization indicators (cycles, throughput, activity duration)
- Energy saturation indicators (battery full / empty share)
- Structural undersizing metrics (daily and evening)
- Power saturation indicators (active and idle constraints)
All results are derived from simulation based on real household load measurements and applied tariff rules.

## 3. What This Report Does Not Cover
This report does not include:
- Hardware degradation modeling beyond maximum cycle assumptions
- Future tariff changes or regulatory impacts
- Dynamic market price arbitrage
- Backup power reliability analysis
- Installation costs or electrical infrastructure upgrades
- Financing structure or tax optimization
The analysis is strictly based on operational performance and direct tariff-based financial impact.

## 4. Target Audience
This report is intended for:
- Homeowners evaluating battery investments
- Energy optimization enthusiasts
- Technical decision makers
- Financial decision makers assessing ROI
- System designers reviewing sizing trade-offs
The document is structured to support both technical and business-level evaluation."""

DEFAULT_DATA_REQUIREMENTS = """To ensure accurate and meaningful results, the simulation requires:

- Historical household consumption measurements without a battery installed
- Grid import and export power data
- 3-phase measurements (A, B, C) with timestamp granularity
- A defined battery configuration for each tested scenario

Measurements can be collected using:

- A 3-phase energy meter such as Shelly 3EM
- Integration with Home Assistant for data logging and export

The simulator uses this real-world baseline to perform controlled charge/discharge modeling under defined tariff rules.

Accurate input data is essential, as simulation quality directly depends on measurement quality."""


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_caption(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def normalize_power(values: list[int | float] | None) -> str:
    if not values:
        return "-"
    if all(v == values[0] for v in values):
        return str(values[0])
    return "/".join(str(v) for v in values)


def extract_config_payload(data: dict) -> dict:
    if "battery" in data and "tariff" in data:
        return data
    if "configuration" in data and "battery" in data["configuration"] and "tariff" in data["configuration"]:
        return data["configuration"]
    raise ValueError("Unsupported config schema (expected config JSON or simulation output JSON).")


def load_configuration_rows(config_paths: list[Path]) -> list[dict]:
    rows = []
    for path in sorted(config_paths):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = extract_config_payload(raw)

        battery = cfg["battery"]
        tariff = cfg["tariff"]
        capacity_per_phase = battery.get("capacity_Wh_per_phase", [0, 0, 0])
        total_kwh = sum(capacity_per_phase) / 1000 if capacity_per_phase else 0
        peak = tariff.get("peak", {})
        off_peak = tariff.get("off_peak", {})

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
                    ("Peak tariff consume / inject", f"{peak.get('tariff_consume', '-')} / {peak.get('tariff_inject', '-')}"),
                    (
                        "Off-peak tariff consume / inject",
                        f"{off_peak.get('tariff_consume', '-')} / {off_peak.get('tariff_inject', '-')}",
                    ),
                ],
            }
        )
    return rows


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
        "Configuration Used",
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

    top_slots = [(0.06, 0.54, 0.88, 0.35), (0.06, 0.10, 0.88, 0.35)]
    for slot, image_path in zip(top_slots, images):
        left, bottom, width, height = slot
        ax = fig.add_axes([left, bottom, width, height])
        ax.axis("off")

        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_aspect("auto")
        ax.text(
            0.0,
            1.02,
            clean_caption(image_path),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#374151",
        )

    ax_bg.text(
        0.06,
        0.03,
        "Battery simulation notebook export",
        ha="left",
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
    configuration_rows: list[dict],
    monthly_images: list[Path],
    seasonal_images: list[Path],
) -> None:
    section_specs = [
        ("Monthly Graphs", monthly_images),
        ("Seasonal Graphs", seasonal_images),
    ]
    intro_pages = paginate_structured_lines(parse_structured_text(intro_text)) if intro_text.strip() else []
    methodology_pages = paginate_structured_lines(parse_structured_text(methodology_text)) if methodology_text.strip() else []
    scope_pages = paginate_structured_lines(parse_structured_text(scope_text)) if scope_text.strip() else []
    data_requirements_pages = (
        paginate_structured_lines(parse_structured_text(data_requirements_text))
        if data_requirements_text.strip()
        else []
    )
    config_page_count = len(list(chunked(configuration_rows, 2))) if configuration_rows else 0
    section_page_counts = [len(list(chunked(images, 2))) for _, images in section_specs]
    total_pages = (
        1
        + len(intro_pages)
        + len(methodology_pages)
        + len(scope_pages)
        + len(data_requirements_pages)
        + config_page_count
        + sum(section_page_counts)
    )
    page_index = 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        draw_cover(pdf, title, subtitle)
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

        for config_chunk in chunked(configuration_rows, 2):
            draw_config_cards_page(
                pdf=pdf,
                rows=config_chunk,
                page_index=page_index,
                total_pages=total_pages,
            )
            page_index += 1

        for (section_title, images), section_page_count in zip(section_specs, section_page_counts):
            if section_page_count == 0:
                continue
            for page_images in chunked(images, 2):
                draw_image_page(
                    pdf=pdf,
                    section_title=section_title,
                    images=page_images,
                    page_index=page_index,
                    total_pages=total_pages,
                )
                page_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PDF report from monthly and seasonal graph images."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    monthly_images = validate_images(args.monthly, "monthly")
    seasonal_images = validate_images(args.seasonal, "seasonal")
    config_paths = validate_json_paths(args.configs, "config") if args.configs else []
    configuration_rows = load_configuration_rows(config_paths) if config_paths else []
    output_pdf = Path(args.output)

    build_pdf(
        output_pdf=output_pdf,
        title=args.title,
        subtitle=args.subtitle,
        intro_text=args.intro,
        methodology_text=args.methodology,
        scope_text=args.scope,
        data_requirements_text=args.data_requirements,
        configuration_rows=configuration_rows,
        monthly_images=monthly_images,
        seasonal_images=seasonal_images,
    )
    print(f"PDF report generated: {output_pdf}")


if __name__ == "__main__":
    main()
