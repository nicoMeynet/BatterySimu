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

DEFAULT_INTRO = """## Goal
This report compares battery scenarios to identify an optimal storage size for your home.
The objective is to maximize self-consumption and financial return while avoiding both oversizing and undersizing.

## Simulation Process
- Input data comes from 3-phase house power measurements (A, B, C) exported from Home Assistant.
- For each battery configuration, the simulator runs a timestamp-by-timestamp charge/discharge simulation with tariff rules.
- Results are compared against the no-battery baseline to quantify net gains, import/export reductions, and battery usage behavior.

## How To Read The Graphs
- Monthly charts show short-term variability and help detect transition periods where behavior changes.
- Seasonal charts highlight stable long-term trends and the influence of weather and consumption patterns.
- Comparing both views helps separate one-off effects from structural sizing issues.

## Decision Criteria For Optimal Battery Size
- Financial: prioritize annual gain and amortization time.
- Energy sizing: monitor battery full/empty share and daily/evening undersize percentages.
- Power sizing: check active and idle power saturation to identify inverter power bottlenecks.
- Practical target: choose the smallest scenario that keeps undersize and saturation at acceptable levels while preserving strong financial performance."""


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


def draw_intro_page(
    pdf: PdfPages,
    intro_text: str,
    page_index: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.06,
        0.965,
        "Introduction",
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
    para_h = 0.012

    for raw_line in intro_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            y -= para_h
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            ax.text(
                x,
                y,
                heading,
                ha="left",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                color="#111827",
            )
            y -= line_h * 1.2
            continue

        if line.startswith("- "):
            bullet_text = line[2:].strip()
            wrapped = textwrap.wrap(bullet_text, width=92)
            if not wrapped:
                continue
            ax.text(
                x,
                y,
                f"• {wrapped[0]}",
                ha="left",
                va="top",
                fontsize=11,
                color="#374151",
            )
            y -= line_h
            for cont in wrapped[1:]:
                ax.text(
                    x + 0.02,
                    y,
                    cont,
                    ha="left",
                    va="top",
                    fontsize=11,
                    color="#374151",
                )
                y -= line_h
            continue

        wrapped = textwrap.wrap(line, width=95)
        for wline in wrapped:
            ax.text(
                x,
                y,
                wline,
                ha="left",
                va="top",
                fontsize=11,
                color="#374151",
            )
            y -= line_h

    pdf.savefig(fig, dpi=220)
    plt.close(fig)


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
    configuration_rows: list[dict],
    monthly_images: list[Path],
    seasonal_images: list[Path],
) -> None:
    section_specs = [
        ("Monthly Graphs", monthly_images),
        ("Seasonal Graphs", seasonal_images),
    ]
    config_page_count = len(list(chunked(configuration_rows, 2))) if configuration_rows else 0
    section_page_counts = [len(list(chunked(images, 2))) for _, images in section_specs]
    total_pages = 1 + 1 + config_page_count + sum(section_page_counts)
    page_index = 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        draw_cover(pdf, title, subtitle)
        page_index += 1
        draw_intro_page(pdf, intro_text, page_index, total_pages)
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
        configuration_rows=configuration_rows,
        monthly_images=monthly_images,
        seasonal_images=seasonal_images,
    )
    print(f"PDF report generated: {output_pdf}")


if __name__ == "__main__":
    main()
