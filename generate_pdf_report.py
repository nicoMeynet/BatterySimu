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
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_caption(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


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


def build_pdf(
    output_pdf: Path,
    title: str,
    subtitle: str | None,
    monthly_images: list[Path],
    seasonal_images: list[Path],
) -> None:
    section_specs = [
        ("Monthly Graphs", monthly_images),
        ("Seasonal Graphs", seasonal_images),
    ]
    section_page_counts = [len(list(chunked(images, 2))) for _, images in section_specs]
    total_pages = 1 + sum(section_page_counts)
    page_index = 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        draw_cover(pdf, title, subtitle)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    monthly_images = validate_images(args.monthly, "monthly")
    seasonal_images = validate_images(args.seasonal, "seasonal")
    output_pdf = Path(args.output)

    build_pdf(
        output_pdf=output_pdf,
        title=args.title,
        subtitle=args.subtitle,
        monthly_images=monthly_images,
        seasonal_images=seasonal_images,
    )
    print(f"PDF report generated: {output_pdf}")


if __name__ == "__main__":
    main()
