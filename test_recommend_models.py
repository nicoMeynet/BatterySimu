#!/usr/bin/env python3
"""
Run `make recommend` against all local Ollama models and store results.

Example:
  venv/bin/python test_recommend_models.py out/battery_graph_report.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test make recommend with all models from `ollama list`."
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default="out/battery_graph_report.pdf",
        help="Input PDF path passed to RECOMMEND_INPUT_PDF (default: out/battery_graph_report.pdf).",
    )
    parser.add_argument(
        "--output",
        default="out/recommend_model_results.csv",
        help="CSV output path (default: out/recommend_model_results.csv).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="OLLAMA_NUM_CTX value for each run (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="OLLAMA_TEMPERATURE for each run (default: 0.2).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="OLLAMA_TOP_P for each run (default: 0.9).",
    )
    return parser.parse_args()


def get_ollama_models() -> list[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown ollama list error"
        raise RuntimeError(f"`ollama list` failed: {msg}")

    lines = [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]
    # Skip header like: NAME ID SIZE MODIFIED
    models: list[str] = []
    for line in lines[1:]:
        # NAME is first whitespace-separated token
        model = line.split()[0].strip()
        if model:
            models.append(model)
    if not models:
        raise RuntimeError("No models found from `ollama list`.")
    return models


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def run_make_recommend(
    model: str,
    pdf_path: Path,
    num_ctx: int,
    temperature: float,
    top_p: float,
) -> tuple[bool, str]:
    model_file = sanitize_filename(model)
    out_md = Path("out") / f"recommendation_{model_file}.md"
    cmd = [
        "make",
        "recommend",
        f"OLLAMA_MODEL={model}",
        f"RECOMMEND_INPUT_PDF={pdf_path}",
        f"RECOMMEND_OUTPUT={out_md}",
        f"OLLAMA_NUM_CTX={num_ctx}",
        f"OLLAMA_TEMPERATURE={temperature}",
        f"OLLAMA_TOP_P={top_p}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return True, "success"

    # Keep one concise line from stderr/stdout
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    details = stderr.splitlines()[-1].strip() if stderr else ""
    if not details and stdout:
        details = stdout.splitlines()[-1].strip()
    if not details:
        details = f"failed (exit={proc.returncode})"
    return False, details


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    models = get_ollama_models()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    print(f"Found {len(models)} model(s). Running tests...")
    for i, model in enumerate(models, start=1):
        print(f"[{i}/{len(models)}] Testing model: {model}")
        ok, result = run_make_recommend(
            model=model,
            pdf_path=pdf_path,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rows.append({"model": model, "result": result if not ok else "success"})

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "result"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
