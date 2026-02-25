#!/usr/bin/env python3
"""
Generate battery sizing recommendations from a PDF report and/or KPI summary using a local Ollama model.

Examples:
  python generate_recommendation.py out/battery_graph_report.pdf --model llama3.1
  python generate_recommendation.py --kpi-summary-json out/kpi_summary/kpi_summary.json --model llama3.1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF file."""
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def build_prompt(report_text: str, *, kpi_summary_json_text: str = "", kpi_summary_markdown_text: str = "") -> str:
    """Build a recommendation prompt from report text and structured KPI summary context."""
    template = """
You are an expert energy storage analyst.
Use the available report content and KPI summary context below to produce a practical battery recommendation.

Return ONLY a valid JSON object (no markdown, no comments) with this schema:
{{
  "recommended_configuration": "string",
  "recommended_total_kwh": "string",
  "executive_summary": "string",
  "economically": ["bullet 1", "bullet 2", "..."],
  "energetically": ["bullet 1", "bullet 2", "..."],
  "system_balance": ["bullet 1", "bullet 2", "..."],
  "data_gaps": ["bullet 1", "..."]
}}

Rules:
- Base recommendation only on provided evidence.
- If KPI Summary JSON is provided, treat it as the authoritative numeric source for rankings, thresholds and winners.
- Use PDF/report text mainly for qualitative context and consistency checks.
- If there is a conflict between PDF narrative and KPI Summary JSON values, prefer the KPI Summary JSON values.
- If a metric is missing, explicitly write "not available in report".
- Do not reference page numbers.
- Keep each bullet concise and professional.

KPI Summary JSON (optional):
---
{kpi_summary_json_text}
---

KPI Summary Markdown (optional):
---
{kpi_summary_markdown_text}
---

Report content extracted from PDF (optional):
---
{report_text}
---
"""
    return template.format(
        report_text=report_text,
        kpi_summary_json_text=kpi_summary_json_text or "not provided",
        kpi_summary_markdown_text=kpi_summary_markdown_text or "not provided",
    )


def call_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    num_ctx: int,
    timeout_s: int = 300,
) -> str:
    """Call local Ollama HTTP API and return generated text."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    generate_url = f"{ollama_host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        url=generate_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        details = body.strip() or f"HTTP {exc.code}"
        if "model" in details.lower() and "not found" in details.lower():
            raise RuntimeError(
                f"Ollama model '{model}' is not available.\n"
                f"Fix:\n"
                f"  1) ollama list\n"
                f"  2) ollama pull {model}\n"
                f"  3) retry: make recommend"
            ) from exc
        raise RuntimeError(
            f"Ollama API error at {generate_url}: {details}\n"
            f"Fix:\n"
            f"  - Ensure Ollama is running: ollama serve\n"
            f"  - Verify API: curl {ollama_host}/api/tags"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {ollama_host}.\n"
            f"Reason: {exc.reason}\n"
            f"Fix:\n"
            f"  1) Start Ollama: ollama serve\n"
            f"  2) Check connectivity: curl {ollama_host}/api/tags\n"
            f"  3) Check model exists: ollama list\n"
            f"  4) If missing, pull it: ollama pull {model}"
        ) from exc

    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return data["response"].strip()


def _extract_json_object(raw: str) -> dict:
    """Extract and parse first JSON object from model output."""
    text = raw.strip()
    # Remove fenced blocks if present.
    text = re.sub(r"^```(?:json)?\\s*", "", text)
    text = re.sub(r"\\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab first balanced {...} block.
    start = text.find("{")
    if start == -1:
        raise RuntimeError("Model did not return JSON content.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Unable to parse JSON recommendation from model output.") from exc
    raise RuntimeError("Unable to locate complete JSON object in model output.")


def _to_bullets(value: object) -> list[str]:
    if isinstance(value, list):
        out = []
        for v in value:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return ["not available in report"]


def format_recommendation(data: dict) -> str:
    """Render a professional recommendation text block from structured JSON."""
    name = str(data.get("recommended_configuration", "not available in report")).strip()
    total = str(data.get("recommended_total_kwh", "not available in report")).strip()
    summary = str(data.get("executive_summary", "")).strip()

    economically = _to_bullets(data.get("economically"))
    energetically = _to_bullets(data.get("energetically"))
    system_balance = _to_bullets(data.get("system_balance"))
    data_gaps = _to_bullets(data.get("data_gaps"))

    lines: list[str] = []
    lines.append(f"Recommended Configuration: {name} ({total})")
    lines.append("")
    if summary:
        lines.append("## Executive Summary")
        lines.append(summary)
        lines.append("")
    lines.append("## Economically")
    lines.extend(f"- {x}" for x in economically)
    lines.append("")
    lines.append("## Energetically")
    lines.extend(f"- {x}" for x in energetically)
    lines.append("")
    lines.append("## System Balance")
    lines.extend(f"- {x}" for x in system_balance)
    lines.append("")
    lines.append("## Data Gaps")
    lines.extend(f"- {x}" for x in data_gaps)
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate battery recommendation from a PDF report using local Ollama."
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default="out/battery_graph_report.pdf",
        help="Path to input PDF report (default: out/battery_graph_report.pdf). Optional if KPI summary is provided.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model name (default: llama3.1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file (.md/.txt). If omitted, prints to stdout only.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Ollama generation (default: 0.2).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p value (default: 0.9).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="Context window size to request from Ollama (default: 32768).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=120000,
        help="Max number of characters from the report to send to the model.",
    )
    parser.add_argument(
        "--kpi-summary-json",
        default=None,
        help="Optional KPI summary JSON file (e.g. out/kpi_summary/kpi_summary.json).",
    )
    parser.add_argument(
        "--kpi-summary-md",
        default=None,
        help="Optional KPI summary Markdown file (e.g. out/kpi_summary/kpi_summary.md).",
    )
    parser.add_argument(
        "--max-kpi-chars",
        type=int,
        default=80000,
        help="Max characters per KPI summary input (JSON/Markdown) to send to the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_file)
    report_text = ""
    if pdf_path.exists() and pdf_path.is_file():
        report_text = extract_pdf_text(pdf_path)
        if not report_text:
            print(f"Warning: No extractable text found in PDF: {pdf_path}", file=sys.stderr)
        else:
            print(f"Loaded PDF report text: {pdf_path}")
    else:
        print(f"Warning: PDF file not found, continuing without PDF context: {pdf_path}", file=sys.stderr)

    kpi_summary_json_text = ""
    if args.kpi_summary_json:
        kpi_json_path = Path(args.kpi_summary_json)
        if kpi_json_path.exists() and kpi_json_path.is_file():
            kpi_summary_json_text = kpi_json_path.read_text(encoding="utf-8")
            print(f"Loaded KPI summary JSON: {kpi_json_path}")
        else:
            print(f"Warning: KPI summary JSON not found, skipping: {kpi_json_path}", file=sys.stderr)

    kpi_summary_markdown_text = ""
    if args.kpi_summary_md:
        kpi_md_path = Path(args.kpi_summary_md)
        if kpi_md_path.exists() and kpi_md_path.is_file():
            kpi_summary_markdown_text = kpi_md_path.read_text(encoding="utf-8")
            print(f"Loaded KPI summary Markdown: {kpi_md_path}")
        else:
            print(f"Warning: KPI summary Markdown not found, skipping: {kpi_md_path}", file=sys.stderr)

    if not report_text and not kpi_summary_json_text and not kpi_summary_markdown_text:
        raise FileNotFoundError(
            "No usable input context found. Provide a PDF report and/or KPI summary (--kpi-summary-json / --kpi-summary-md)."
        )

    clipped_report = report_text[: args.max_chars] if report_text else ""
    clipped_kpi_json = kpi_summary_json_text[: args.max_kpi_chars] if kpi_summary_json_text else ""
    clipped_kpi_md = kpi_summary_markdown_text[: args.max_kpi_chars] if kpi_summary_markdown_text else ""
    prompt = build_prompt(
        clipped_report,
        kpi_summary_json_text=clipped_kpi_json,
        kpi_summary_markdown_text=clipped_kpi_md,
    )
    raw = call_ollama(
        args.model,
        prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
    )
    structured = _extract_json_object(raw)
    recommendation = format_recommendation(structured)

    print(recommendation)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(recommendation + "\n", encoding="utf-8")
        print(f"\nSaved recommendation to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
