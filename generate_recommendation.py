#!/usr/bin/env python3
"""
Generate battery sizing recommendations from KPI summary Markdown using a local Ollama model.

Example:
  python generate_recommendation.py --kpi-summary-md out/kpi_summary/kpi_summary.md --model llama3.1
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

RESPONSE_SCHEMA_TEXT = """{
  "recommended_configuration": "string",
  "recommended_total_kwh": "string",
  "executive_summary": "string",
  "economically": ["bullet 1", "bullet 2", "..."],
  "energetically": ["bullet 1", "bullet 2", "..."],
  "system_balance": ["bullet 1", "bullet 2", "..."],
  "data_gaps": ["bullet 1", "..."]
}"""
def build_prompt(kpi_summary_markdown_text: str) -> str:
    """Build a recommendation prompt from the graph-KPI summary Markdown."""
    template = """
You are an expert energy storage analyst.
You are given a KPI summary Markdown document generated from battery simulation results.
The document is graph-KPI-focused and contains the decision constraints and step-delta rules to use.

Return ONLY a valid JSON object (no markdown, no comments) with this schema:
{schema_text}

How to use the KPI Markdown:
- Treat all sections with "(Constraint Cap)" as hard filters/exclusions first.
- Treat step-delta sections as "meaningful incremental gain" rules (last size that still adds enough value).
- If multiple battery sizes remain valid after constraints, choose one and explain the tradeoff explicitly.
- Prefer exact scenario names, battery sizes, thresholds, and violating examples as written in the Markdown.
- Do not invent metrics or values that are not present in the KPI Markdown.
- If information is missing, explicitly write "not available in report".
- Keep each bullet concise and professional.

Important graph-KPI interpretation:
- `Global Energy Reduction (Consumed % Step Delta)` and `Global Energy Financial Impact (Bill Offset % Step Delta)` are incremental-value KPIs.
- `Global Rentability Overview (Amortization Cap)` is an affordability constraint.
- `Global Battery Utilization (Cycle Wear Cap)` is a durability constraint.
- `Global Battery Status Heatmap (Empty Share Cap)` is an availability/comfort constraint.
- `Seasonal Power Saturation At Max Limit (Constraint Cap)` is a seasonal power bottleneck constraint.
- `Monthly Structural Evening Energy Undersizing Peak Period (Constraint Cap)` is a monthly peak-period adequacy constraint.
- The Markdown "Notes for LLM Recommendation" section is part of the authoritative guidance and should be followed.

KPI Summary Markdown (graph-KPI-only):
---
{kpi_summary_markdown_text}
---
"""
    return template.format(
        schema_text=RESPONSE_SCHEMA_TEXT,
        kpi_summary_markdown_text=kpi_summary_markdown_text or "not provided",
    )


def build_json_repair_prompt(raw_model_output: str) -> str:
    """Ask the model to convert its own non-JSON answer into the required JSON schema."""
    template = """
Convert the following battery recommendation text into a valid JSON object ONLY (no markdown, no comments).

Required schema:
{schema_text}

Rules:
- Preserve the original recommendation intent.
- If information is missing, fill with "not available in report" (or ["not available in report"] for arrays).
- Return a single valid JSON object only.

Text to convert:
---
{raw_model_output}
---
"""
    return template.format(schema_text=RESPONSE_SCHEMA_TEXT, raw_model_output=raw_model_output)


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
    # Remove model reasoning blocks sometimes emitted by local models.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    # Remove fenced blocks if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

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


def _estimate_prompt_char_budget(num_ctx: int) -> int:
    """
    Roughly estimate a safe input character budget for the model prompt from num_ctx.
    Keeps reserve for the response + safety margin to avoid context truncation.
    """
    if num_ctx <= 0:
        return 12000
    reserve_for_output_tokens = max(768, min(2048, num_ctx // 4))
    reserve_for_safety_tokens = max(512, min(1024, num_ctx // 8))
    usable_tokens = max(1024, num_ctx - reserve_for_output_tokens - reserve_for_safety_tokens)
    # Conservative chars/token heuristic for mixed prose + JSON.
    return usable_tokens * 3


def _fit_prompt_inputs_to_budget(
    *,
    kpi_md_text: str,
    num_ctx: int,
    max_kpi_chars: int,
) -> tuple[str, int]:
    """
    Fit KPI Markdown input to a context-aware character budget.
    """
    prompt_budget = _estimate_prompt_char_budget(num_ctx)
    kpi_md = kpi_md_text[:max_kpi_chars] if kpi_md_text else ""
    prompt = build_prompt(kpi_md)
    if len(prompt) <= prompt_budget:
        return kpi_md, prompt_budget

    # Reserve the fixed prompt template overhead, then trim Markdown content.
    template_without_md = build_prompt("")
    remaining_for_md = max(1000, prompt_budget - len(template_without_md))
    kpi_md = kpi_md[:remaining_for_md]
    return kpi_md, prompt_budget


def _default_raw_debug_path(output_path: Path | None) -> Path:
    if output_path is not None:
        suffix = output_path.suffix or ".txt"
        return output_path.with_suffix(suffix + ".raw_model.txt")
    return Path("out/simulation_llm_recommendation/recommendation_last.raw_model.txt")


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
        description="Generate battery recommendation from KPI summary Markdown using local Ollama."
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
        "--kpi-summary-md",
        default="out/kpi_summary/kpi_summary.md",
        help="KPI summary Markdown file (default: out/kpi_summary/kpi_summary.md).",
    )
    parser.add_argument(
        "--max-kpi-chars",
        type=int,
        default=40000,
        help="Max characters from KPI summary Markdown to send to the model.",
    )
    parser.add_argument(
        "--raw-debug-output",
        default=None,
        help="Optional file path to save raw model output (useful when JSON parsing fails).",
    )
    parser.add_argument(
        "--no-json-repair",
        action="store_true",
        help="Disable automatic second-pass repair when the model does not return JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kpi_summary_markdown_text = ""
    if args.kpi_summary_md:
        kpi_md_path = Path(args.kpi_summary_md)
        if kpi_md_path.exists() and kpi_md_path.is_file():
            kpi_summary_markdown_text = kpi_md_path.read_text(encoding="utf-8")
            print(f"Loaded KPI summary Markdown: {kpi_md_path}")
        else:
            raise FileNotFoundError(f"KPI summary Markdown not found: {kpi_md_path}")

    if not kpi_summary_markdown_text:
        raise FileNotFoundError("No usable input context found. Provide --kpi-summary-md.")

    clipped_kpi_md, prompt_budget = _fit_prompt_inputs_to_budget(
        kpi_md_text=kpi_summary_markdown_text,
        num_ctx=args.num_ctx,
        max_kpi_chars=args.max_kpi_chars,
    )
    if kpi_summary_markdown_text:
        print(
            f"Using KPI Markdown chars in prompt: {len(clipped_kpi_md)} / "
            f"{min(len(kpi_summary_markdown_text), args.max_kpi_chars)}"
        )
    prompt = build_prompt(clipped_kpi_md)
    if len(prompt) > prompt_budget:
        print(
            "Warning: Prompt still exceeds estimated safe budget "
            f"({len(prompt)} chars > ~{prompt_budget} chars for num_ctx={args.num_ctx}).",
            file=sys.stderr,
        )
    raw = call_ollama(
        args.model,
        prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
    )
    raw_debug_path = Path(args.raw_debug_output) if args.raw_debug_output else _default_raw_debug_path(
        Path(args.output) if args.output else None
    )
    raw_debug_path.parent.mkdir(parents=True, exist_ok=True)
    raw_debug_path.write_text(raw + "\n", encoding="utf-8")

    try:
        structured = _extract_json_object(raw)
    except Exception as first_exc:
        if args.no_json_repair:
            raise RuntimeError(f"{first_exc} (raw model output saved to {raw_debug_path})") from first_exc
        print(
            f"Warning: Model did not return valid JSON. Attempting repair pass... (raw saved to {raw_debug_path})",
            file=sys.stderr,
        )
        repair_source = raw[: max(8000, min(len(raw), 40000))]
        # If the first answer is empty or mostly non-JSON prose, ask again from authoritative KPI data only.
        if not repair_source.strip() and clipped_kpi_json:
            repair_source = (
                "No usable answer returned in first pass. Reconstruct a recommendation from this KPI context:\n"
                + clipped_kpi_json[: max(4000, min(len(clipped_kpi_json), 20000))]
            )
        repair_prompt = build_json_repair_prompt(repair_source)
        repaired_raw = call_ollama(
            args.model,
            repair_prompt,
            temperature=0.0,
            top_p=0.9,
            num_ctx=args.num_ctx,
        )
        repaired_debug_path = raw_debug_path.with_name(raw_debug_path.stem + ".repaired" + raw_debug_path.suffix)
        repaired_debug_path.write_text(repaired_raw + "\n", encoding="utf-8")
        try:
            structured = _extract_json_object(repaired_raw)
        except Exception as second_exc:
            raise RuntimeError(
                "Model did not return JSON content, and repair pass also failed. "
                f"Raw outputs saved to {raw_debug_path} and {repaired_debug_path}"
            ) from second_exc
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
