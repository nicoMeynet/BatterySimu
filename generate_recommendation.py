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


RESPONSE_SCHEMA_TEXT = """{
  "recommended_configuration": "string",
  "recommended_total_kwh": "string",
  "executive_summary": "string",
  "economically": ["bullet 1", "bullet 2", "..."],
  "energetically": ["bullet 1", "bullet 2", "..."],
  "system_balance": ["bullet 1", "bullet 2", "..."],
  "data_gaps": ["bullet 1", "..."]
}"""


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
{schema_text}

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
        schema_text=RESPONSE_SCHEMA_TEXT,
        report_text=report_text,
        kpi_summary_json_text=kpi_summary_json_text or "not provided",
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


def _compact_kpi_summary_json_for_prompt(raw_json_text: str) -> str:
    """
    Reduce KPI summary JSON to the most decision-relevant parts to stay within model context.
    If parsing fails, return the raw text unchanged (caller will clip).
    """
    try:
        data = json.loads(raw_json_text)
    except Exception:
        return raw_json_text

    compact: dict[str, object] = {
        "generated_at_utc": data.get("generated_at_utc"),
        "baseline": data.get("baseline"),
        "knee_point": data.get("knee_point"),
        "best_by_metric": data.get("best_by_metric"),
        "decision_profiles_best_by_profile": data.get("decision_profiles", {}).get("best_by_profile", {}),
        "seasonal_smallest_good_enough": data.get("seasonal_smallest_good_enough", {}),
        "seasonal_knee_points": data.get("seasonal_knee_points", {}),
    }

    rankings = data.get("decision_profiles", {}).get("rankings", {})
    compact_rankings: dict[str, object] = {}
    if isinstance(rankings, dict):
        for profile_name, rows in rankings.items():
            if not isinstance(rows, list):
                continue
            compact_rows = []
            for row in rows[:3]:
                if not isinstance(row, dict):
                    continue
                compact_rows.append(
                    {
                        "rank": row.get("rank"),
                        "scenario": row.get("scenario"),
                        "battery_size_kwh": row.get("battery_size_kwh"),
                        "score": row.get("score"),
                        "key_metrics": row.get("key_metrics", {}),
                    }
                )
            compact_rankings[profile_name] = compact_rows
    compact["decision_profiles_top3"] = compact_rankings

    candidates = data.get("candidates", [])
    if isinstance(candidates, list):
        compact_candidates = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            compact_candidates.append(
                {
                    "scenario": cand.get("scenario"),
                    "battery_size_kwh": cand.get("battery_size_kwh"),
                    "is_baseline": cand.get("is_baseline"),
                    "metrics": {
                        k: cand.get("metrics", {}).get(k)
                        for k in [
                            "annualized_gain_chf",
                            "amortization_years",
                            "bill_reduction_pct_vs_no_battery",
                            "grid_consumed_reduction_pct_vs_no_battery",
                            "max_season_evening_undersize_pct",
                            "max_season_energy_undersize_pct",
                            "winter_total_gain_chf",
                        ]
                    },
                    "derived": {
                        "marginal_annualized_gain_chf_per_added_kwh": cand.get("derived", {}).get(
                            "marginal_annualized_gain_chf_per_added_kwh"
                        )
                    },
                }
            )
        compact["candidates_compact"] = compact_candidates

    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _shrink_kpi_json_text_for_prompt(kpi_json_text: str, max_chars: int) -> str:
    """
    Keep KPI JSON valid while shrinking for prompt budget.
    Prefer preserving winners/knee points and reduce candidate/ranking detail first.
    """
    if not kpi_json_text or len(kpi_json_text) <= max_chars:
        return kpi_json_text
    try:
        data = json.loads(kpi_json_text)
    except Exception:
        # Last resort: raw trim. (Should rarely happen because caller compacts first.)
        return kpi_json_text[:max_chars]

    def dumps(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # Tier 0: already compact object.
    out = dumps(data)
    if len(out) <= max_chars:
        return out

    # Tier 1: reduce profile ranking depth.
    tier = dict(data)
    if isinstance(tier.get("decision_profiles_top3"), dict):
        reduced_rankings = {}
        for profile, rows in tier["decision_profiles_top3"].items():
            if isinstance(rows, list):
                reduced_rows = []
                for row in rows[:2]:
                    if isinstance(row, dict):
                        reduced_rows.append(
                            {
                                "rank": row.get("rank"),
                                "scenario": row.get("scenario"),
                                "battery_size_kwh": row.get("battery_size_kwh"),
                                "score": row.get("score"),
                            }
                        )
                reduced_rankings[profile] = reduced_rows
        tier["decision_profiles_top3"] = reduced_rankings
    out = dumps(tier)
    if len(out) <= max_chars:
        return out

    # Tier 2: flatten candidate detail to a minimal per-candidate table.
    tier2 = dict(tier)
    candidates = tier2.pop("candidates_compact", None)
    if isinstance(candidates, list):
        minimal_candidates = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            metrics = cand.get("metrics", {}) if isinstance(cand.get("metrics"), dict) else {}
            derived = cand.get("derived", {}) if isinstance(cand.get("derived"), dict) else {}
            minimal_candidates.append(
                {
                    "scenario": cand.get("scenario"),
                    "battery_size_kwh": cand.get("battery_size_kwh"),
                    "is_baseline": cand.get("is_baseline"),
                    "annualized_gain_chf": metrics.get("annualized_gain_chf"),
                    "amortization_years": metrics.get("amortization_years"),
                    "bill_reduction_pct": metrics.get("bill_reduction_pct_vs_no_battery"),
                    "grid_consumed_reduction_pct": metrics.get("grid_consumed_reduction_pct_vs_no_battery"),
                    "winter_total_gain_chf": metrics.get("winter_total_gain_chf"),
                    "marginal_gain_chf_per_added_kwh": derived.get("marginal_annualized_gain_chf_per_added_kwh"),
                }
            )
        tier2["candidates_minimal"] = minimal_candidates
    out = dumps(tier2)
    if len(out) <= max_chars:
        return out

    # Tier 3: keep only winners / knee points / seasonal picks.
    keys_to_keep = {
        "generated_at_utc",
        "baseline",
        "knee_point",
        "best_by_metric",
        "decision_profiles_best_by_profile",
        "seasonal_smallest_good_enough",
        "seasonal_knee_points",
    }
    tier3 = {k: v for k, v in tier2.items() if k in keys_to_keep}
    out = dumps(tier3)
    if len(out) <= max_chars:
        return out

    # Tier 4: minimal winners only.
    tier4 = {
        "knee_point": tier3.get("knee_point"),
        "decision_profiles_best_by_profile": tier3.get("decision_profiles_best_by_profile", {}),
        "seasonal_smallest_good_enough": tier3.get("seasonal_smallest_good_enough", {}),
        "seasonal_knee_points": tier3.get("seasonal_knee_points", {}),
    }
    out = dumps(tier4)
    if len(out) <= max_chars:
        return out

    tier5 = {
        "knee_point": (tier4.get("knee_point") or {}),
        "decision_profiles_best_by_profile": (tier4.get("decision_profiles_best_by_profile") or {}),
        "note": "KPI prompt context truncated due model context limit.",
    }
    out = dumps(tier5)
    if len(out) <= max_chars:
        return out

    # Absolute last resort: keep only profile winners and the note.
    tier6 = {
        "decision_profiles_best_by_profile": (tier4.get("decision_profiles_best_by_profile") or {}),
        "note": "KPI prompt context truncated due model context limit.",
    }
    out = dumps(tier6)
    if len(out) <= max_chars:
        return out

    return dumps({"note": "KPI summary omitted from prompt because num_ctx is too small."})


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
    report_text: str,
    kpi_json_text: str,
    kpi_md_text: str,
    num_ctx: int,
    max_report_chars: int,
    max_kpi_chars: int,
) -> tuple[str, str, str, int]:
    """
    Fit prompt inputs to a context-aware character budget.
    Prioritize KPI JSON, then KPI Markdown, then PDF report text.
    """
    prompt_budget = _estimate_prompt_char_budget(num_ctx)
    report = report_text[:max_report_chars] if report_text else ""
    kpi_json = _shrink_kpi_json_text_for_prompt(kpi_json_text, max_kpi_chars) if kpi_json_text else ""
    kpi_md = kpi_md_text[:max_kpi_chars] if kpi_md_text else ""

    prompt = build_prompt(report, kpi_summary_json_text=kpi_json, kpi_summary_markdown_text=kpi_md)
    if len(prompt) <= prompt_budget:
        return report, kpi_json, kpi_md, prompt_budget

    # First trim report aggressively; KPI JSON is more authoritative.
    if report:
        template_without_report = build_prompt("", kpi_summary_json_text=kpi_json, kpi_summary_markdown_text=kpi_md)
        remaining_for_report = max(0, prompt_budget - len(template_without_report))
        report = report[:remaining_for_report]
        prompt = build_prompt(report, kpi_summary_json_text=kpi_json, kpi_summary_markdown_text=kpi_md)
        if len(prompt) <= prompt_budget:
            return report, kpi_json, kpi_md, prompt_budget

    # If still too large, drop KPI Markdown before shrinking KPI JSON.
    if kpi_md:
        kpi_md = ""
        prompt = build_prompt(report, kpi_summary_json_text=kpi_json, kpi_summary_markdown_text=kpi_md)
        if len(prompt) <= prompt_budget:
            return report, kpi_json, kpi_md, prompt_budget

    # If still too large, keep report empty and shrink KPI JSON to fit.
    if kpi_json:
        template_no_inputs = build_prompt("", kpi_summary_json_text="", kpi_summary_markdown_text="")
        remaining_for_kpi = max(2000, prompt_budget - len(template_no_inputs))
        # Prefer KPI over PDF when budget is tight.
        report = ""
        kpi_json = _shrink_kpi_json_text_for_prompt(kpi_json, remaining_for_kpi)
        prompt = build_prompt(report, kpi_summary_json_text=kpi_json, kpi_summary_markdown_text="")
        if len(prompt) <= prompt_budget:
            return report, kpi_json, "", prompt_budget

    # Final fallback: minimal prompt with whatever fits.
    if len(prompt) > prompt_budget and report:
        report = ""
    return report, kpi_json, kpi_md, prompt_budget


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
        description="Generate battery recommendation from KPI summary and/or PDF report using local Ollama."
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
        default=40000,
        help="Max characters per KPI summary input (JSON/Markdown) to send to the model.",
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
    if kpi_summary_json_text:
        kpi_summary_json_text = _compact_kpi_summary_json_for_prompt(kpi_summary_json_text)

    kpi_summary_markdown_text = ""
    if args.kpi_summary_md:
        kpi_md_path = Path(args.kpi_summary_md)
        if kpi_md_path.exists() and kpi_md_path.is_file():
            kpi_summary_markdown_text = kpi_md_path.read_text(encoding="utf-8")
            print(f"Loaded KPI summary Markdown: {kpi_md_path}")
        else:
            print(f"Warning: KPI summary Markdown not found, skipping: {kpi_md_path}", file=sys.stderr)

    # Avoid duplicating KPI context when JSON is already provided.
    if kpi_summary_json_text and kpi_summary_markdown_text:
        print("Info: KPI summary JSON provided; skipping KPI Markdown in prompt to reduce context duplication.")
        kpi_summary_markdown_text = ""

    if not report_text and not kpi_summary_json_text and not kpi_summary_markdown_text:
        raise FileNotFoundError(
            "No usable input context found. Provide a PDF report and/or KPI summary (--kpi-summary-json / --kpi-summary-md)."
        )

    clipped_report, clipped_kpi_json, clipped_kpi_md, prompt_budget = _fit_prompt_inputs_to_budget(
        report_text=report_text,
        kpi_json_text=kpi_summary_json_text,
        kpi_md_text=kpi_summary_markdown_text,
        num_ctx=args.num_ctx,
        max_report_chars=args.max_chars,
        max_kpi_chars=args.max_kpi_chars,
    )
    if report_text:
        print(f"Using PDF report text chars in prompt: {len(clipped_report)} / {min(len(report_text), args.max_chars)}")
    if kpi_summary_json_text:
        print(
            f"Using KPI JSON chars in prompt: {len(clipped_kpi_json)} / "
            f"{min(len(kpi_summary_json_text), args.max_kpi_chars)}"
        )
    if kpi_summary_markdown_text:
        print(
            f"Using KPI Markdown chars in prompt: {len(clipped_kpi_md)} / "
            f"{min(len(kpi_summary_markdown_text), args.max_kpi_chars)}"
        )
    prompt = build_prompt(
        clipped_report,
        kpi_summary_json_text=clipped_kpi_json,
        kpi_summary_markdown_text=clipped_kpi_md,
    )
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
