#!/usr/bin/env python3
"""
Compute deterministic KPI rankings from simulation output JSON files.

Outputs:
  - JSON summary for machine/LLM consumption
  - Markdown summary for human review (and optional PDF embedding later)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SEASON_ORDER = ["winter", "spring", "summer", "autumn"]


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def round_or_none(value: float | None, ndigits: int = 2) -> float | None:
    return round(value, ndigits) if value is not None else None


def nested_get(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def nested_float(obj: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value = safe_float(nested_get(obj, *keys))
    return default if value is None else value


def battery_size_kwh(data: dict[str, Any]) -> float:
    caps = nested_get(data, "configuration", "battery", "capacity_Wh_per_phase", default=[0, 0, 0])
    if not isinstance(caps, list):
        return 0.0
    total_wh = 0.0
    for v in caps:
        fv = safe_float(v)
        if fv is not None:
            total_wh += fv
    return total_wh / 1000.0


def season_metrics_from_blob(values: dict[str, Any]) -> dict[str, float]:
    flows = values.get("energy_flows", {}) or {}
    with_batt = flows.get("with_battery", {}) or {}
    without_batt = flows.get("without_battery", {}) or {}
    saturation = values.get("battery_saturation", {}) or {}
    energy_undersize = values.get("battery_energy_undersize_days", {}) or {}
    evening_undersize = values.get("battery_evening_undersize_days", {}) or {}
    pu = values.get("power_usage", {}) or {}

    charging_at_max = nested_float(pu, "charging", "at_max", "samples_percent", default=0.0)
    discharging_at_max = nested_float(pu, "discharging", "at_max", "samples_percent", default=0.0)
    idle_could_charge_at_max = nested_float(pu, "idle", "could_charge", "at_max", "samples_percent", default=0.0)
    idle_could_discharge_at_max = nested_float(pu, "idle", "could_discharge", "at_max", "samples_percent", default=0.0)

    grid_consumed_with_kwh = nested_float(with_batt, "grid_consumed_kwh", default=0.0)
    grid_consumed_without_kwh = nested_float(without_batt, "grid_consumed_kwh", default=0.0)
    grid_injected_with_kwh = nested_float(with_batt, "grid_injected_kwh", default=0.0)
    grid_injected_without_kwh = nested_float(without_batt, "grid_injected_kwh", default=0.0)

    return {
        "total_gain_chf": nested_float(values, "total_gain_chf", default=0.0),
        "average_monthly_gain_chf": nested_float(values, "average_monthly_gain_chf", default=0.0),
        "grid_consumed_with_kwh": grid_consumed_with_kwh,
        "grid_consumed_without_kwh": grid_consumed_without_kwh,
        "grid_import_reduction_kwh": grid_consumed_without_kwh - grid_consumed_with_kwh,
        "grid_injected_with_kwh": grid_injected_with_kwh,
        "grid_injected_without_kwh": grid_injected_without_kwh,
        "grid_export_reduction_kwh": grid_injected_without_kwh - grid_injected_with_kwh,
        "battery_charged_kwh": nested_float(with_batt, "battery_charged_kwh", default=0.0),
        "battery_discharged_kwh": nested_float(with_batt, "battery_discharged_kwh", default=0.0),
        "avg_full_pct": nested_float(saturation, "average_full_share_percent", default=0.0),
        "avg_empty_pct": nested_float(saturation, "average_empty_share_percent", default=0.0),
        "energy_undersize_pct": nested_float(energy_undersize, "percent", default=0.0),
        "evening_undersize_pct": nested_float(evening_undersize, "percent", default=0.0),
        "power_at_max_pct": charging_at_max
        + discharging_at_max
        + idle_could_charge_at_max
        + idle_could_discharge_at_max,
        "active_power_at_max_pct": charging_at_max + discharging_at_max,
        "idle_power_limited_pct": idle_could_charge_at_max + idle_could_discharge_at_max,
    }


def aggregate_season_metrics(seasons: dict[str, dict[str, float]]) -> dict[str, float | None]:
    if not seasons:
        return {}

    def vals(key: str) -> list[float]:
        return [float(seasons[s].get(key, 0.0)) for s in seasons]

    gains = vals("total_gain_chf")
    energy_undersize = vals("energy_undersize_pct")
    evening_undersize = vals("evening_undersize_pct")
    empty = vals("avg_empty_pct")
    full = vals("avg_full_pct")
    power_at_max = vals("power_at_max_pct")
    active_power_at_max = vals("active_power_at_max_pct")
    idle_power_limited = vals("idle_power_limited_pct")

    winter = seasons.get("winter", {})

    return {
        "season_count": float(len(seasons)),
        "avg_season_gain_chf": round_or_none(mean(gains)),
        "min_season_gain_chf": round_or_none(min(gains)),
        "max_season_gain_chf": round_or_none(max(gains)),
        "season_gain_spread_chf": round_or_none(max(gains) - min(gains)),
        "max_season_energy_undersize_pct": round_or_none(max(energy_undersize)),
        "max_season_evening_undersize_pct": round_or_none(max(evening_undersize)),
        "avg_season_avg_empty_pct": round_or_none(mean(empty)),
        "max_season_avg_empty_pct": round_or_none(max(empty)),
        "avg_season_avg_full_pct": round_or_none(mean(full)),
        "max_season_avg_full_pct": round_or_none(max(full)),
        "avg_season_power_at_max_pct": round_or_none(mean(power_at_max)),
        "max_season_power_at_max_pct": round_or_none(max(power_at_max)),
        "avg_season_active_power_at_max_pct": round_or_none(mean(active_power_at_max)),
        "max_season_active_power_at_max_pct": round_or_none(max(active_power_at_max)),
        "avg_season_idle_power_limited_pct": round_or_none(mean(idle_power_limited)),
        "max_season_idle_power_limited_pct": round_or_none(max(idle_power_limited)),
        "winter_total_gain_chf": round_or_none(safe_float(winter.get("total_gain_chf")) or 0.0),
        "winter_energy_undersize_pct": round_or_none(safe_float(winter.get("energy_undersize_pct")) or 0.0),
        "winter_evening_undersize_pct": round_or_none(safe_float(winter.get("evening_undersize_pct")) or 0.0),
        "winter_avg_empty_pct": round_or_none(safe_float(winter.get("avg_empty_pct")) or 0.0),
        "winter_power_at_max_pct": round_or_none(safe_float(winter.get("power_at_max_pct")) or 0.0),
    }


def build_candidate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem.replace("config_", "")
    size_kwh = round(battery_size_kwh(data), 2)
    is_baseline = size_kwh == 0.0 or "nobattery" in scenario.lower()

    rent = nested_get(data, "global", "rentability", default={}) or {}
    seasons_blob = data.get("seasons", {}) or {}

    season_metrics: dict[str, dict[str, float]] = {}
    for season_name, values in seasons_blob.items():
        season_metrics[season_name.lower()] = season_metrics_from_blob(values or {})

    season_agg = aggregate_season_metrics(season_metrics)

    global_metrics = {
        "total_gain_chf": round_or_none(safe_float(rent.get("total_gain_chf")) or 0.0),
        "annualized_gain_chf": round_or_none(safe_float(rent.get("annualized_gain_chf"))),
        "amortization_years": round_or_none(safe_float(rent.get("amortization_years"))),
        "bill_without_battery_chf": round_or_none(safe_float(rent.get("bill_without_battery_chf"))),
        "bill_with_battery_chf": round_or_none(safe_float(rent.get("bill_with_battery_chf"))),
        "bill_reduction_chf": round_or_none(safe_float(rent.get("bill_reduction_chf"))),
        "bill_reduction_pct_vs_no_battery": round_or_none(safe_float(rent.get("bill_reduction_pct_vs_no_battery"))),
        "grid_consumed_without_battery_kwh": round_or_none(safe_float(rent.get("grid_consumed_without_battery_kwh"))),
        "grid_consumed_with_battery_kwh": round_or_none(safe_float(rent.get("grid_consumed_with_battery_kwh"))),
        "grid_consumed_reduction_kwh": round_or_none(safe_float(rent.get("grid_consumed_reduction_kwh"))),
        "grid_consumed_reduction_pct_vs_no_battery": round_or_none(
            safe_float(rent.get("grid_consumed_reduction_pct_vs_no_battery"))
        ),
    }

    # Flatten core metrics for ranking/scoring convenience.
    metrics: dict[str, float | None] = {
        "battery_size_kwh": size_kwh,
        **global_metrics,
    }
    for key, value in season_agg.items():
        metrics[key] = value if isinstance(value, (int, float)) or value is None else safe_float(value)

    return {
        "scenario": scenario,
        "scenario_label": "noBattery" if is_baseline else f"{size_kwh:.2f} kWh",
        "source_file": str(path),
        "is_baseline": is_baseline,
        "battery_size_kwh": size_kwh,
        "global": global_metrics,
        "seasonal_by_season": {
            season: {k: round_or_none(v) for k, v in vals.items()} for season, vals in season_metrics.items()
        },
        "seasonal_aggregate": season_agg,
        "metrics": metrics,
        "derived": {},
    }


def compute_marginals(candidates: list[dict[str, Any]]) -> None:
    ranked = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    prev = None
    for cand in ranked:
        if cand["is_baseline"]:
            cand["derived"].update(
                {
                    "delta_size_vs_previous_kwh": None,
                    "delta_annualized_gain_vs_previous_chf": None,
                    "marginal_annualized_gain_chf_per_added_kwh": None,
                }
            )
            prev = cand
            continue

        if prev is None:
            prev = cand
            continue

        delta_size = cand["battery_size_kwh"] - prev["battery_size_kwh"]
        prev_gain = safe_float(prev["metrics"].get("annualized_gain_chf"))
        cur_gain = safe_float(cand["metrics"].get("annualized_gain_chf"))
        delta_gain = None if prev_gain is None or cur_gain is None else (cur_gain - prev_gain)
        marginal = None
        if delta_gain is not None and delta_size > 0:
            marginal = delta_gain / delta_size

        cand["derived"].update(
            {
                "delta_size_vs_previous_kwh": round_or_none(delta_size),
                "delta_annualized_gain_vs_previous_chf": round_or_none(delta_gain),
                "marginal_annualized_gain_chf_per_added_kwh": round_or_none(marginal),
            }
        )
        prev = cand


def metric_value(cand: dict[str, Any], key: str) -> float | None:
    return safe_float(cand.get("metrics", {}).get(key))


def norm_metric(value: float | None, values: list[float], higher_is_better: bool) -> float | None:
    if value is None:
        return None
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return 0.5
    score = (value - lo) / (hi - lo)
    return score if higher_is_better else 1.0 - score


def rank_profiles(
    candidates: list[dict[str, Any]],
    knee_point: dict[str, Any] | None,
) -> dict[str, Any]:
    nonbaseline = [c for c in candidates if not c["is_baseline"]]
    profiles: dict[str, dict[str, Any]] = {
        "balanced": {
            "description": "Balanced tradeoff across finance, autonomy, robustness, and oversizing risk.",
            "weights": [
                ("annualized_gain_chf", "max", 0.25),
                ("bill_reduction_pct_vs_no_battery", "max", 0.20),
                ("grid_consumed_reduction_pct_vs_no_battery", "max", 0.15),
                ("amortization_years", "min", 0.15),
                ("max_season_evening_undersize_pct", "min", 0.15),
                ("max_season_energy_undersize_pct", "min", 0.05),
                ("avg_season_avg_full_pct", "min", 0.05),
            ],
        },
        "roi_first": {
            "description": "Prioritize financial return and payback speed while avoiding severe seasonal shortfalls.",
            "weights": [
                ("annualized_gain_chf", "max", 0.35),
                ("amortization_years", "min", 0.35),
                ("bill_reduction_pct_vs_no_battery", "max", 0.15),
                ("max_season_evening_undersize_pct", "min", 0.10),
                ("battery_size_kwh", "min", 0.05),
            ],
        },
        "autonomy_first": {
            "description": "Prioritize grid import reduction and seasonal energy adequacy.",
            "weights": [
                ("grid_consumed_reduction_pct_vs_no_battery", "max", 0.30),
                ("bill_reduction_pct_vs_no_battery", "max", 0.10),
                ("max_season_evening_undersize_pct", "min", 0.25),
                ("max_season_energy_undersize_pct", "min", 0.20),
                ("max_season_avg_empty_pct", "min", 0.10),
                ("avg_season_power_at_max_pct", "min", 0.05),
            ],
        },
        "winter_robustness": {
            "description": "Prioritize winter performance and peak-period coverage robustness.",
            "weights": [
                ("winter_total_gain_chf", "max", 0.15),
                ("winter_evening_undersize_pct", "min", 0.30),
                ("winter_energy_undersize_pct", "min", 0.25),
                ("winter_avg_empty_pct", "min", 0.15),
                ("annualized_gain_chf", "max", 0.10),
                ("battery_size_kwh", "min", 0.05),
            ],
        },
    }

    # Pre-compute metric ranges from non-baseline candidates.
    metric_ranges: dict[str, list[float]] = {}
    for profile in profiles.values():
        for key, _, _ in profile["weights"]:
            if key not in metric_ranges:
                vals = [v for v in (metric_value(c, key) for c in nonbaseline) if v is not None]
                metric_ranges[key] = vals

    rankings: dict[str, list[dict[str, Any]]] = {}
    best_by_profile: dict[str, dict[str, Any]] = {}

    for profile_name, profile in profiles.items():
        rows: list[dict[str, Any]] = []
        for cand in nonbaseline:
            weighted_sum = 0.0
            weight_total = 0.0
            breakdown: dict[str, dict[str, float | None | str]] = {}
            for key, direction, weight in profile["weights"]:
                raw = metric_value(cand, key)
                norm = norm_metric(raw, metric_ranges.get(key, []), higher_is_better=(direction == "max"))
                breakdown[key] = {
                    "direction": direction,
                    "weight": weight,
                    "raw": round_or_none(raw),
                    "normalized": round_or_none(norm, 4) if norm is not None else None,
                    "weighted": round_or_none((norm * weight), 4) if norm is not None else None,
                }
                if norm is None:
                    continue
                weighted_sum += norm * weight
                weight_total += weight

            score_pct = None if weight_total == 0 else round(weighted_sum / weight_total * 100.0, 2)
            rows.append(
                {
                    "scenario": cand["scenario"],
                    "battery_size_kwh": cand["battery_size_kwh"],
                    "score": score_pct,
                    "score_breakdown": breakdown,
                    "key_metrics": {
                        "annualized_gain_chf": metric_value(cand, "annualized_gain_chf"),
                        "bill_reduction_pct_vs_no_battery": metric_value(cand, "bill_reduction_pct_vs_no_battery"),
                        "grid_consumed_reduction_pct_vs_no_battery": metric_value(
                            cand, "grid_consumed_reduction_pct_vs_no_battery"
                        ),
                        "amortization_years": metric_value(cand, "amortization_years"),
                        "max_season_evening_undersize_pct": metric_value(cand, "max_season_evening_undersize_pct"),
                    },
                }
            )

        rows.sort(
            key=lambda r: (
                -999999 if r["score"] is None else -r["score"],
                r["battery_size_kwh"],
                r["scenario"],
            )
        )
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
        rankings[profile_name] = rows

        if rows:
            winner = rows[0]
            reason = profile["description"]
            if profile_name == "balanced" and knee_point:
                if abs(float(winner["battery_size_kwh"]) - float(knee_point["battery_size_kwh"])) < 1e-9:
                    reason += " Winner also matches the knee-point (diminishing returns threshold)."
            best_by_profile[profile_name] = {
                "scenario": winner["scenario"],
                "battery_size_kwh": winner["battery_size_kwh"],
                "score": winner["score"],
                "reason": reason,
            }

    # Smallest-good-enough profile (constraint-style selector).
    smallest_good_enough = compute_smallest_good_enough(nonbaseline, knee_point)
    if smallest_good_enough is not None:
        best_by_profile["smallest_good_enough"] = smallest_good_enough["winner"]
        rankings["smallest_good_enough"] = smallest_good_enough["ranking"]
        profiles["smallest_good_enough"] = smallest_good_enough["profile"]

    return {
        "profiles": profiles,
        "rankings": rankings,
        "best_by_profile": best_by_profile,
    }


def compute_smallest_good_enough(
    nonbaseline: list[dict[str, Any]],
    knee_point: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not nonbaseline:
        return None

    gains = [metric_value(c, "annualized_gain_chf") or 0.0 for c in nonbaseline]
    bills = [metric_value(c, "bill_reduction_pct_vs_no_battery") or 0.0 for c in nonbaseline]
    max_gain = max(gains) if gains else 0.0
    max_bill_pct = max(bills) if bills else 0.0
    gain_target = 0.90 * max_gain if max_gain > 0 else 0.0
    bill_target = 0.90 * max_bill_pct if max_bill_pct > 0 else 0.0

    rows: list[dict[str, Any]] = []
    for cand in sorted(nonbaseline, key=lambda c: (c["battery_size_kwh"], c["scenario"])):
        gain = metric_value(cand, "annualized_gain_chf") or 0.0
        bill = metric_value(cand, "bill_reduction_pct_vs_no_battery") or 0.0
        evening = metric_value(cand, "max_season_evening_undersize_pct")
        meets = gain >= gain_target and bill >= bill_target
        rows.append(
            {
                "scenario": cand["scenario"],
                "battery_size_kwh": cand["battery_size_kwh"],
                "score": None,
                "rank": None,
                "meets_thresholds": meets,
                "threshold_checks": {
                    "annualized_gain_chf_ge_90pct_max": meets and gain >= gain_target or gain >= gain_target,
                    "bill_reduction_pct_ge_90pct_max": meets and bill >= bill_target or bill >= bill_target,
                    "max_season_evening_undersize_pct": evening,
                },
                "key_metrics": {
                    "annualized_gain_chf": gain,
                    "bill_reduction_pct_vs_no_battery": bill,
                    "max_season_evening_undersize_pct": evening,
                },
            }
        )

    eligible = [r for r in rows if r["meets_thresholds"]]
    if eligible:
        winner_row = min(eligible, key=lambda r: (r["battery_size_kwh"], r["scenario"]))
        reason = (
            "Smallest battery size reaching at least 90% of max annualized gain and "
            "90% of max bill reduction among analyzed configurations."
        )
    elif knee_point:
        winner_row = min(
            rows,
            key=lambda r: (
                abs(r["battery_size_kwh"] - float(knee_point["battery_size_kwh"])),
                r["battery_size_kwh"],
            ),
        )
        reason = "No candidate met the good-enough thresholds; fell back to knee-point candidate."
    else:
        winner_row = rows[0]
        reason = "No candidate met the good-enough thresholds; fell back to smallest analyzed battery."

    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        row["score"] = 100.0 if row["scenario"] == winner_row["scenario"] else None

    return {
        "profile": {
            "description": "Constraint-based selector for smallest good-enough battery size.",
            "thresholds": {
                "annualized_gain_chf_fraction_of_max": 0.90,
                "bill_reduction_pct_fraction_of_max": 0.90,
                "computed_annualized_gain_target_chf": round_or_none(gain_target),
                "computed_bill_reduction_target_pct": round_or_none(bill_target),
            },
        },
        "ranking": rows,
        "winner": {
            "scenario": winner_row["scenario"],
            "battery_size_kwh": winner_row["battery_size_kwh"],
            "score": None,
            "reason": reason,
        },
    }


def compute_knee_point(
    candidates: list[dict[str, Any]],
    marginal_gain_threshold: float,
) -> dict[str, Any] | None:
    nonbaseline = sorted([c for c in candidates if not c["is_baseline"]], key=lambda c: c["battery_size_kwh"])
    if not nonbaseline:
        return None
    if len(nonbaseline) == 1:
        only = nonbaseline[0]
        return {
            "battery_size_kwh": only["battery_size_kwh"],
            "scenario": only["scenario"],
            "reason": "Only one non-baseline battery configuration available.",
            "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
            "trigger_pair": None,
        }

    previous = None
    for cand in sorted(candidates, key=lambda c: c["battery_size_kwh"]):
        if previous is None:
            previous = cand
            continue
        if cand["is_baseline"]:
            previous = cand
            continue
        marginal = safe_float(cand.get("derived", {}).get("marginal_annualized_gain_chf_per_added_kwh"))
        if marginal is not None and marginal < marginal_gain_threshold and previous is not None:
            return {
                "battery_size_kwh": previous["battery_size_kwh"],
                "scenario": previous["scenario"],
                "reason": "First size before marginal annualized gain per added kWh drops below threshold.",
                "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
                "trigger_pair": {
                    "from_scenario": previous["scenario"],
                    "to_scenario": cand["scenario"],
                    "to_battery_size_kwh": cand["battery_size_kwh"],
                    "marginal_annualized_gain_chf_per_added_kwh": marginal,
                },
            }
        previous = cand

    largest = nonbaseline[-1]
    return {
        "battery_size_kwh": largest["battery_size_kwh"],
        "scenario": largest["scenario"],
        "reason": "Marginal annualized gain per added kWh never dropped below threshold.",
        "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
        "trigger_pair": None,
    }


def best_candidate_by_metric(
    candidates: list[dict[str, Any]],
    key: str,
    direction: str,
) -> dict[str, Any] | None:
    pool = [c for c in candidates if not c["is_baseline"]]
    values: list[tuple[dict[str, Any], float]] = []
    for cand in pool:
        v = metric_value(cand, key)
        if v is None:
            continue
        values.append((cand, v))
    if not values:
        return None
    if direction == "max":
        cand, value = max(values, key=lambda item: (item[1], -item[0]["battery_size_kwh"]))
    else:
        cand, value = min(values, key=lambda item: (item[1], item[0]["battery_size_kwh"]))
    return {
        "scenario": cand["scenario"],
        "battery_size_kwh": cand["battery_size_kwh"],
        "metric": key,
        "direction": direction,
        "value": round_or_none(value),
    }


def compute_best_by_metric(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    specs = {
        "annualized_gain_chf": "max",
        "bill_reduction_pct_vs_no_battery": "max",
        "grid_consumed_reduction_pct_vs_no_battery": "max",
        "amortization_years": "min",
        "max_season_evening_undersize_pct": "min",
        "max_season_energy_undersize_pct": "min",
        "avg_season_power_at_max_pct": "min",
    }
    out: dict[str, Any] = {}
    for key, direction in specs.items():
        best = best_candidate_by_metric(candidates, key, direction)
        if best is not None:
            out[key] = best
    return out


def compute_best_by_season_gain(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pool = [c for c in candidates if not c["is_baseline"]]
    for season in SEASON_ORDER:
        best = None
        for cand in pool:
            season_blob = cand.get("seasonal_by_season", {}).get(season, {})
            v = safe_float(season_blob.get("total_gain_chf"))
            if v is None:
                continue
            if best is None or v > best[1] or (v == best[1] and cand["battery_size_kwh"] < best[0]["battery_size_kwh"]):
                best = (cand, v)
        if best:
            out[season] = {
                "scenario": best[0]["scenario"],
                "battery_size_kwh": best[0]["battery_size_kwh"],
                "season_total_gain_chf": round_or_none(best[1]),
            }
    return out


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Battery Sizing KPI Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at_utc']}")
    lines.append(f"- Simulation files analyzed: {len(summary['source_files'])}")
    lines.append(
        f"- Non-baseline candidates ranked: {len([c for c in summary['candidates'] if not c.get('is_baseline')])}"
    )
    lines.append("")

    baseline = summary.get("baseline", {})
    if baseline:
        lines.append("## Baseline (No Battery)")
        lines.append("")
        lines.append(f"- Scenario: `{baseline.get('scenario')}`")
        lines.append(f"- Total grid bill (CHF): `{baseline.get('bill_without_battery_chf')}`")
        lines.append(f"- Total grid consumed (kWh): `{baseline.get('grid_consumed_without_battery_kwh')}`")
        lines.append("")

    knee = summary.get("knee_point")
    if knee:
        lines.append("## Knee Point (Diminishing Returns)")
        lines.append("")
        lines.append(f"- Selected size: `{knee.get('battery_size_kwh')} kWh` (`{knee.get('scenario')}`)")
        lines.append(f"- Threshold: `{knee.get('marginal_gain_threshold_chf_per_added_kwh')} CHF/year per added kWh`")
        lines.append(f"- Reason: {knee.get('reason')}")
        trigger = knee.get("trigger_pair")
        if trigger:
            lines.append(
                "- Trigger transition: "
                f"`{trigger.get('from_scenario')}` -> `{trigger.get('to_scenario')}` "
                f"(marginal `{trigger.get('marginal_annualized_gain_chf_per_added_kwh')} CHF/year/kWh`)"
            )
        lines.append("")

    lines.append("## Best Candidates by Decision Profile")
    lines.append("")
    lines.append("| Profile | Battery Size (kWh) | Scenario | Score | Reason |")
    lines.append("|---|---:|---|---:|---|")
    for profile, winner in summary.get("decision_profiles", {}).get("best_by_profile", {}).items():
        score = winner.get("score")
        score_text = "" if score is None else f"{score:.2f}"
        lines.append(
            f"| {profile} | {winner.get('battery_size_kwh')} | {winner.get('scenario')} | {score_text} | {winner.get('reason','')} |"
        )
    lines.append("")

    season_winners = summary.get("best_by_season_total_gain", {})
    if season_winners:
        lines.append("## Best Battery Size by Season (Net Financial Gain)")
        lines.append("")
        for season in SEASON_ORDER:
            item = season_winners.get(season)
            if not item:
                continue
            lines.append(
                f"- `{season}`: `{item['battery_size_kwh']} kWh` (`{item['scenario']}`) "
                f"with `{item['season_total_gain_chf']} CHF`"
            )
        lines.append("")

    lines.append("## Candidate Comparison (Global + Seasonal Robustness)")
    lines.append("")
    lines.append(
        "| Battery Size (kWh) | Scenario | Annualized Gain (CHF) | Bill Reduction (%) | Grid Import Reduction (%) | "
        "Amortization (y) | Max Evening Undersize (%) | Marginal Gain (CHF/y per added kWh) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for cand in sorted(
        [c for c in summary["candidates"] if not c.get("is_baseline")], key=lambda c: c["battery_size_kwh"]
    ):
        m = cand["metrics"]
        d = cand.get("derived", {})
        lines.append(
            "| "
            f"{cand['battery_size_kwh']} | {cand['scenario']} | "
            f"{m.get('annualized_gain_chf','')} | {m.get('bill_reduction_pct_vs_no_battery','')} | "
            f"{m.get('grid_consumed_reduction_pct_vs_no_battery','')} | {m.get('amortization_years','')} | "
            f"{m.get('max_season_evening_undersize_pct','')} | "
            f"{d.get('marginal_annualized_gain_chf_per_added_kwh','')} |"
        )
    lines.append("")

    lines.append("## Notes for LLM Recommendation")
    lines.append("")
    lines.append("- Prefer the `balanced` winner as the default recommendation candidate unless explicit user priorities differ.")
    lines.append("- Use `roi_first`, `autonomy_first`, and `winter_robustness` winners as alternatives with clear tradeoffs.")
    lines.append("- Use the knee point as the practical sizing reference to justify avoiding oversizing.")
    lines.append("- When bill reduction exceeds 100%, interpret it as bill elimination plus net credit.")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute battery-sizing KPI rankings from simulation output JSON files."
    )
    parser.add_argument(
        "--simulation-jsons",
        nargs="*",
        default=[],
        help="Simulation JSON files to analyze (default: out/simulation_json/config_*.json).",
    )
    parser.add_argument(
        "--output-json",
        default="out/kpi_summary/kpi_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-markdown",
        default="out/kpi_summary/kpi_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--marginal-gain-threshold",
        type=float,
        default=20.0,
        help="Threshold (CHF/year per added kWh) for knee-point detection (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_paths = [Path(p) for p in args.simulation_jsons] if args.simulation_jsons else sorted(
        Path("out/simulation_json").glob("config_*.json")
    )
    sim_paths = [p for p in sim_paths if p.exists() and p.is_file()]
    if not sim_paths:
        raise FileNotFoundError("No simulation JSON files found to analyze.")

    candidates = [build_candidate(p) for p in sorted(sim_paths)]
    candidates.sort(key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    compute_marginals(candidates)
    knee_point = compute_knee_point(candidates, args.marginal_gain_threshold)
    decision_profiles = rank_profiles(candidates, knee_point)

    baseline = next((c for c in candidates if c.get("is_baseline")), None)
    best_by_metric = compute_best_by_metric(candidates)
    best_by_season_total_gain = compute_best_by_season_gain(candidates)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "marginal_gain_threshold_chf_per_added_kwh": args.marginal_gain_threshold,
        "source_files": [str(p) for p in sorted(sim_paths)],
        "baseline": None
        if baseline is None
        else {
            "scenario": baseline["scenario"],
            "battery_size_kwh": baseline["battery_size_kwh"],
            "bill_without_battery_chf": baseline["global"].get("bill_without_battery_chf"),
            "grid_consumed_without_battery_kwh": baseline["global"].get("grid_consumed_without_battery_kwh"),
        },
        "knee_point": knee_point,
        "best_by_metric": best_by_metric,
        "best_by_season_total_gain": best_by_season_total_gain,
        "decision_profiles": decision_profiles,
        "candidates": candidates,
    }
    summary = sanitize_for_json(summary)

    out_json = Path(args.output_json)
    out_md = Path(args.output_markdown)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"KPI JSON summary written: {out_json}")
    print(f"KPI Markdown summary written: {out_md}")

    for profile in ("balanced", "roi_first", "autonomy_first", "winter_robustness", "smallest_good_enough"):
        winner = summary.get("decision_profiles", {}).get("best_by_profile", {}).get(profile)
        if winner:
            print(
                f"[{profile}] {winner['battery_size_kwh']} kWh ({winner['scenario']})"
                + (f" score={winner['score']}" if winner.get("score") is not None else "")
            )


if __name__ == "__main__":
    main()
