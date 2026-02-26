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


SEASON_ORDER = ["spring", "summer", "autumn", "winter"]


def clamp_fraction(value: Any, *, default: float) -> float:
    v = safe_float(value)
    if v is None:
        return default
    return max(0.0, min(1.0, v))


def positive_or_default(value: Any, *, default: float) -> float:
    v = safe_float(value)
    if v is None or v <= 0:
        return default
    return v


def nonnegative_int_or_default(value: Any, *, default: int) -> int:
    v = safe_float(value)
    if v is None:
        return default
    try:
        n = int(round(v))
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def load_kpi_config(config_path: Path | None) -> tuple[dict[str, Any], str]:
    """
    Load KPI tuning config from JSON.
    The config file is mandatory.
    """
    if config_path is None:
        raise ValueError("KPI config path is required (use --config).")

    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"KPI config file not found: {config_path}")

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"KPI config must be a JSON object: {config_path}")
    return loaded, str(config_path)


def build_decision_profiles_from_config(kpi_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles_blob = kpi_config.get("decision_profiles", {}) or {}
    profiles: dict[str, dict[str, Any]] = {}

    if not isinstance(profiles_blob, dict):
        profiles_blob = {}

    for profile_name, raw_profile in profiles_blob.items():
        if not isinstance(raw_profile, dict):
            continue
        description = str(raw_profile.get("description", "")).strip() or f"{profile_name} decision profile."
        raw_weights = raw_profile.get("weights", [])
        weights: list[tuple[str, str, float]] = []
        if not isinstance(raw_weights, list):
            raw_weights = []
        for item in raw_weights:
            key = None
            direction = None
            weight = None
            if isinstance(item, dict):
                key = item.get("metric") or item.get("key")
                direction = item.get("direction")
                weight = item.get("weight")
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                key, direction, weight = item

            key = str(key).strip() if key is not None else ""
            direction = str(direction).strip().lower() if direction is not None else ""
            weight_f = safe_float(weight)
            if not key or direction not in ("max", "min") or weight_f is None or weight_f <= 0:
                continue
            weights.append((key, direction, weight_f))

        if weights:
            profiles[str(profile_name)] = {
                "description": description,
                "weights": weights,
            }

    if profiles:
        return profiles

    return {}


def resolve_kpi_settings(
    kpi_config: dict[str, Any],
    *,
    cli_global_knee_threshold: float | None,
    cli_seasonal_knee_threshold: float | None,
) -> dict[str, Any]:
    thresholds = kpi_config.get("thresholds", {}) or {}
    if not isinstance(thresholds, dict):
        thresholds = {}

    sge = thresholds.get("smallest_good_enough", {}) or {}
    if not isinstance(sge, dict):
        sge = {}
    seasonal_sge = thresholds.get("seasonal_smallest_good_enough", {}) or {}
    if not isinstance(seasonal_sge, dict):
        seasonal_sge = {}
    graph_kpis = thresholds.get("graph_kpis", {}) or {}
    if not isinstance(graph_kpis, dict):
        graph_kpis = {}
    global_energy_reduction_kwh = graph_kpis.get("global_energy_reduction_kwh", {}) or {}
    if not isinstance(global_energy_reduction_kwh, dict):
        global_energy_reduction_kwh = {}
    global_energy_financial_impact_chf = graph_kpis.get("global_energy_financial_impact_chf", {}) or {}
    if not isinstance(global_energy_financial_impact_chf, dict):
        global_energy_financial_impact_chf = {}
    global_rentability_overview = graph_kpis.get("global_rentability_overview", {}) or {}
    if not isinstance(global_rentability_overview, dict):
        global_rentability_overview = {}
    global_battery_utilization = graph_kpis.get("global_battery_utilization", {}) or {}
    if not isinstance(global_battery_utilization, dict):
        global_battery_utilization = {}
    global_battery_status_heatmap = graph_kpis.get("global_battery_status_heatmap", {}) or {}
    if not isinstance(global_battery_status_heatmap, dict):
        global_battery_status_heatmap = {}
    seasonal_power_saturation_at_max_limit = graph_kpis.get("seasonal_power_saturation_at_max_limit", {}) or {}
    if not isinstance(seasonal_power_saturation_at_max_limit, dict):
        seasonal_power_saturation_at_max_limit = {}
    monthly_structural_evening_energy_undersizing_peak_period = (
        graph_kpis.get("monthly_structural_evening_energy_undersizing_peak_period", {}) or {}
    )
    if not isinstance(monthly_structural_evening_energy_undersizing_peak_period, dict):
        monthly_structural_evening_energy_undersizing_peak_period = {}

    global_knee = (
        positive_or_default(cli_global_knee_threshold, default=20.0)
        if cli_global_knee_threshold is not None
        else positive_or_default(thresholds.get("global_knee_marginal_gain_chf_per_added_kwh"), default=20.0)
    )
    seasonal_knee = (
        positive_or_default(cli_seasonal_knee_threshold, default=global_knee)
        if cli_seasonal_knee_threshold is not None
        else positive_or_default(thresholds.get("seasonal_knee_marginal_gain_chf_per_added_kwh"), default=global_knee)
    )

    return {
        "global_knee_marginal_gain_chf_per_added_kwh": round_or_none(global_knee),
        "seasonal_knee_marginal_gain_chf_per_added_kwh": round_or_none(seasonal_knee),
        "smallest_good_enough": {
            "annualized_gain_fraction_of_max": round_or_none(
                clamp_fraction(sge.get("annualized_gain_fraction_of_max"), default=0.90), 4
            ),
            "bill_reduction_fraction_of_max": round_or_none(
                clamp_fraction(sge.get("bill_reduction_fraction_of_max"), default=0.90), 4
            ),
        },
        "seasonal_smallest_good_enough": {
            "gain_fraction_of_max": round_or_none(
                clamp_fraction(seasonal_sge.get("gain_fraction_of_max"), default=0.90), 4
            ),
        },
        "graph_kpis": {
            "global_energy_reduction_kwh": {
                "consumed_reduction_increment_pct_points_min": round_or_none(
                    positive_or_default(
                        global_energy_reduction_kwh.get("consumed_reduction_increment_pct_points_min"),
                        default=10.0,
                    ),
                    4,
                )
            },
            "global_energy_financial_impact_chf": {
                "bill_offset_increment_pct_points_min": round_or_none(
                    positive_or_default(
                        global_energy_financial_impact_chf.get("bill_offset_increment_pct_points_min"),
                        default=10.0,
                    ),
                    4,
                )
            },
            "global_rentability_overview": {
                "amortization_years_max": round_or_none(
                    positive_or_default(
                        global_rentability_overview.get("amortization_years_max"),
                        default=8.0,
                    ),
                    4,
                )
            },
            "global_battery_utilization": {
                "pct_max_cycles_per_year_max": round_or_none(
                    positive_or_default(
                        global_battery_utilization.get("pct_max_cycles_per_year_max"),
                        default=4.0,
                    ),
                    4,
                )
            },
            "global_battery_status_heatmap": {
                "empty_pct_max": round_or_none(
                    positive_or_default(
                        global_battery_status_heatmap.get("empty_pct_max"),
                        default=20.0,
                    ),
                    4,
                )
            },
            "seasonal_power_saturation_at_max_limit": {
                "power_saturation_pct_any_season_max": round_or_none(
                    positive_or_default(
                        seasonal_power_saturation_at_max_limit.get("power_saturation_pct_any_season_max"),
                        default=10.0,
                    ),
                    4,
                )
            },
            "monthly_structural_evening_energy_undersizing_peak_period": {
                "evening_undersize_pct_per_month_max": round_or_none(
                    positive_or_default(
                        monthly_structural_evening_energy_undersizing_peak_period.get(
                            "evening_undersize_pct_per_month_max"
                        ),
                        default=40.0,
                    ),
                    4,
                ),
                "max_months_above_threshold": nonnegative_int_or_default(
                    monthly_structural_evening_energy_undersizing_peak_period.get("max_months_above_threshold"),
                    default=1,
                ),
            },
        },
    }


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
    battery_global = nested_get(data, "global", "battery", default={}) or {}
    battery_utilization = nested_get(battery_global, "utilization", default={}) or {}
    battery_status = nested_get(battery_global, "status", default={}) or {}
    seasons_blob = data.get("seasons", {}) or {}
    months_blob = data.get("months", []) or []

    season_metrics: dict[str, dict[str, float]] = {}
    for season_name, values in seasons_blob.items():
        season_metrics[season_name.lower()] = season_metrics_from_blob(values or {})

    season_agg = aggregate_season_metrics(season_metrics)

    def avg_global_status_pct(key: str) -> float:
        values: list[float] = []
        for phase in ("A", "B", "C"):
            values.append(float(nested_float(battery_status, phase, key, "samples_percent", default=0.0)))
        return float(mean(values)) if values else 0.0

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
        "cycles_per_year": round_or_none(safe_float(battery_utilization.get("cycles_per_year"))),
        "pct_max_cycles_per_year": round_or_none(safe_float(battery_utilization.get("percent_of_max_cycles_per_year"))),
        "status_charging_pct": round_or_none(avg_global_status_pct("charging")),
        "status_discharging_pct": round_or_none(avg_global_status_pct("discharging")),
        "status_full_pct": round_or_none(avg_global_status_pct("full")),
        "status_empty_pct": round_or_none(avg_global_status_pct("empty")),
        "status_idle_pct": round_or_none(avg_global_status_pct("idle")),
    }

    # Flatten core metrics for ranking/scoring convenience.
    metrics: dict[str, float | None] = {
        "battery_size_kwh": size_kwh,
        **global_metrics,
    }
    for key, value in season_agg.items():
        metrics[key] = value if isinstance(value, (int, float)) or value is None else safe_float(value)

    monthly_evening_undersize_pct_by_month: dict[str, float | None] = {}
    for month_item in months_blob:
        if not isinstance(month_item, dict):
            continue
        month_range = month_item.get("range", {}) or {}
        is_full_month = bool(month_range.get("is_full_month", False))
        n_days = safe_float(month_range.get("calendar_duration_days")) or 0.0
        # Mirror the notebook filtering logic used for monthly charts.
        if not is_full_month and n_days < 28:
            continue
        month_id = month_item.get("range_id")
        if not month_id:
            continue
        battery_block = nested_get(month_item, "results", "battery", default={}) or {}
        evening_undersize_days = battery_block.get("evening_undersize_days", {}) or {}
        monthly_evening_undersize_pct_by_month[str(month_id)] = round_or_none(
            safe_float(evening_undersize_days.get("percent")) or 0.0
        )

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
        "monthly_evening_undersize_pct_by_month": dict(sorted(monthly_evening_undersize_pct_by_month.items())),
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
    *,
    profile_definitions: dict[str, dict[str, Any]],
    smallest_good_enough_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nonbaseline = [c for c in candidates if not c["is_baseline"]]
    profiles: dict[str, dict[str, Any]] = {
        name: {
            "description": str(profile.get("description", "")),
            "weights": list(profile.get("weights", [])),
        }
        for name, profile in profile_definitions.items()
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
    sge_cfg = smallest_good_enough_config or {}
    smallest_good_enough = compute_smallest_good_enough(
        nonbaseline,
        knee_point,
        annualized_gain_fraction_of_max=safe_float(sge_cfg.get("annualized_gain_fraction_of_max")) or 0.90,
        bill_reduction_fraction_of_max=safe_float(sge_cfg.get("bill_reduction_fraction_of_max")) or 0.90,
    )
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
    *,
    annualized_gain_fraction_of_max: float = 0.90,
    bill_reduction_fraction_of_max: float = 0.90,
) -> dict[str, Any] | None:
    if not nonbaseline:
        return None

    gains = [metric_value(c, "annualized_gain_chf") or 0.0 for c in nonbaseline]
    bills = [metric_value(c, "bill_reduction_pct_vs_no_battery") or 0.0 for c in nonbaseline]
    max_gain = max(gains) if gains else 0.0
    max_bill_pct = max(bills) if bills else 0.0
    annualized_gain_fraction_of_max = clamp_fraction(annualized_gain_fraction_of_max, default=0.90)
    bill_reduction_fraction_of_max = clamp_fraction(bill_reduction_fraction_of_max, default=0.90)

    gain_target = annualized_gain_fraction_of_max * max_gain if max_gain > 0 else 0.0
    bill_target = bill_reduction_fraction_of_max * max_bill_pct if max_bill_pct > 0 else 0.0

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
            "Smallest battery size reaching at least "
            f"{int(round(annualized_gain_fraction_of_max * 100))}% of max annualized gain and "
            f"{int(round(bill_reduction_fraction_of_max * 100))}% of max bill reduction among analyzed configurations."
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
                "annualized_gain_chf_fraction_of_max": round_or_none(annualized_gain_fraction_of_max, 4),
                "bill_reduction_pct_fraction_of_max": round_or_none(bill_reduction_fraction_of_max, 4),
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


def compute_global_energy_reduction_consumed_step_delta_kpi(
    candidates: list[dict[str, Any]],
    *,
    threshold_pct_points: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for Global Energy Reduction (kWh chart right axis):
    select the largest battery size whose incremental consumed-reduction gain (% points)
    vs the previous size remains above the configured threshold.
    """
    threshold_pct_points = positive_or_default(threshold_pct_points, default=10.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    if not ordered:
        return None

    baseline = next((c for c in ordered if c.get("is_baseline")), None)
    steps: list[dict[str, Any]] = []
    selected_step: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None

    for cand in ordered:
        if previous is None:
            previous = cand
            continue
        if cand.get("is_baseline"):
            previous = cand
            continue

        prev_pct = metric_value(previous, "grid_consumed_reduction_pct_vs_no_battery")
        cur_pct = metric_value(cand, "grid_consumed_reduction_pct_vs_no_battery")
        if previous.get("is_baseline") and prev_pct is None:
            prev_pct = 0.0
        delta_pct_points = None if prev_pct is None or cur_pct is None else (cur_pct - prev_pct)
        meets = bool(delta_pct_points is not None and delta_pct_points >= threshold_pct_points)

        step = {
            "from_scenario": previous.get("scenario"),
            "from_battery_size_kwh": round_or_none(safe_float(previous.get("battery_size_kwh"))),
            "from_grid_consumed_reduction_pct_vs_no_battery": round_or_none(prev_pct),
            "to_scenario": cand.get("scenario"),
            "to_battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "to_grid_consumed_reduction_pct_vs_no_battery": round_or_none(cur_pct),
            "delta_pct_points_vs_previous_size": round_or_none(delta_pct_points),
            "meets_increment_threshold": meets,
        }
        steps.append(step)
        if meets:
            selected_step = step

        previous = cand

    nonbaseline = [c for c in ordered if not c.get("is_baseline")]
    recommendation: dict[str, Any] | None = None
    if selected_step is not None:
        recommendation = {
            "scenario": selected_step["to_scenario"],
            "battery_size_kwh": selected_step["to_battery_size_kwh"],
            "grid_consumed_reduction_pct_vs_no_battery": selected_step["to_grid_consumed_reduction_pct_vs_no_battery"],
            "selection_rule": (
                "Largest battery size whose incremental gain in consumed reduction (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": (
                f"Selected because the step gain vs previous size is "
                f"{selected_step['delta_pct_points_vs_previous_size']} percentage points, "
                f"meeting the threshold of {round_or_none(threshold_pct_points)}."
            ),
        }
    elif baseline is not None:
        baseline_pct = metric_value(baseline, "grid_consumed_reduction_pct_vs_no_battery")
        recommendation = {
            "scenario": baseline.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(baseline.get("battery_size_kwh"))),
            "grid_consumed_reduction_pct_vs_no_battery": round_or_none(0.0 if baseline_pct is None else baseline_pct),
            "selection_rule": (
                "Largest battery size whose incremental gain in consumed reduction (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": (
                "No analyzed battery step met the incremental gain threshold; "
                "fallback to baseline (0 kWh / no battery)."
            ),
        }
    elif nonbaseline:
        first = min(nonbaseline, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
        recommendation = {
            "scenario": first.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(first.get("battery_size_kwh"))),
            "grid_consumed_reduction_pct_vs_no_battery": round_or_none(
                metric_value(first, "grid_consumed_reduction_pct_vs_no_battery")
            ),
            "selection_rule": (
                "Largest battery size whose incremental gain in consumed reduction (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": "No baseline candidate found; fallback to smallest analyzed battery.",
        }

    if not steps and recommendation is None:
        return None

    return {
        "name": "Global Energy Reduction KWh – Consumed reduction step-delta KPI",
        "metric": "grid_consumed_reduction_pct_vs_no_battery",
        "threshold_increment_pct_points": round_or_none(threshold_pct_points),
        "selection_rule": (
            "Select the largest battery size whose incremental consumed-reduction gain "
            "vs the previous analyzed size is >= threshold (percentage points)."
        ),
        "recommendation": recommendation,
        "steps": steps,
    }


def compute_global_energy_financial_impact_bill_offset_step_delta_kpi(
    candidates: list[dict[str, Any]],
    *,
    threshold_pct_points: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for Global Energy Financial Impact (CHF chart right axis):
    select the largest battery size whose incremental bill-offset gain (% points)
    vs the previous size remains above the configured threshold.
    """
    threshold_pct_points = positive_or_default(threshold_pct_points, default=10.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    if not ordered:
        return None

    baseline = next((c for c in ordered if c.get("is_baseline")), None)
    steps: list[dict[str, Any]] = []
    selected_step: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None

    for cand in ordered:
        if previous is None:
            previous = cand
            continue
        if cand.get("is_baseline"):
            previous = cand
            continue

        # This metric is used as the right-axis % on the financial impact chart.
        prev_pct = metric_value(previous, "bill_reduction_pct_vs_no_battery")
        cur_pct = metric_value(cand, "bill_reduction_pct_vs_no_battery")
        if previous.get("is_baseline") and prev_pct is None:
            prev_pct = 0.0
        delta_pct_points = None if prev_pct is None or cur_pct is None else (cur_pct - prev_pct)
        meets = bool(delta_pct_points is not None and delta_pct_points >= threshold_pct_points)

        step = {
            "from_scenario": previous.get("scenario"),
            "from_battery_size_kwh": round_or_none(safe_float(previous.get("battery_size_kwh"))),
            "from_bill_offset_pct_vs_no_battery": round_or_none(prev_pct),
            "to_scenario": cand.get("scenario"),
            "to_battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "to_bill_offset_pct_vs_no_battery": round_or_none(cur_pct),
            "delta_pct_points_vs_previous_size": round_or_none(delta_pct_points),
            "meets_increment_threshold": meets,
        }
        steps.append(step)
        if meets:
            selected_step = step

        previous = cand

    nonbaseline = [c for c in ordered if not c.get("is_baseline")]
    recommendation: dict[str, Any] | None = None
    if selected_step is not None:
        recommendation = {
            "scenario": selected_step["to_scenario"],
            "battery_size_kwh": selected_step["to_battery_size_kwh"],
            "bill_offset_pct_vs_no_battery": selected_step["to_bill_offset_pct_vs_no_battery"],
            "selection_rule": (
                "Largest battery size whose incremental gain in bill offset (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": (
                f"Selected because the step gain vs previous size is "
                f"{selected_step['delta_pct_points_vs_previous_size']} percentage points, "
                f"meeting the threshold of {round_or_none(threshold_pct_points)}."
            ),
        }
    elif baseline is not None:
        baseline_pct = metric_value(baseline, "bill_reduction_pct_vs_no_battery")
        recommendation = {
            "scenario": baseline.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(baseline.get("battery_size_kwh"))),
            "bill_offset_pct_vs_no_battery": round_or_none(0.0 if baseline_pct is None else baseline_pct),
            "selection_rule": (
                "Largest battery size whose incremental gain in bill offset (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": (
                "No analyzed battery step met the incremental gain threshold; "
                "fallback to baseline (0 kWh / no battery)."
            ),
        }
    elif nonbaseline:
        first = min(nonbaseline, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
        recommendation = {
            "scenario": first.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(first.get("battery_size_kwh"))),
            "bill_offset_pct_vs_no_battery": round_or_none(metric_value(first, "bill_reduction_pct_vs_no_battery")),
            "selection_rule": (
                "Largest battery size whose incremental gain in bill offset (% points) "
                "vs the previous analyzed size remains at or above threshold."
            ),
            "reason": "No baseline candidate found; fallback to smallest analyzed battery.",
        }

    if not steps and recommendation is None:
        return None

    return {
        "name": "Global Energy Financial Impact CHF – Bill offset step-delta KPI",
        "metric": "bill_reduction_pct_vs_no_battery",
        "threshold_increment_pct_points": round_or_none(threshold_pct_points),
        "selection_rule": (
            "Select the largest battery size whose incremental bill-offset gain "
            "vs the previous analyzed size is >= threshold (percentage points)."
        ),
        "recommendation": recommendation,
        "steps": steps,
    }


def compute_global_rentability_amortization_cap_kpi(
    candidates: list[dict[str, Any]],
    *,
    amortization_years_max: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for Global Rentability Overview:
    treat battery sizes with amortization above the configured threshold as too expensive
    and recommend the largest battery size that remains at or below the threshold.
    """
    amortization_years_max = positive_or_default(amortization_years_max, default=8.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cand in ordered:
        if cand.get("is_baseline"):
            continue
        amort = metric_value(cand, "amortization_years")
        if amort is None:
            continue
        within = amort <= amortization_years_max
        row = {
            "scenario": cand.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "amortization_years": round_or_none(amort),
            "annualized_gain_chf": round_or_none(metric_value(cand, "annualized_gain_chf")),
            "total_gain_chf": round_or_none(metric_value(cand, "total_gain_chf")),
            "bill_reduction_pct_vs_no_battery": round_or_none(metric_value(cand, "bill_reduction_pct_vs_no_battery")),
            "within_amortization_cap": within,
            "status": "acceptable" if within else "too_expensive",
        }
        evaluated.append(row)
        if within:
            selected = row

    if not evaluated:
        return None

    recommendation: dict[str, Any]
    first_too_expensive = next((row for row in evaluated if not row.get("within_amortization_cap")), None)
    if selected is not None:
        recommendation = {
            "scenario": selected["scenario"],
            "battery_size_kwh": selected["battery_size_kwh"],
            "amortization_years": selected["amortization_years"],
            "annualized_gain_chf": selected["annualized_gain_chf"],
            "total_gain_chf": selected["total_gain_chf"],
            "bill_reduction_pct_vs_no_battery": selected["bill_reduction_pct_vs_no_battery"],
            "selection_rule": (
                "Select the largest battery size whose amortization stays at or below the configured maximum."
            ),
            "reason": (
                f"Selected as the largest analyzed size with amortization "
                f"{selected['amortization_years']} years <= {round_or_none(amortization_years_max)} years."
            ),
        }
        if first_too_expensive and first_too_expensive["battery_size_kwh"] > (selected["battery_size_kwh"] or 0.0):
            recommendation["next_too_expensive_candidate"] = {
                "scenario": first_too_expensive["scenario"],
                "battery_size_kwh": first_too_expensive["battery_size_kwh"],
                "amortization_years": first_too_expensive["amortization_years"],
            }
    else:
        best_amort = min(evaluated, key=lambda row: (row.get("amortization_years") or float("inf"), row["battery_size_kwh"]))
        recommendation = {
            "scenario": best_amort["scenario"],
            "battery_size_kwh": best_amort["battery_size_kwh"],
            "amortization_years": best_amort["amortization_years"],
            "annualized_gain_chf": best_amort["annualized_gain_chf"],
            "total_gain_chf": best_amort["total_gain_chf"],
            "bill_reduction_pct_vs_no_battery": best_amort["bill_reduction_pct_vs_no_battery"],
            "selection_rule": (
                "Select the largest battery size whose amortization stays at or below the configured maximum."
            ),
            "reason": (
                "No analyzed battery size meets the amortization cap; "
                "fallback to the battery with the lowest amortization."
            ),
        }

    return {
        "name": "Global Rentability Overview – Amortization affordability cap KPI",
        "metric": "amortization_years",
        "threshold_amortization_years_max": round_or_none(amortization_years_max),
        "selection_rule": (
            "Treat battery sizes with amortization > threshold as too expensive and "
            "recommend the largest size with amortization <= threshold."
        ),
        "recommendation": recommendation,
        "evaluated_candidates": evaluated,
    }


def compute_global_battery_utilization_cycle_wear_cap_kpi(
    candidates: list[dict[str, Any]],
    *,
    pct_max_cycles_per_year_max: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for Global Battery Utilization:
    exclude battery sizes whose % of max cycles/year exceeds the configured threshold,
    and recommend the smallest battery size that remains at or below the threshold.
    """
    pct_max_cycles_per_year_max = positive_or_default(pct_max_cycles_per_year_max, default=4.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cand in ordered:
        if cand.get("is_baseline"):
            continue
        pct = metric_value(cand, "pct_max_cycles_per_year")
        cycles = metric_value(cand, "cycles_per_year")
        if pct is None:
            continue
        within = pct <= pct_max_cycles_per_year_max
        implied_life_years = None if pct <= 0 else (100.0 / pct)
        row = {
            "scenario": cand.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "cycles_per_year": round_or_none(cycles),
            "pct_max_cycles_per_year": round_or_none(pct),
            "implied_life_years_from_pct_max_cycles": round_or_none(implied_life_years),
            "within_cycle_wear_cap": within,
            "status": "acceptable" if within else "too_high_cycle_wear",
        }
        evaluated.append(row)
        if within and selected is None:
            # Smallest acceptable size (first match in ascending order).
            selected = row

    if not evaluated:
        return None

    recommendation: dict[str, Any]
    if selected is not None:
        recommendation = {
            "scenario": selected["scenario"],
            "battery_size_kwh": selected["battery_size_kwh"],
            "cycles_per_year": selected["cycles_per_year"],
            "pct_max_cycles_per_year": selected["pct_max_cycles_per_year"],
            "implied_life_years_from_pct_max_cycles": selected["implied_life_years_from_pct_max_cycles"],
            "selection_rule": (
                "Exclude sizes above the max % of cycles/year threshold and select the smallest size that stays within the cap."
            ),
            "reason": (
                f"Selected as the smallest analyzed size with cycle wear "
                f"{selected['pct_max_cycles_per_year']}% <= {round_or_none(pct_max_cycles_per_year_max)}% of max cycles/year "
                f"(approx. {selected['implied_life_years_from_pct_max_cycles']} years from this proxy)."
            ),
        }
        prev_excluded = next(
            (
                row
                for row in evaluated
                if not row.get("within_cycle_wear_cap")
                and (row.get("battery_size_kwh") or 0.0) < (selected.get("battery_size_kwh") or 0.0)
            ),
            None,
        )
        if prev_excluded:
            recommendation["previous_excluded_candidate"] = {
                "scenario": prev_excluded["scenario"],
                "battery_size_kwh": prev_excluded["battery_size_kwh"],
                "pct_max_cycles_per_year": prev_excluded["pct_max_cycles_per_year"],
                "implied_life_years_from_pct_max_cycles": prev_excluded["implied_life_years_from_pct_max_cycles"],
            }
    else:
        best_durable = min(
            evaluated,
            key=lambda row: (
                row.get("pct_max_cycles_per_year") if row.get("pct_max_cycles_per_year") is not None else float("inf"),
                row.get("battery_size_kwh") if row.get("battery_size_kwh") is not None else float("inf"),
            ),
        )
        recommendation = {
            "scenario": best_durable["scenario"],
            "battery_size_kwh": best_durable["battery_size_kwh"],
            "cycles_per_year": best_durable["cycles_per_year"],
            "pct_max_cycles_per_year": best_durable["pct_max_cycles_per_year"],
            "implied_life_years_from_pct_max_cycles": best_durable["implied_life_years_from_pct_max_cycles"],
            "selection_rule": (
                "Exclude sizes above the max % of cycles/year threshold and select the smallest size that stays within the cap."
            ),
            "reason": (
                "No analyzed battery size meets the cycle-wear cap; "
                "fallback to the battery with the lowest % of max cycles/year."
            ),
        }

    return {
        "name": "Global Battery Utilization – Cycle wear cap KPI",
        "metric": "pct_max_cycles_per_year",
        "threshold_pct_max_cycles_per_year_max": round_or_none(pct_max_cycles_per_year_max),
        "implied_min_life_years_from_threshold": round_or_none(100.0 / pct_max_cycles_per_year_max)
        if pct_max_cycles_per_year_max > 0
        else None,
        "selection_rule": (
            "Treat battery sizes with % of max cycles/year > threshold as excluded and "
            "recommend the smallest size with % of max cycles/year <= threshold."
        ),
        "recommendation": recommendation,
        "evaluated_candidates": evaluated,
    }


def compute_global_battery_status_heatmap_empty_cap_kpi(
    candidates: list[dict[str, Any]],
    *,
    empty_pct_max: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for Global Battery Status Heatmap:
    exclude battery sizes whose average (across phases) empty-share exceeds the threshold.
    Recommend the smallest battery size satisfying the constraint.
    """
    threshold = positive_or_default(empty_pct_max, default=20.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cand in ordered:
        if cand.get("is_baseline"):
            continue
        empty_pct = metric_value(cand, "status_empty_pct")
        if empty_pct is None:
            continue
        row = {
            "scenario": cand.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "status_empty_pct": round_or_none(empty_pct),
            "status_full_pct": round_or_none(metric_value(cand, "status_full_pct")),
            "status_charging_pct": round_or_none(metric_value(cand, "status_charging_pct")),
            "status_discharging_pct": round_or_none(metric_value(cand, "status_discharging_pct")),
            "status_idle_pct": round_or_none(metric_value(cand, "status_idle_pct")),
            "within_empty_share_cap": bool(empty_pct <= threshold),
            "status": "acceptable" if empty_pct <= threshold else "too_often_empty",
        }
        evaluated.append(row)
        if row["within_empty_share_cap"] and selected is None:
            selected = row

    if not evaluated:
        return None

    recommendation: dict[str, Any]
    if selected is not None:
        recommendation = {
            "scenario": selected["scenario"],
            "battery_size_kwh": selected["battery_size_kwh"],
            "status_empty_pct": selected["status_empty_pct"],
            "status_full_pct": selected["status_full_pct"],
            "selection_rule": (
                "Exclude sizes above the global empty-share threshold and select the smallest size within the cap."
            ),
            "reason": (
                f"Selected as the smallest analyzed size with global average empty share "
                f"{selected['status_empty_pct']}% <= {round_or_none(threshold)}%."
            ),
        }
        prev_excluded_candidates = [
            row
            for row in evaluated
            if not row.get("within_empty_share_cap")
            and (row.get("battery_size_kwh") or 0.0) < (selected.get("battery_size_kwh") or 0.0)
        ]
        prev_excluded = (
            max(prev_excluded_candidates, key=lambda row: (row.get("battery_size_kwh") or 0.0))
            if prev_excluded_candidates
            else None
        )
        if prev_excluded:
            recommendation["previous_excluded_candidate"] = {
                "scenario": prev_excluded["scenario"],
                "battery_size_kwh": prev_excluded["battery_size_kwh"],
                "status_empty_pct": prev_excluded["status_empty_pct"],
            }
    else:
        best = min(
            evaluated,
            key=lambda row: (
                row.get("status_empty_pct") if row.get("status_empty_pct") is not None else float("inf"),
                row.get("battery_size_kwh") if row.get("battery_size_kwh") is not None else float("inf"),
            ),
        )
        recommendation = {
            "scenario": best["scenario"],
            "battery_size_kwh": best["battery_size_kwh"],
            "status_empty_pct": best["status_empty_pct"],
            "status_full_pct": best["status_full_pct"],
            "selection_rule": (
                "Exclude sizes above the global empty-share threshold and select the smallest size within the cap."
            ),
            "reason": (
                "No analyzed battery size meets the global empty-share cap; "
                "fallback to the battery with the lowest empty-share percentage."
            ),
        }

    return {
        "name": "Global Battery Status Heatmap – Empty-share cap KPI",
        "metric": "status_empty_pct",
        "threshold_status_empty_pct_max": round_or_none(threshold),
        "selection_rule": (
            "Treat battery sizes as excluded if global average empty-share (heatmap 'Empty', average across phases) "
            "exceeds the threshold; recommend the smallest size that satisfies the cap."
        ),
        "recommendation": recommendation,
        "evaluated_candidates": evaluated,
    }


def compute_seasonal_power_saturation_at_max_limit_cap_kpi(
    candidates: list[dict[str, Any]],
    *,
    power_saturation_pct_any_season_max: float,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for "Power Saturation At Max Limit Vs Battery Size Per Season":
    exclude battery sizes if any season has power_at_max_pct above the threshold.
    Recommend the smallest battery size that satisfies the constraint.
    """
    threshold = positive_or_default(power_saturation_pct_any_season_max, default=10.0)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cand in ordered:
        if cand.get("is_baseline"):
            continue
        seasonal_blob = cand.get("seasonal_by_season", {}) or {}
        season_values: dict[str, float | None] = {}
        for season in SEASON_ORDER:
            v = safe_float((seasonal_blob.get(season) or {}).get("power_at_max_pct"))
            season_values[season] = round_or_none(v) if v is not None else None
        values_present = [v for v in season_values.values() if v is not None]
        if not values_present:
            continue
        max_any = max(values_present)
        within = max_any <= threshold
        violating_seasons = [season for season in SEASON_ORDER if (season_values.get(season) or 0.0) > threshold]
        row = {
            "scenario": cand.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "seasonal_power_at_max_pct": season_values,
            "max_power_saturation_pct_any_season": round_or_none(max_any),
            "violating_seasons": violating_seasons,
            "within_power_saturation_cap": within,
            "status": "acceptable" if within else "too_high_power_saturation",
        }
        evaluated.append(row)
        if within and selected is None:
            selected = row  # smallest acceptable size

    if not evaluated:
        return None

    recommendation: dict[str, Any]
    if selected is not None:
        recommendation = {
            "scenario": selected["scenario"],
            "battery_size_kwh": selected["battery_size_kwh"],
            "max_power_saturation_pct_any_season": selected["max_power_saturation_pct_any_season"],
            "seasonal_power_at_max_pct": selected["seasonal_power_at_max_pct"],
            "selection_rule": (
                "Exclude sizes where any season exceeds the max power saturation threshold and "
                "select the smallest size that remains within the cap."
            ),
            "reason": (
                f"Selected as the smallest analyzed size with worst-season power saturation "
                f"{selected['max_power_saturation_pct_any_season']}% <= {round_or_none(threshold)}%."
            ),
        }
        prev_excluded = next(
            (
                row
                for row in evaluated
                if not row.get("within_power_saturation_cap")
                and (row.get("battery_size_kwh") or 0.0) < (selected.get("battery_size_kwh") or 0.0)
            ),
            None,
        )
        if prev_excluded:
            recommendation["previous_excluded_candidate"] = {
                "scenario": prev_excluded["scenario"],
                "battery_size_kwh": prev_excluded["battery_size_kwh"],
                "max_power_saturation_pct_any_season": prev_excluded["max_power_saturation_pct_any_season"],
                "violating_seasons": prev_excluded["violating_seasons"],
            }
    else:
        # No size meets threshold: return the least-bad candidate (lowest worst-season saturation).
        best = min(
            evaluated,
            key=lambda row: (
                row.get("max_power_saturation_pct_any_season")
                if row.get("max_power_saturation_pct_any_season") is not None
                else float("inf"),
                row.get("battery_size_kwh") if row.get("battery_size_kwh") is not None else float("inf"),
            ),
        )
        recommendation = {
            "scenario": best["scenario"],
            "battery_size_kwh": best["battery_size_kwh"],
            "max_power_saturation_pct_any_season": best["max_power_saturation_pct_any_season"],
            "seasonal_power_at_max_pct": best["seasonal_power_at_max_pct"],
            "selection_rule": (
                "Exclude sizes where any season exceeds the max power saturation threshold and "
                "select the smallest size that remains within the cap."
            ),
            "reason": (
                "No analyzed battery size meets the seasonal power saturation cap; "
                "fallback to the battery with the lowest worst-season power saturation."
            ),
        }

    return {
        "name": "Seasonal Power Saturation At Max Limit – Constraint KPI",
        "metric": "max_season_power_at_max_pct",
        "threshold_power_saturation_pct_any_season_max": round_or_none(threshold),
        "selection_rule": (
            "Treat battery sizes as excluded if any season power saturation at max limit exceeds the threshold; "
            "recommend the smallest size that satisfies the cap."
        ),
        "recommendation": recommendation,
        "evaluated_candidates": evaluated,
    }


def compute_monthly_structural_evening_energy_undersizing_peak_period_cap_kpi(
    candidates: list[dict[str, Any]],
    *,
    evening_undersize_pct_per_month_max: float,
    max_months_above_threshold: int,
) -> dict[str, Any] | None:
    """
    Graph-informed KPI for "Monthly Structural Evening Energy Undersizing (Peak Period)":
    exclude battery sizes if the monthly evening undersize percentage exceeds the threshold
    in more than the allowed number of months.
    Recommend the smallest battery size satisfying the rule.
    """
    month_threshold = positive_or_default(evening_undersize_pct_per_month_max, default=40.0)
    allowed_months = nonnegative_int_or_default(max_months_above_threshold, default=1)
    ordered = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))

    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for cand in ordered:
        if cand.get("is_baseline"):
            continue
        monthly = cand.get("monthly_evening_undersize_pct_by_month", {}) or {}
        if not isinstance(monthly, dict) or not monthly:
            continue

        normalized_monthly: dict[str, float | None] = {
            str(month): round_or_none(safe_float(value) or 0.0)
            for month, value in sorted(monthly.items())
        }
        months_above = []
        for month, value in normalized_monthly.items():
            if value is not None and value > month_threshold:
                months_above.append({"month": month, "evening_undersize_pct": value})

        max_monthly_pct = max((v for v in normalized_monthly.values() if v is not None), default=None)
        within = len(months_above) <= allowed_months
        row = {
            "scenario": cand.get("scenario"),
            "battery_size_kwh": round_or_none(safe_float(cand.get("battery_size_kwh"))),
            "monthly_evening_undersize_pct_by_month": normalized_monthly,
            "months_above_threshold": months_above,
            "months_above_threshold_count": len(months_above),
            "max_monthly_evening_undersize_pct": round_or_none(max_monthly_pct),
            "within_monthly_evening_undersize_cap": within,
            "status": "acceptable" if within else "too_many_high_undersize_months",
        }
        evaluated.append(row)
        if within and selected is None:
            selected = row

    if not evaluated:
        return None

    recommendation: dict[str, Any]
    if selected is not None:
        recommendation = {
            "scenario": selected["scenario"],
            "battery_size_kwh": selected["battery_size_kwh"],
            "months_above_threshold_count": selected["months_above_threshold_count"],
            "months_above_threshold": selected["months_above_threshold"],
            "max_monthly_evening_undersize_pct": selected["max_monthly_evening_undersize_pct"],
            "selection_rule": (
                "Exclude sizes with more than the allowed number of months above the monthly evening undersizing threshold; "
                "select the smallest size that satisfies the cap."
            ),
            "reason": (
                f"Selected as the smallest analyzed size with "
                f"{selected['months_above_threshold_count']} month(s) above {round_or_none(month_threshold)}% "
                f"(allowed maximum: {allowed_months})."
            ),
        }
        prev_excluded = next(
            (
                row
                for row in evaluated
                if not row.get("within_monthly_evening_undersize_cap")
                and (row.get("battery_size_kwh") or 0.0) < (selected.get("battery_size_kwh") or 0.0)
            ),
            None,
        )
        if prev_excluded:
            recommendation["previous_excluded_candidate"] = {
                "scenario": prev_excluded["scenario"],
                "battery_size_kwh": prev_excluded["battery_size_kwh"],
                "months_above_threshold_count": prev_excluded["months_above_threshold_count"],
                "months_above_threshold": prev_excluded["months_above_threshold"],
            }
    else:
        best = min(
            evaluated,
            key=lambda row: (
                row.get("months_above_threshold_count", 0),
                row.get("max_monthly_evening_undersize_pct")
                if row.get("max_monthly_evening_undersize_pct") is not None
                else float("inf"),
                row.get("battery_size_kwh") if row.get("battery_size_kwh") is not None else float("inf"),
            ),
        )
        recommendation = {
            "scenario": best["scenario"],
            "battery_size_kwh": best["battery_size_kwh"],
            "months_above_threshold_count": best["months_above_threshold_count"],
            "months_above_threshold": best["months_above_threshold"],
            "max_monthly_evening_undersize_pct": best["max_monthly_evening_undersize_pct"],
            "selection_rule": (
                "Exclude sizes with more than the allowed number of months above the monthly evening undersizing threshold; "
                "select the smallest size that satisfies the cap."
            ),
            "reason": (
                "No analyzed battery size meets the monthly evening undersizing cap; "
                "fallback to the battery with the fewest violating months."
            ),
        }

    return {
        "name": "Monthly Structural Evening Energy Undersizing Peak Period – Constraint KPI",
        "metric": "monthly_evening_undersize_pct_by_month",
        "threshold_evening_undersize_pct_per_month_max": round_or_none(month_threshold),
        "threshold_max_months_above_threshold": allowed_months,
        "selection_rule": (
            "Treat battery sizes as excluded if more than the allowed number of months exceed the monthly evening "
            "undersizing threshold; recommend the smallest size that satisfies the cap."
        ),
        "recommendation": recommendation,
        "evaluated_candidates": evaluated,
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


def compute_highest_seasonal_gain_absolute(candidates: list[dict[str, Any]]) -> dict[str, Any]:
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


def compute_seasonal_smallest_good_enough(
    candidates: list[dict[str, Any]],
    *,
    gain_fraction_of_max: float = 0.90,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pool = sorted([c for c in candidates if not c["is_baseline"]], key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    for season in SEASON_ORDER:
        values: list[tuple[dict[str, Any], float]] = []
        for cand in pool:
            season_blob = cand.get("seasonal_by_season", {}).get(season, {})
            v = safe_float(season_blob.get("total_gain_chf"))
            if v is None:
                continue
            values.append((cand, v))
        if not values:
            continue

        max_gain = max(v for _, v in values)
        target_gain = max_gain * gain_fraction_of_max
        eligible = [(cand, v) for cand, v in values if v >= target_gain]
        winner_cand, winner_val = min(eligible, key=lambda item: (item[0]["battery_size_kwh"], item[0]["scenario"]))

        out[season] = {
            "scenario": winner_cand["scenario"],
            "battery_size_kwh": winner_cand["battery_size_kwh"],
            "season_total_gain_chf": round_or_none(winner_val),
            "max_season_gain_chf": round_or_none(max_gain),
            "target_fraction_of_max_gain": round_or_none(gain_fraction_of_max, 3),
            "target_gain_chf": round_or_none(target_gain),
            "reason": (
                f"Smallest battery size reaching at least {round(gain_fraction_of_max * 100)}% "
                "of the maximum seasonal net financial gain."
            ),
        }
    return out


def compute_seasonal_knee_points(
    candidates: list[dict[str, Any]],
    marginal_gain_threshold: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ordered_candidates = sorted(candidates, key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    for season in SEASON_ORDER:
        rows: list[tuple[dict[str, Any], float]] = []
        for cand in ordered_candidates:
            season_blob = cand.get("seasonal_by_season", {}).get(season, {})
            v = safe_float(season_blob.get("total_gain_chf"))
            if v is None:
                continue
            rows.append((cand, v))

        nonbaseline_rows = [row for row in rows if not row[0].get("is_baseline")]
        if not nonbaseline_rows:
            continue

        if len(nonbaseline_rows) == 1:
            only_cand, _only_gain = nonbaseline_rows[0]
            out[season] = {
                "scenario": only_cand["scenario"],
                "battery_size_kwh": only_cand["battery_size_kwh"],
                "reason": "Only one non-baseline battery configuration available for this season.",
                "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
                "trigger_pair": None,
            }
            continue

        previous: tuple[dict[str, Any], float] | None = None
        selected: dict[str, Any] | None = None
        for cand, gain in rows:
            if previous is None:
                previous = (cand, gain)
                continue
            if cand.get("is_baseline"):
                previous = (cand, gain)
                continue

            prev_cand, prev_gain = previous
            delta_size = float(cand["battery_size_kwh"]) - float(prev_cand["battery_size_kwh"])
            if delta_size <= 0:
                previous = (cand, gain)
                continue

            marginal = (gain - prev_gain) / delta_size
            if marginal < marginal_gain_threshold:
                if prev_cand.get("is_baseline"):
                    selected = {
                        "scenario": cand["scenario"],
                        "battery_size_kwh": cand["battery_size_kwh"],
                        "reason": (
                            "First non-baseline step already falls below marginal gain threshold; "
                            "selected the smallest analyzed battery for this season."
                        ),
                        "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
                        "trigger_pair": {
                            "from_scenario": prev_cand["scenario"],
                            "to_scenario": cand["scenario"],
                            "to_battery_size_kwh": cand["battery_size_kwh"],
                            "marginal_season_gain_chf_per_added_kwh": round_or_none(marginal),
                        },
                    }
                    break
                selected = {
                    "scenario": prev_cand["scenario"],
                    "battery_size_kwh": prev_cand["battery_size_kwh"],
                    "reason": (
                        "First size before marginal seasonal net gain per added kWh drops below threshold."
                    ),
                    "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
                    "trigger_pair": {
                        "from_scenario": prev_cand["scenario"],
                        "to_scenario": cand["scenario"],
                        "to_battery_size_kwh": cand["battery_size_kwh"],
                        "marginal_season_gain_chf_per_added_kwh": round_or_none(marginal),
                    },
                }
                break
            previous = (cand, gain)

        if selected is None:
            largest_cand, _largest_gain = max(nonbaseline_rows, key=lambda item: item[0]["battery_size_kwh"])
            selected = {
                "scenario": largest_cand["scenario"],
                "battery_size_kwh": largest_cand["battery_size_kwh"],
                "reason": "Marginal seasonal net gain per added kWh never dropped below threshold.",
                "marginal_gain_threshold_chf_per_added_kwh": marginal_gain_threshold,
                "trigger_pair": None,
            }
        out[season] = selected
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
    lines.append(f"- Generated at: {summary.get('generated_at_utc')}")
    source_files = summary.get("source_files") or []
    lines.append(f"- Simulation files analyzed: {len(source_files)}")
    candidates = summary.get("candidates")
    if isinstance(candidates, list):
        lines.append(f"- Non-baseline candidates ranked: {len([c for c in candidates if not c.get('is_baseline')])}")
    if summary.get("summary_mode") == "graph_kpis_only":
        graph_kpi_count = 0
        for group in (summary.get("graph_kpis") or {}).values():
            if isinstance(group, dict):
                graph_kpi_count += sum(1 for v in group.values() if v)
        lines.append(f"- Graph KPI sections included: `{graph_kpi_count}`")
    if summary.get("kpi_config_source"):
        lines.append(f"- KPI config source: `{summary['kpi_config_source']}`")
    kpi_settings = summary.get("kpi_settings", {})
    if kpi_settings:
        if summary.get("summary_mode") == "graph_kpis_only":
            lines.append("- KPI scope: graph-KPI-only summary (generic profile scoring and non-graph KPI sections omitted).")
        elif (
            kpi_settings.get("global_knee_marginal_gain_chf_per_added_kwh") is not None
            or kpi_settings.get("seasonal_knee_marginal_gain_chf_per_added_kwh") is not None
        ):
            lines.append(
                "- KPI thresholds: "
                f"global knee `{kpi_settings.get('global_knee_marginal_gain_chf_per_added_kwh')}` CHF/year per added kWh; "
                f"seasonal knee `{kpi_settings.get('seasonal_knee_marginal_gain_chf_per_added_kwh')}` CHF/season per added kWh"
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

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("global_energy_reduction_kwh", {})
        .get("consumed_reduction_step_delta_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Global Energy Reduction (Consumed % Step Delta)")
        lines.append("")
        lines.append(
            f"- Threshold: `{graph_kpi.get('threshold_increment_pct_points')} percentage points` incremental gain vs previous size"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with consumed reduction `{rec.get('grid_consumed_reduction_pct_vs_no_battery')}%`"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
        lines.append("")
        steps = graph_kpi.get("steps") or []
        if steps:
            lines.append("| From Size (kWh) | To Size (kWh) | From Reduction (%) | To Reduction (%) | Delta (pp) | Meets Threshold |")
            lines.append("|---:|---:|---:|---:|---:|:---:|")
            for step in steps:
                meets = "yes" if step.get("meets_increment_threshold") else "no"
                lines.append(
                    f"| {step.get('from_battery_size_kwh','')} | {step.get('to_battery_size_kwh','')} | "
                    f"{step.get('from_grid_consumed_reduction_pct_vs_no_battery','')} | "
                    f"{step.get('to_grid_consumed_reduction_pct_vs_no_battery','')} | "
                    f"{step.get('delta_pct_points_vs_previous_size','')} | {meets} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("global_energy_financial_impact_chf", {})
        .get("bill_offset_step_delta_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Global Energy Financial Impact (Bill Offset % Step Delta)")
        lines.append("")
        lines.append(
            f"- Threshold: `{graph_kpi.get('threshold_increment_pct_points')} percentage points` incremental gain vs previous size"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with bill offset `{rec.get('bill_offset_pct_vs_no_battery')}%`"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
        lines.append("")
        steps = graph_kpi.get("steps") or []
        if steps:
            lines.append("| From Size (kWh) | To Size (kWh) | From Bill Offset (%) | To Bill Offset (%) | Delta (pp) | Meets Threshold |")
            lines.append("|---:|---:|---:|---:|---:|:---:|")
            for step in steps:
                meets = "yes" if step.get("meets_increment_threshold") else "no"
                lines.append(
                    f"| {step.get('from_battery_size_kwh','')} | {step.get('to_battery_size_kwh','')} | "
                    f"{step.get('from_bill_offset_pct_vs_no_battery','')} | "
                    f"{step.get('to_bill_offset_pct_vs_no_battery','')} | "
                    f"{step.get('delta_pct_points_vs_previous_size','')} | {meets} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("global_rentability_overview", {})
        .get("amortization_cap_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Global Rentability Overview (Amortization Cap)")
        lines.append("")
        lines.append(f"- Threshold: amortization `<= {graph_kpi.get('threshold_amortization_years_max')} years`")
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with amortization `{rec.get('amortization_years')} years`"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
            next_bad = rec.get("next_too_expensive_candidate") or {}
            if next_bad:
                lines.append(
                    "- Next size excluded as too expensive: "
                    f"`{next_bad.get('battery_size_kwh')} kWh` (`{next_bad.get('scenario')}`) "
                    f"with amortization `{next_bad.get('amortization_years')} years`"
                )
        lines.append("")
        rows = graph_kpi.get("evaluated_candidates") or []
        if rows:
            lines.append("| Battery Size (kWh) | Scenario | Amortization (years) | Annualized Gain (CHF/year) | Bill Offset (%) | Status |")
            lines.append("|---:|---|---:|---:|---:|---|")
            for row in rows:
                lines.append(
                    f"| {row.get('battery_size_kwh','')} | {row.get('scenario','')} | "
                    f"{row.get('amortization_years','')} | {row.get('annualized_gain_chf','')} | "
                    f"{row.get('bill_reduction_pct_vs_no_battery','')} | {row.get('status','')} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("global_battery_utilization", {})
        .get("cycle_wear_cap_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Global Battery Utilization (Cycle Wear Cap)")
        lines.append("")
        lines.append(
            f"- Threshold: `% of max cycles/year <= {graph_kpi.get('threshold_pct_max_cycles_per_year_max')}%` "
            f"(proxy minimum life `~{graph_kpi.get('implied_min_life_years_from_threshold')} years`)"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with `{rec.get('pct_max_cycles_per_year')}%` of max cycles/year "
                f"(`{rec.get('cycles_per_year')} cycles/year`, proxy life `~{rec.get('implied_life_years_from_pct_max_cycles')} years`)"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
            prev_bad = rec.get("previous_excluded_candidate") or {}
            if prev_bad:
                lines.append(
                    "- Previous size excluded for high cycle wear: "
                    f"`{prev_bad.get('battery_size_kwh')} kWh` (`{prev_bad.get('scenario')}`) "
                    f"at `{prev_bad.get('pct_max_cycles_per_year')}%` "
                    f"(proxy life `~{prev_bad.get('implied_life_years_from_pct_max_cycles')} years`)"
                )
        lines.append("")
        rows = graph_kpi.get("evaluated_candidates") or []
        if rows:
            lines.append("| Battery Size (kWh) | Scenario | Cycles/Year | % Max Cycles/Year | Proxy Life (years) | Status |")
            lines.append("|---:|---|---:|---:|---:|---|")
            for row in rows:
                lines.append(
                    f"| {row.get('battery_size_kwh','')} | {row.get('scenario','')} | "
                    f"{row.get('cycles_per_year','')} | {row.get('pct_max_cycles_per_year','')} | "
                    f"{row.get('implied_life_years_from_pct_max_cycles','')} | {row.get('status','')} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("global_battery_status_heatmap", {})
        .get("empty_share_cap_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Global Battery Status Heatmap (Empty Share Cap)")
        lines.append("")
        lines.append(
            f"- Threshold: global heatmap `Empty` share (average across phases) `<= {graph_kpi.get('threshold_status_empty_pct_max')}%`"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with empty share `{rec.get('status_empty_pct')}%`"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
            prev_bad = rec.get("previous_excluded_candidate") or {}
            if prev_bad:
                lines.append(
                    "- Previous size excluded for excessive empty time: "
                    f"`{prev_bad.get('battery_size_kwh')} kWh` (`{prev_bad.get('scenario')}`) "
                    f"with empty share `{prev_bad.get('status_empty_pct')}%`"
                )
        lines.append("")
        rows = graph_kpi.get("evaluated_candidates") or []
        if rows:
            lines.append(
                "| Battery Size (kWh) | Scenario | Empty (%) | Full (%) | Charging (%) | Discharging (%) | Idle (%) | Status |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
            for row in rows:
                lines.append(
                    f"| {row.get('battery_size_kwh','')} | {row.get('scenario','')} | "
                    f"{row.get('status_empty_pct','')} | {row.get('status_full_pct','')} | "
                    f"{row.get('status_charging_pct','')} | {row.get('status_discharging_pct','')} | "
                    f"{row.get('status_idle_pct','')} | {row.get('status','')} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("seasonal_power_saturation_at_max_limit", {})
        .get("constraint_cap_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Seasonal Power Saturation At Max Limit (Constraint Cap)")
        lines.append("")
        lines.append(
            f"- Threshold: `power saturation at max limit <= {graph_kpi.get('threshold_power_saturation_pct_any_season_max')}%` for every season"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with worst-season power saturation `{rec.get('max_power_saturation_pct_any_season')}%`"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
            prev_bad = rec.get("previous_excluded_candidate") or {}
            if prev_bad:
                seasons = ", ".join(prev_bad.get("violating_seasons") or [])
                lines.append(
                    "- Previous size excluded: "
                    f"`{prev_bad.get('battery_size_kwh')} kWh` (`{prev_bad.get('scenario')}`) "
                    f"with worst-season `{prev_bad.get('max_power_saturation_pct_any_season')}%` "
                    f"(violating season(s): {seasons})"
                )
        lines.append("")
        rows = graph_kpi.get("evaluated_candidates") or []
        if rows:
            lines.append(
                "| Battery Size (kWh) | Scenario | Spring (%) | Summer (%) | Autumn (%) | Winter (%) | Worst Season (%) | Status |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
            for row in rows:
                by_season = row.get("seasonal_power_at_max_pct") or {}
                lines.append(
                    f"| {row.get('battery_size_kwh','')} | {row.get('scenario','')} | "
                    f"{by_season.get('spring','')} | {by_season.get('summer','')} | "
                    f"{by_season.get('autumn','')} | {by_season.get('winter','')} | "
                    f"{row.get('max_power_saturation_pct_any_season','')} | {row.get('status','')} |"
                )
            lines.append("")

    graph_kpi = (
        summary.get("graph_kpis", {})
        .get("monthly_structural_evening_energy_undersizing_peak_period", {})
        .get("constraint_cap_kpi")
    )
    if graph_kpi:
        lines.append("## Graph KPI: Monthly Structural Evening Energy Undersizing Peak Period (Constraint Cap)")
        lines.append("")
        lines.append(
            f"- Threshold: monthly evening undersizing `<= {graph_kpi.get('threshold_evening_undersize_pct_per_month_max')}%` "
            f"for at least all but `{graph_kpi.get('threshold_max_months_above_threshold')}` month(s)"
        )
        lines.append(f"- Selection rule: {graph_kpi.get('selection_rule')}")
        rec = graph_kpi.get("recommendation") or {}
        if rec:
            lines.append(
                f"- Recommendation: `{rec.get('battery_size_kwh')} kWh` (`{rec.get('scenario')}`) "
                f"with `{rec.get('months_above_threshold_count')} month(s)` above threshold "
                f"(worst month `{rec.get('max_monthly_evening_undersize_pct')}%`)"
            )
            lines.append(f"- Rationale: {rec.get('reason')}")
            prev_bad = rec.get("previous_excluded_candidate") or {}
            if prev_bad:
                violating = ", ".join(
                    f"{m.get('month')} ({m.get('evening_undersize_pct')}%)"
                    for m in (prev_bad.get("months_above_threshold") or [])
                )
                lines.append(
                    "- Previous size excluded: "
                    f"`{prev_bad.get('battery_size_kwh')} kWh` (`{prev_bad.get('scenario')}`) "
                    f"with `{prev_bad.get('months_above_threshold_count')} violating month(s)`"
                    + (f": {violating}" if violating else "")
                )
        lines.append("")
        rows = graph_kpi.get("evaluated_candidates") or []
        if rows:
            lines.append(
                "| Battery Size (kWh) | Scenario | Months > Threshold | Worst Month (%) | Violating Months | Status |"
            )
            lines.append("|---:|---|---:|---:|---|---|")
            for row in rows:
                viol = ", ".join(
                    f"{m.get('month')} ({m.get('evening_undersize_pct')}%)"
                    for m in (row.get("months_above_threshold") or [])
                )
                lines.append(
                    f"| {row.get('battery_size_kwh','')} | {row.get('scenario','')} | "
                    f"{row.get('months_above_threshold_count','')} | {row.get('max_monthly_evening_undersize_pct','')} | "
                    f"{viol} | {row.get('status','')} |"
                )
            lines.append("")

    if summary.get("summary_mode") == "graph_kpis_only":
        lines.append("## Notes for LLM Recommendation")
        lines.append("")
        lines.append("- Use these 7 graph KPIs as the authoritative sizing constraints and step-delta rules.")
        lines.append("- When graph KPIs disagree, explain the tradeoff explicitly instead of averaging them into a hidden score.")
        lines.append("- Treat constraint-cap KPIs as exclusions first, then use the remaining graph KPIs to justify the final size.")
        lines.append("- When bill offset exceeds 100%, interpret it as bill elimination plus net credit.")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

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

    seasonal_smallest_good_enough = summary.get("seasonal_smallest_good_enough", {})
    if seasonal_smallest_good_enough:
        lines.append("## Seasonal Smallest Good-Enough Battery Size")
        lines.append("")
        for season in SEASON_ORDER:
            item = seasonal_smallest_good_enough.get(season)
            if not item:
                continue
            lines.append(
                f"- `{season}`: `{item['battery_size_kwh']} kWh` (`{item['scenario']}`) "
                f"with `{item['season_total_gain_chf']} CHF` "
                f"(target `{item['target_gain_chf']} CHF` = {int(round(float(item['target_fraction_of_max_gain']) * 100))}% of max `{item['max_season_gain_chf']} CHF`)"
            )
        lines.append("")

    seasonal_knees = summary.get("seasonal_knee_points", {})
    if seasonal_knees:
        lines.append("## Seasonal Knee Points (Diminishing Returns)")
        lines.append("")
        for season in SEASON_ORDER:
            item = seasonal_knees.get(season)
            if not item:
                continue
            lines.append(
                f"- `{season}`: `{item['battery_size_kwh']} kWh` (`{item['scenario']}`) "
                f"threshold `{item['marginal_gain_threshold_chf_per_added_kwh']} CHF per added kWh`"
            )
            trigger = item.get("trigger_pair")
            if trigger:
                lines.append(
                    "  Trigger: "
                    f"`{trigger.get('from_scenario')}` -> `{trigger.get('to_scenario')}` "
                    f"(marginal `{trigger.get('marginal_season_gain_chf_per_added_kwh')} CHF per added kWh`)"
                )
        lines.append("")

    seasonal_absolute_max = summary.get("highest_seasonal_gain_absolute", {})
    if seasonal_absolute_max:
        lines.append("## Highest Seasonal Net Gain (Absolute, Not a Sizing KPI)")
        lines.append("")
        lines.append(
            "- This section is kept for reference only. It often favors the largest battery size because it uses absolute seasonal CHF gain."
        )
        lines.append("")
        for season in SEASON_ORDER:
            item = seasonal_absolute_max.get(season)
            if not item:
                continue
            lines.append(
                f"- `{season}`: `{item['battery_size_kwh']} kWh` (`{item['scenario']}`) "
                f"with absolute seasonal max `{item['season_total_gain_chf']} CHF`"
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
    lines.append(
        "- Use the graph KPI `Global Energy Reduction (Consumed % Step Delta)` to justify the last battery size that still delivers a meaningful incremental consumption-reduction gain vs the previous size."
    )
    lines.append(
        "- Use the graph KPI `Global Energy Financial Impact (Bill Offset % Step Delta)` to justify the last battery size that still delivers a meaningful incremental bill-offset gain vs the previous size."
    )
    lines.append(
        "- Use the graph KPI `Global Rentability Overview (Amortization Cap)` to exclude battery sizes whose amortization exceeds the user-defined affordability threshold."
    )
    lines.append(
        "- Use the graph KPI `Global Battery Utilization (Cycle Wear Cap)` to exclude battery sizes that exceed the allowed % of max cycles/year (durability proxy)."
    )
    lines.append(
        "- Use the graph KPI `Global Battery Status Heatmap (Empty Share Cap)` to exclude battery sizes that are empty too often (global heatmap Empty % average across phases)."
    )
    lines.append(
        "- Use the graph KPI `Seasonal Power Saturation At Max Limit (Constraint Cap)` to exclude battery sizes where any season exceeds the allowed power saturation threshold."
    )
    lines.append(
        "- Use the graph KPI `Monthly Structural Evening Energy Undersizing Peak Period (Constraint Cap)` to exclude battery sizes with too many months above the allowed evening undersizing threshold."
    )
    lines.append("- Prefer `seasonal_smallest_good_enough` over absolute seasonal max-gain when discussing seasonal sizing.")
    lines.append("- When bill reduction exceeds 100%, interpret it as bill elimination plus net credit.")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute battery-sizing KPI rankings from simulation output JSON files."
    )
    parser.add_argument(
        "--config",
        default="config/kpi_scoring.json",
        help="KPI tuning config JSON path (default: config/kpi_scoring.json). File is required.",
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
        default=None,
        help="Override global knee threshold (CHF/year per added kWh). If omitted, uses KPI config/default.",
    )
    parser.add_argument(
        "--seasonal-marginal-gain-threshold",
        type=float,
        default=None,
        help="Override seasonal knee threshold (CHF/season per added kWh). If omitted, uses KPI config/default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kpi_config_path = Path(args.config) if args.config else None
    kpi_config, kpi_config_source = load_kpi_config(kpi_config_path)
    applied_settings = resolve_kpi_settings(
        kpi_config,
        cli_global_knee_threshold=args.marginal_gain_threshold,
        cli_seasonal_knee_threshold=args.seasonal_marginal_gain_threshold,
    )

    sim_paths = [Path(p) for p in args.simulation_jsons] if args.simulation_jsons else sorted(
        Path("out/simulation_json").glob("config_*.json")
    )
    sim_paths = [p for p in sim_paths if p.exists() and p.is_file()]
    if not sim_paths:
        raise FileNotFoundError("No simulation JSON files found to analyze.")

    candidates = [build_candidate(p) for p in sorted(sim_paths)]
    candidates.sort(key=lambda c: (c["battery_size_kwh"], c["scenario"]))
    graph_kpi_threshold = float(
        applied_settings["graph_kpis"]["global_energy_reduction_kwh"]["consumed_reduction_increment_pct_points_min"]
    )
    graph_kpi_bill_offset_threshold = float(
        applied_settings["graph_kpis"]["global_energy_financial_impact_chf"]["bill_offset_increment_pct_points_min"]
    )
    graph_kpi_amortization_years_max = float(
        applied_settings["graph_kpis"]["global_rentability_overview"]["amortization_years_max"]
    )
    graph_kpi_pct_max_cycles_per_year_max = float(
        applied_settings["graph_kpis"]["global_battery_utilization"]["pct_max_cycles_per_year_max"]
    )
    graph_kpi_global_status_empty_pct_max = float(
        applied_settings["graph_kpis"]["global_battery_status_heatmap"]["empty_pct_max"]
    )
    graph_kpi_power_saturation_pct_any_season_max = float(
        applied_settings["graph_kpis"]["seasonal_power_saturation_at_max_limit"]["power_saturation_pct_any_season_max"]
    )
    graph_kpi_monthly_evening_undersize_pct_per_month_max = float(
        applied_settings["graph_kpis"]["monthly_structural_evening_energy_undersizing_peak_period"][
            "evening_undersize_pct_per_month_max"
        ]
    )
    graph_kpi_monthly_evening_max_months_above_threshold = int(
        applied_settings["graph_kpis"]["monthly_structural_evening_energy_undersizing_peak_period"][
            "max_months_above_threshold"
        ]
    )
    global_energy_reduction_consumed_step_delta = compute_global_energy_reduction_consumed_step_delta_kpi(
        candidates,
        threshold_pct_points=graph_kpi_threshold,
    )
    global_energy_financial_impact_bill_offset_step_delta = (
        compute_global_energy_financial_impact_bill_offset_step_delta_kpi(
            candidates,
            threshold_pct_points=graph_kpi_bill_offset_threshold,
        )
    )
    global_rentability_amortization_cap = compute_global_rentability_amortization_cap_kpi(
        candidates,
        amortization_years_max=graph_kpi_amortization_years_max,
    )
    global_battery_utilization_cycle_wear_cap = compute_global_battery_utilization_cycle_wear_cap_kpi(
        candidates,
        pct_max_cycles_per_year_max=graph_kpi_pct_max_cycles_per_year_max,
    )
    global_battery_status_heatmap_empty_cap = compute_global_battery_status_heatmap_empty_cap_kpi(
        candidates,
        empty_pct_max=graph_kpi_global_status_empty_pct_max,
    )
    seasonal_power_saturation_constraint_cap = compute_seasonal_power_saturation_at_max_limit_cap_kpi(
        candidates,
        power_saturation_pct_any_season_max=graph_kpi_power_saturation_pct_any_season_max,
    )
    monthly_structural_evening_undersizing_constraint_cap = (
        compute_monthly_structural_evening_energy_undersizing_peak_period_cap_kpi(
            candidates,
            evening_undersize_pct_per_month_max=graph_kpi_monthly_evening_undersize_pct_per_month_max,
            max_months_above_threshold=graph_kpi_monthly_evening_max_months_above_threshold,
        )
    )

    graph_kpis = {
        "global_energy_reduction_kwh": {
            "consumed_reduction_step_delta_kpi": global_energy_reduction_consumed_step_delta
        },
        "global_energy_financial_impact_chf": {
            "bill_offset_step_delta_kpi": global_energy_financial_impact_bill_offset_step_delta
        },
        "global_rentability_overview": {
            "amortization_cap_kpi": global_rentability_amortization_cap
        },
        "global_battery_utilization": {
            "cycle_wear_cap_kpi": global_battery_utilization_cycle_wear_cap
        },
        "global_battery_status_heatmap": {
            "empty_share_cap_kpi": global_battery_status_heatmap_empty_cap
        },
        "seasonal_power_saturation_at_max_limit": {
            "constraint_cap_kpi": seasonal_power_saturation_constraint_cap
        },
        "monthly_structural_evening_energy_undersizing_peak_period": {
            "constraint_cap_kpi": monthly_structural_evening_undersizing_constraint_cap
        },
    }

    summary = {
        "summary_mode": "graph_kpis_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kpi_config_source": kpi_config_source,
        "kpi_settings": {
            "graph_kpis": applied_settings.get("graph_kpis", {}),
        },
        "source_files": [str(p) for p in sorted(sim_paths)],
        "graph_kpis": graph_kpis,
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
    print(f"KPI config source: {kpi_config_source}")
    print(
        "Graph KPI sections included: "
        f"{sum(1 for group in graph_kpis.values() for value in group.values() if value)}"
    )


if __name__ == "__main__":
    main()
