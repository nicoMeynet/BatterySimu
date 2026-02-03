#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to GitHub-flavored Markdown table."""
    return df.to_markdown(index=False)


def main():
    if len(sys.argv) != 2:
        print("Usage: ./generate_report.py <simulation.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = input_path.with_suffix(".md")

    with open(input_path, "r") as f:
        data = json.load(f)

    # ============================================================
    # TABLE A — GLOBAL SUMMARY
    # ============================================================

    global_range = next(
        r for r in data["simulation"]["ranges"]
        if r["range_type"] == "global"
    )

    rent = global_range["results"]["rentability"]
    battery = global_range["results"]["battery"]
    util = battery["utilization"]
    headroom = battery["headroom"]

    table_global = pd.DataFrame([{
        "Simulation days": round(global_range["range"]["calendar_duration_days"], 1),
        "Annual gain (CHF)": round(rent["annualized_gain_chf"], 2),
        "Total gain (CHF)": round(rent["total_gain_chf"], 2),
        "Average monthly gain (CHF)": round(
            data["simulation"]["seasonal_profitability"]["average_monthly_gain_chf"], 2
        ),
        "Battery cost (CHF)": data["configuration"]["battery"]["cost_chf"],
        "Amortization (years)": round(rent["amortization_years"], 2),
        "Cycles per year": round(util["cycles_per_year"], 1),
        "Cycle budget used per year (%)": round(util["percent_of_max_cycles_per_year"], 2),
        "Expected lifetime (years)": (
            f"{min(util['per_phase']['expected_life_years'].values()):.1f}"
            f" – {max(util['per_phase']['expected_life_years'].values()):.1f}"
        ),
        "Charge headroom (kWh)": round(headroom["charge_headroom_kwh"], 1),
        "Discharge headroom (kWh)": round(headroom["discharge_headroom_kwh"], 1),
    }])

    # ============================================================
    # TABLE B — MONTHLY TABLE
    # ============================================================

    monthly_rows = []

    for m in data["simulation"]["seasonal_profitability"]["monthly_evolution"]:
        monthly_rows.append({
            "Month": m["month_id"],
            "Season": m["season"],
            "Gain (CHF)": round(m["gain_chf"], 2),

            "Grid consumed (kWh)": round(
                m["energy_flows"]["with_battery"]["grid_consumed_kwh"], 1
            ),
            "Grid injected (kWh)": round(
                m["energy_flows"]["with_battery"]["grid_injected_kwh"], 1
            ),
            "Battery charged (kWh)": round(
                m["energy_flows"]["with_battery"]["battery_charged_kwh"], 1
            ),
            "Battery discharged (kWh)": round(
                m["energy_flows"]["with_battery"]["battery_discharged_kwh"], 1
            ),
            "Battery losses (kWh)": round(
                m["energy_flows"]["battery_losses_kwh"], 1
            ),

            "Battery full (%)": round(
                m["battery_saturation"]["battery_full_share_percent"], 1
            ),
            "Battery empty (%)": round(
                m["battery_saturation"]["battery_empty_share_percent"], 1
            ),
            "Charge saturation (%)": round(
                m["battery_saturation"]["charge_power_saturation_percent"], 1
            ),

            "Charge headroom (kWh)": round(
                m["battery_headroom"]["charge_headroom_kwh"], 1
            ),
            "Discharge headroom (kWh)": round(
                m["battery_headroom"]["discharge_headroom_kwh"], 1
            ),

            "Energy undersize (%)": round(
                m["daily_energy_undersize"]["percent"], 1
            ),
            "Evening undersize (%)": round(
                m["daily_evening_undersize"]["percent"], 1
            ),
        })

    table_monthly = (
        pd.DataFrame(monthly_rows)
        .sort_values("Month")
    )

    # ============================================================
    # TABLE C — SEASONAL TABLE
    # ============================================================

    seasonal_rows = []

    for season, s in data["simulation"]["seasonal_profitability"]["seasonal_breakdown"].items():
        seasonal_rows.append({
            "Season": season,
            "Months": s["months_count"],
            "Total gain (CHF)": round(s["total_gain_chf"], 2),
            "Avg monthly gain (CHF)": round(s["average_monthly_gain_chf"], 2),

            "Avg battery full (%)": round(
                s["battery_saturation"]["average_full_share_percent"], 1
            ),
            "Avg battery empty (%)": round(
                s["battery_saturation"]["average_empty_share_percent"], 1
            ),

            "Energy undersize (%)": round(
                s["battery_energy_undersize_days"]["percent"], 1
            ),
            "Evening undersize (%)": round(
                s["battery_evening_undersize_days"]["percent"], 1
            ),

            "Charge headroom (kWh)": round(
                s["battery_headroom"]["charge_headroom_kwh"], 1
            ),
            "Discharge headroom (kWh)": round(
                s["battery_headroom"]["discharge_headroom_kwh"], 1
            ),
        })

    table_seasonal = pd.DataFrame(seasonal_rows)

    # ============================================================
    # MARKDOWN REPORT
    # ============================================================

    md = []
    md.append(f"# Battery Simulation Report\n")
    md.append(f"**Source file:** `{input_path.name}`\n")

    md.append("## Configuration summary\n")
    cfg = data["configuration"]["battery"]
    md.append(
        f"- Battery capacity per phase: {cfg['capacity_Wh_per_phase']} Wh\n"
        f"- Max charge / discharge power: {cfg['max_charge_power_watts']} W\n"
        f"- Charge efficiency: {cfg['charge_efficiency']}\n"
        f"- Discharge efficiency: {cfg['discharge_efficiency']}\n"
        f"- SOC limits: {cfg['soc_min_pct']}% – {cfg['soc_max_pct']}%\n"
        f"- Battery cost: {cfg['cost_chf']} CHF\n"
    )

    md.append("\n## A. Global results\n")
    md.append(md_table(table_global))

    md.append("\n## B. Monthly results\n")
    md.append(md_table(table_monthly))

    md.append("\n## C. Seasonal results\n")
    md.append(md_table(table_seasonal))

    output_path.write_text("\n".join(md), encoding="utf-8")

    print(f"✔ Report generated: {output_path}")


if __name__ == "__main__":
    main()
