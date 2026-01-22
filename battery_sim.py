#!/usr/bin/env python
"""
Battery Simulation Script
This script simulates the behavior of a battery system in a house with three phases (A, B, and C). 
It reads power consumption data from CSV files, processes the data, and simulates the battery's 
charging and discharging behavior based on the house's power consumption and predefined battery 
parameters. The script also calculates the financial impact of using the battery system.
"""
import argparse
import pandas as pd
import sys
from tabulate import tabulate
import json
import hashlib
from datetime import datetime, UTC

###################################################################
# CONFIGURATION
###################################################################
PHASE_KEYS = ["A", "B", "C"]
# ---- Battery parameters ----
#battery_capacity_Wh = [2880, 2880, 2880]        # Battery capacity per phase (Wh)
#battery_cost = 3270                             # Battery cost (CHF)
battery_capacity_Wh = [5760, 5760, 5760]        # Battery capacity per phase (Wh)
battery_cost = 4800                             # Battery cost (CHF)
#battery_capacity_Wh = [8640, 8640, 8640]        # Battery capacity per phase (Wh)
#battery_cost = 7500                             # Battery cost (CHF)
#battery_capacity_Wh = [11520, 11520, 11520]     # Battery capacity per phase (Wh)
#battery_cost = 9600                             # Battery cost (CHF)
max_charge_power_watts = [2400, 2400, 2400]     # Max charge power per phase (W)
max_discharge_power_watts = [2400, 2400, 2400]  # Max discharge power per phase (W)
battery_charge_efficiency = 0.9                 # Charge efficiency (90%)
battery_discharge_efficiency = 0.9              # Discharge efficiency (90%)
battery_max_cycles = 6000                       # Battery lifespan in cycles
battery_soc = [100, 100, 100]                   # Initial state of charge in % (when the simulation starts)

# ---- Electricity tariff configuration ----
tariff_config = {
    "peak": {
        "tariff_consume": 0.34,      # CHF/kWh
        "tariff_inject": 0.06,       # CHF/kWh
        "days": [0, 1, 2, 3, 4],     # 0:Monday to 4:Friday
        "hours": range(17, 22)       # 5 PM to 10 PM
    },
    "off_peak": {
        "tariff_consume": 0.34,      # CHF/kWh for the rest of the time
        "tariff_inject": 0.06        # CHF/kWh for the rest of the time
    }
}

###################################################################
# FUNCTIONS
###################################################################
# ---- Function to determine the current tariff mode ----
# HP = Peak Hours
# HC = Off-Peak Hours
def get_current_tariff_mode(hour, day):
    if day in tariff_config["peak"]["days"] and hour in tariff_config["peak"]["hours"]:
        return "HP" # Peak hours
    return "HC" # Off-peak hours

# ---- Function to determine the current tariff price ----
def get_current_tariff_price(tariff_mode, action):
    if tariff_mode == "HP":
        if action == "consume":
            return tariff_config["peak"]["tariff_consume"]
        elif action == "inject":
            return tariff_config["peak"]["tariff_inject"]
    else:
        if action == "consume":
            return tariff_config["off_peak"]["tariff_consume"]
        elif action == "inject":
            return tariff_config["off_peak"]["tariff_inject"]
    return 0

# ---- Function to calculate energy in Wh ----
def calculate_energy_Wh(power_watts, duration_minutes):
    return power_watts * duration_minutes / 60

# ---- Update the consumed or injected energy for a given phase and tariff mode ----
def update_energy(energy, tariff_mode, consumed_dict, injected_dict, phase):
    if energy < 0:
        injected_dict[phase][tariff_mode] += abs(energy)
    else:
        consumed_dict[phase][tariff_mode] += energy

# ---- Simulation ----
def simulate_battery_behavior(house_grid_power_watts):
    global energy_in_battery_Wh 

    # Calculate production and consumption per phase
    new_house_grid_power_watts = [0, 0, 0]
    battery_status = [0, 0, 0]
    battery_cycle = [0, 0, 0]
    charge_power_watts = [0, 0, 0]
    discharge_power_watts = [0, 0, 0]
    for phase in range(3):
        if house_grid_power_watts[phase] < 0:
            # Inject into the grid, charge the battery with the surplus
            surplus_watts = abs(house_grid_power_watts[phase])
            if energy_in_battery_Wh[phase] < battery_capacity_Wh[phase]:
                # Charge the battery if not full
                charge_power_watts[phase] = min(surplus_watts, max_charge_power_watts[phase])
                energy_to_be_injected_in_battery_Wh = calculate_energy_Wh(charge_power_watts[phase], 1) * battery_charge_efficiency
                energy_in_battery_Wh[phase] += energy_to_be_injected_in_battery_Wh
                energy_in_battery_Wh[phase] = min(energy_in_battery_Wh[phase], battery_capacity_Wh[phase])
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase] + charge_power_watts[phase]
                battery_cycle[phase] = abs(energy_to_be_injected_in_battery_Wh / battery_capacity_Wh[phase] / 2)
                battery_status[phase] = "charging"
            else:
                # Battery full
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase]
                battery_status[phase] = "full"
        else:
            # Consume from the grid, discharge the battery to compensate
            deficit_watts = house_grid_power_watts[phase]
            discharge_power_watts[phase] = min(deficit_watts, max_discharge_power_watts[phase])
            if energy_in_battery_Wh[phase] > 0:
                # Discharge the battery if not empty
                energy_to_be_consumed_from_battery_Wh = calculate_energy_Wh(discharge_power_watts[phase], 1) * (1 / battery_discharge_efficiency)
                energy_in_battery_Wh[phase] -= energy_to_be_consumed_from_battery_Wh
                energy_in_battery_Wh[phase] = max(0, energy_in_battery_Wh[phase])
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase] - discharge_power_watts[phase]
                battery_cycle[phase] = abs(energy_to_be_consumed_from_battery_Wh / battery_capacity_Wh[phase] / 2)
                battery_status[phase] = "discharging"
            else:
                # No energy in the battery
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase]
                battery_status[phase] = "empty"

    return {
        "A": {
            "new_house_grid_power_watts": new_house_grid_power_watts[0],
            "energy_in_battery_Wh": energy_in_battery_Wh[0],
            "battery_status": battery_status[0],
            "battery_cycle": battery_cycle[0],
            "charge_power_watts": charge_power_watts[0],
            "discharge_power_watts": discharge_power_watts[0]
        },
        "B": {
            "new_house_grid_power_watts": new_house_grid_power_watts[1],
            "energy_in_battery_Wh": energy_in_battery_Wh[1],
            "battery_status": battery_status[1],
            "battery_cycle": battery_cycle[1],
            "charge_power_watts": charge_power_watts[1],
            "discharge_power_watts": discharge_power_watts[1]
        },
        "C": {
            "new_house_grid_power_watts": new_house_grid_power_watts[2],
            "energy_in_battery_Wh": energy_in_battery_Wh[2],
            "battery_status": battery_status[2],
            "battery_cycle": battery_cycle[2],
            "charge_power_watts": charge_power_watts[2],
            "discharge_power_watts": discharge_power_watts[2]
        }
    }

def build_configuration_section():
    return {
        "battery": {
            "capacity_Wh_per_phase": battery_capacity_Wh,
            "cost_chf": battery_cost,
            "max_charge_power_W_per_phase": max_charge_power_watts,
            "max_discharge_power_W_per_phase": max_discharge_power_watts,
            "charge_efficiency": battery_charge_efficiency,
            "discharge_efficiency": battery_discharge_efficiency,
            "max_cycles": battery_max_cycles,
            "initial_soc_percent_per_phase": battery_soc
        },
        "tariff": {
            "peak": {
                "tariff_consume_chf_per_kwh": tariff_config["peak"]["tariff_consume"],
                "tariff_inject_chf_per_kwh": tariff_config["peak"]["tariff_inject"],
                "weekdays": tariff_config["peak"]["days"],
                "hours": list(tariff_config["peak"]["hours"])
            },
            "off_peak": {
                "tariff_consume_chf_per_kwh": tariff_config["off_peak"]["tariff_consume"],
                "tariff_inject_chf_per_kwh": tariff_config["off_peak"]["tariff_inject"]
            }
        }
    }

def compute_configuration_hash(configuration: dict) -> str:
    """
    Compute a deterministic SHA-256 hash of the configuration.

    - Keys are sorted
    - No whitespace differences
    - Stable across runs
    """
    canonical_json = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":")
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

def build_json_report(simulation_ranges):
    configuration = build_configuration_section()
    configuration_hash = compute_configuration_hash(configuration)

    return {
        "schema_version": "1.4",
        "simulation_id": datetime.now(UTC).isoformat(),
        "phase_convention": {
            "type": "electrical",
            "values": PHASE_KEYS
        },

        "configuration": configuration,
        "configuration_hash": configuration_hash,

        "simulation": {
            "ranges": simulation_ranges
        }
    }

def export_json_report(report, filename="simulation_report.json"):
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"+ JSON report exported to {filename}")

def build_range_metadata(df, duration_days):
    return {
        "start_date": df["timestamp"].min().isoformat(),
        "end_date": df["timestamp"].max().isoformat(),
        "number_of_points": len(df),
        "duration_days": duration_days
    }

def build_battery_statistics(
    df,
    *,
    cycles=None,
    battery_max_cycles=None
):
    return {
        "measured_cycles": cycles,
        "cycle_definition": "1.0 = full charge + discharge",
        "cycles_scope": "global_only" if cycles is not None else "not_applicable",
        "max_cycles": battery_max_cycles,
        "remaining_energy_Wh": {
            phase: r(df[f"battery_energy_phase_{phase}_Wh"].iloc[-1], 1)
            for phase in PHASE_KEYS
        }
    }

def compute_battery_status(df):
    def stats(col):
        total = len(df)
        return {
            status: {
                "count": int((df[col] == status).sum()),
                "percent": round((df[col] == status).sum() / total * 100, 2)
            }
            for status in ["full", "empty", "charging", "discharging"]
        }

    return {
        phase: stats(f"battery_status_phase_{phase}")
        for phase in PHASE_KEYS
    } | {
        "total_samples": len(df)
    }

def compute_power_at_peak(df, max_charge_power_watts, max_discharge_power_watts):
    def power_stats(col, max_power):
        total = len(df[df[col] != 0])
        if total == 0:
            return {
                "data_available": False,
                "at_max": {"count": 0, "percent": 0.0},
                "not_at_max": {"count": 0, "percent": 0.0}
            }

        at_max = (df[col] == max_power).sum()
        return {
            "data_available": True,
            "at_max": {
                "count": int(at_max),
                "percent": round(at_max / total * 100, 2)
            },
            "not_at_max": {
                "count": int(total - at_max),
                "percent": round((total - at_max) / total * 100, 2)
            }
        }

    return {
        "charging": {
            phase: power_stats(
                f"charge_power_phase_{phase}_W",
                max_charge_power_watts[i]
            )
            for i, phase in enumerate(PHASE_KEYS)
        },
        "discharging": {
            phase: power_stats(
                f"discharge_power_phase_{phase}_W",
                max_discharge_power_watts[i]
            )
            for i, phase in enumerate(PHASE_KEYS)
        }
    }

def build_canonical_results(
    df,
    battery_cost,
    duration_days,
    *,
    cycles=None
):
    """
    Canonical results builder used for ALL ranges (global & monthly)

    - Global:
        * cycles provided
        * amortization computed
        * note = "Global range"
    - Monthly:
        * cycles = None
        * amortization disabled
        * note = "Delta vs without battery"
    """

    energy_rent = compute_energy_and_rentability_from_df(df)

    total_gain = energy_rent["rentability"]["total_gain_chf"]

    annualized_gain = (
        r(total_gain / duration_days * 365)
        if cycles is not None and duration_days >= 7
        else None
    )

    amortization_years = (
        r(battery_cost / annualized_gain)
        if cycles is not None
        and annualized_gain is not None
        and annualized_gain > 0
        else None
    )

    return {
        "energy": energy_rent["energy"],

        "rentability": {
            "total_gain_chf": energy_rent["rentability"]["total_gain_chf"],
            "annualized_gain_chf": annualized_gain,
            "annualization_method": "linear_extrapolation",
            "amortization_years": amortization_years,
            "profitable": energy_rent["rentability"]["total_gain_chf"] > 0,
            "note": (
                "Global range (amortization enabled)"
                if cycles is not None
                else "Delta vs without battery (monthly slice)"
            )
        },

        "battery": {
            "statistics": build_battery_statistics(
                df,
                cycles=cycles,
                battery_max_cycles=battery_max_cycles
            ),
            "utilization": compute_battery_utilization(
                cycles,
                duration_days,
                battery_max_cycles
            ),
            "status": compute_battery_status(df),
            "power_at_peak": compute_power_at_peak(
                df,
                max_charge_power_watts,
                max_discharge_power_watts
            )
        }
    }

def compute_energy_and_rentability_from_df(df):
    """
    Compute DELTA injected / consumed energy and CHF gain
    (with battery vs without battery)
    """

    # Wh accumulators
    injected_without = {"A": {"HP": 0, "HC": 0}, "B": {"HP": 0, "HC": 0}, "C": {"HP": 0, "HC": 0}}
    injected_with    = {"A": {"HP": 0, "HC": 0}, "B": {"HP": 0, "HC": 0}, "C": {"HP": 0, "HC": 0}}
    consumed_without = {"A": {"HP": 0, "HC": 0}, "B": {"HP": 0, "HC": 0}, "C": {"HP": 0, "HC": 0}}
    consumed_with    = {"A": {"HP": 0, "HC": 0}, "B": {"HP": 0, "HC": 0}, "C": {"HP": 0, "HC": 0}}

    for _, row in df.iterrows():
        tarif = row["tariff_mode"]

        for phase in PHASE_KEYS:
            p = phase.lower()

            no_batt = row[f"house_consumption_phase_{p}_Wh"]
            with_batt = row[f"simulated_house_consumption_phase_{p}_Wh"]

            if no_batt < 0:
                injected_without[phase][tarif] += abs(no_batt)
            else:
                consumed_without[phase][tarif] += no_batt

            if with_batt < 0:
                injected_with[phase][tarif] += abs(with_batt)
            else:
                consumed_with[phase][tarif] += with_batt

    def wh_to_kwh(x):
        return round(x / 1000, 3)

    injected_table = []
    consumed_table = []
    total_gain_chf = 0.0

    for phase in PHASE_KEYS:
        for tarif in ["HC", "HP"]:
            inj_delta_kwh = wh_to_kwh(injected_with[phase][tarif] - injected_without[phase][tarif])
            con_delta_kwh = wh_to_kwh(consumed_with[phase][tarif] - consumed_without[phase][tarif])

            inj_chf = get_current_tariff_price(tarif, "inject") * inj_delta_kwh
            con_chf = get_current_tariff_price(tarif, "consume") * con_delta_kwh

            injected_table.append({
                "phase": phase.upper(),
                "tariff": tarif,
                "energy_kwh": inj_delta_kwh,
                "delta_chf": r(inj_chf)
            })

            consumed_table.append({
                "phase": phase.upper(),
                "tariff": tarif,
                "energy_kwh": con_delta_kwh,
                "delta_chf": r(con_chf)
            })

            total_gain_chf += inj_chf - con_chf

    return {
        "energy": {
            "unit": "kWh",
            "sign_convention": "negative = reduction vs without battery",
            "injected": injected_table,
            "consumed": consumed_table
        },
        "rentability": {
            "total_gain_chf": r(total_gain_chf),
            "annualized_gain_chf": None,
            "amortization_years": None,
            "note": "Delta vs without battery"
        }
    }

def r(x, digits=2):
    v = round(x, digits)
    return 0.0 if abs(v) < 1e-9 else v

def print_progress_bar(current, total, prefix="Progress"):
    percent = current / total * 100
    bar_width = 40
    filled = int(percent / 100 * bar_width)
    sys.stdout.write(
        f"\r+ {prefix}: [{'#' * filled}{' ' * (bar_width - filled)}] {percent:.2f}%"
    )
    sys.stdout.flush()

def compute_battery_utilization(cycles, duration_days, max_cycles):
    if cycles is None or duration_days <= 0:
        return None

    # cycles/year per phase
    cycles_per_year = {
        phase: r(c / duration_days * 365, 1) if c is not None else None
        for phase, c in cycles.items()
    }

    # expected life per phase (years)
    expected_life_years = {
        phase: (r(max_cycles / cpy, 1) if cpy and cpy > 0 else None)
        for phase, cpy in cycles_per_year.items()
    }

    # average across phases (what you currently report)
    cpy_values = [v for v in cycles_per_year.values() if v is not None]
    avg_cpy = sum(cpy_values) / len(cpy_values) if cpy_values else None

    return {
        "cycles_per_year": r(avg_cpy, 1) if avg_cpy is not None else None,
        "percent_of_max_cycles_per_year": r((avg_cpy / max_cycles) * 100, 2) if avg_cpy else None,

        # ✅ new: per-phase details
        "per_phase": {
            "cycles_per_year": cycles_per_year,
            "percent_of_max_cycles_per_year": {
                phase: (r((cpy / max_cycles) * 100, 2) if cpy else None)
                for phase, cpy in cycles_per_year.items()
            },
            "expected_life_years": expected_life_years
        }
    }

def compute_expected_life_years_per_phase(cycles: dict, max_cycles: float):
    """
    Estimate battery life per phase (years) assuming max_cycles is EOL and
    future usage is similar to measured cycles/year.
    """
    out = {}
    for phase, c in cycles.items():
        if c is None:
            out[phase] = None
            continue
        # c is cycles over the simulated duration
        # We'll convert to cycles/year when building results (needs duration_days),
        # so here we just keep structure; real calc done below.
        out[phase] = None
    return out

def empty_energy_dict():
    return {p: {"HP": 0, "HC": 0} for p in PHASE_KEYS}

###################################################################
# MAIN
###################################################################
parser = argparse.ArgumentParser(
    description="Battery simulation with optional exports"
)

# Positional inputs
parser.add_argument("house_phase_a", help="CSV file for phase A")
parser.add_argument("house_phase_b", help="CSV file for phase B")
parser.add_argument("house_phase_c", help="CSV file for phase C")

# Optional exports
parser.add_argument(
    "--export-csv",
    metavar="FILE",
    help="Export per-timestep simulation results to CSV"
)

parser.add_argument(
    "--export-json",
    metavar="FILE",
    help="Export full aggregated simulation report to JSON"
)

args = parser.parse_args()

house_phase_a_file = args.house_phase_a
house_phase_b_file = args.house_phase_b
house_phase_c_file = args.house_phase_c


# Initial battery state
energy_in_battery_Wh = [battery_capacity_Wh[i] * (battery_soc[i] / 100) for i in range(3)]
print("Initial battery state:")
print(f"+ phase A: {energy_in_battery_Wh[0]} Wh")
print(f"+ phase B: {energy_in_battery_Wh[1]} Wh")
print(f"+ phase C: {energy_in_battery_Wh[2]} Wh")

# Loading CSV files
print("Loading CSV files")
house_phase_a = pd.read_csv(house_phase_a_file, parse_dates=["last_changed"])
house_phase_b = pd.read_csv(house_phase_b_file, parse_dates=["last_changed"])
house_phase_c = pd.read_csv(house_phase_c_file, parse_dates=["last_changed"])
print(f"+ Successfully loaded {house_phase_a_file} {house_phase_b_file} {house_phase_c_file}")

# Data formatting (cleaning, transformation, etc.)
print("Formatting data")
house_phase_a.drop(columns=["entity_id"], inplace=True)
house_phase_a.rename(columns={"last_changed": "timestamp", "state": "phase_a"}, inplace=True)
house_phase_b.drop(columns=["entity_id"], inplace=True)
house_phase_b.rename(columns={"last_changed": "timestamp", "state": "phase_b"}, inplace=True)
house_phase_c.drop(columns=["entity_id"], inplace=True)
house_phase_c.rename(columns={"last_changed": "timestamp", "state": "phase_c"}, inplace=True)
print(f"+ Successfully formatted")

# Display information about the loaded data
print("Data information")
house_phase_a_start_timestamp = house_phase_a["timestamp"].min()
house_phase_a_end_timestamp = house_phase_a["timestamp"].max()
house_phase_a_data_quantity = len(house_phase_a)
print(f"+ Phase A - Timestamp: {house_phase_a_start_timestamp} to {house_phase_a_end_timestamp} with {house_phase_a_data_quantity} lines (number of days: {(house_phase_a_end_timestamp - house_phase_a_start_timestamp).days} days)")
house_phase_b_start_timestamp = house_phase_b["timestamp"].min()
house_phase_b_end_timestamp = house_phase_b["timestamp"].max()
house_phase_b_data_quantity = len(house_phase_b)
print(f"+ Phase B - Timestamp: {house_phase_b_start_timestamp} to {house_phase_b_end_timestamp} with {house_phase_b_data_quantity} lines (number of days: {(house_phase_b_end_timestamp - house_phase_b_start_timestamp).days} days)")
house_phase_c_start_timestamp = house_phase_c["timestamp"].min()
house_phase_c_end_timestamp = house_phase_c["timestamp"].max()
house_phase_c_data_quantity = len(house_phase_c)
print(f"+ Phase C - Timestamp: {house_phase_c_start_timestamp} to {house_phase_c_end_timestamp} with {house_phase_c_data_quantity} lines (number of days: {(house_phase_c_end_timestamp - house_phase_c_start_timestamp).days} days)")

# Round timestamps to the nearest minute
print("Rounding timestamps to the nearest minute")
house_phase_a["timestamp"] = house_phase_a["timestamp"].dt.floor("min")
house_phase_b["timestamp"] = house_phase_b["timestamp"].dt.floor("min")
house_phase_c["timestamp"] = house_phase_c["timestamp"].dt.floor("min")
print("+ Successfully rounded timestamps")

# Sort columns to have 'timestamp' first
print("Sorting columns to have 'timestamp' first")
columns_order = ["timestamp"] + [col for col in house_phase_a.columns if col != "timestamp"]
house_phase_a = house_phase_a[columns_order]
columns_order = ["timestamp"] + [col for col in house_phase_b.columns if col != "timestamp"]
house_phase_b = house_phase_b[columns_order]
columns_order = ["timestamp"] + [col for col in house_phase_c.columns if col != "timestamp"]
house_phase_c = house_phase_c[columns_order]
print("+ Successfully sorted columns")

# Group by timestamp and calculate the mean of the phase column
print("Grouping data by timestamp")
house_phase_a["phase_a"] = pd.to_numeric(house_phase_a["phase_a"], errors="coerce")
house_phase_a = house_phase_a.dropna(subset=["phase_a"])
house_phase_a = house_phase_a.groupby("timestamp").agg({"phase_a": "mean"}).reset_index()
house_phase_b["phase_b"] = pd.to_numeric(house_phase_b["phase_b"], errors="coerce")
house_phase_b = house_phase_b.dropna(subset=["phase_b"])
house_phase_b = house_phase_b.groupby("timestamp").agg({"phase_b": "mean"}).reset_index()
house_phase_c["phase_c"] = pd.to_numeric(house_phase_c["phase_c"], errors="coerce")
house_phase_c = house_phase_c.dropna(subset=["phase_c"])
house_phase_c = house_phase_c.groupby("timestamp").agg({"phase_c": "mean"}).reset_index()
print("+ Successfully grouped data by timestamp")

# Fill missing timestamps in phase A with an average value between the values before and after the missing timestamp
print("Filling missing timestamps with an average value between the values before and after the missing timestamp:")
all_timestamps = pd.date_range(start=house_phase_a_start_timestamp, end=house_phase_a_end_timestamp, freq='min')
house_phase_a = house_phase_a.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_a["phase_a"] = house_phase_a["phase_a"].interpolate(method='linear').apply(lambda x: int(x))
print("+ Successfully filled missing timestamps in phase A")

# Fill missing timestamps in phase B with an average value between the values before and after the missing timestamp
all_timestamps = pd.date_range(start=house_phase_b_start_timestamp, end=house_phase_b_end_timestamp, freq='min')
house_phase_b = house_phase_b.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_b["phase_b"] = house_phase_b["phase_b"].interpolate(method='linear').apply(lambda x: int(x))
print("+ Successfully filled missing timestamps in phase B")

# Fill missing timestamps in phase C with an average value between the values before and after the missing timestamp
all_timestamps = pd.date_range(start=house_phase_c_start_timestamp, end=house_phase_c_end_timestamp, freq='min')
house_phase_c = house_phase_c.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_c["phase_c"] = house_phase_c["phase_c"].interpolate(method='linear').apply(lambda x: int(x))
print("+ Successfully filled missing timestamps in phase C")

# Merge the 4 datasets based on the timestamp, filling missing values with 0
print("Merging data into a single table")
merged_data = house_phase_a.merge(house_phase_b, on="timestamp", how="outer")
merged_data = merged_data.merge(house_phase_c, on="timestamp", how="outer")
print("+ Successfully merged data")

# Fill missing values with 0
print("Filling missing values with 0")
merged_data = merged_data.fillna(0)
merged_data = merged_data.infer_objects(copy=False)
print("+ Successfully filled missing values with 0")

# Display information about the merged data
print("Information about the merged data")
merged_start_timestamp = merged_data["timestamp"].min()
merged_end_timestamp = merged_data["timestamp"].max()
merged_data_quantity = len(merged_data)
print(f"+ Merged - Timestamp: {merged_start_timestamp} to {merged_end_timestamp} with {merged_data_quantity} lines")

# Number of days to process
duration_days_global = max(
    1,
    (merged_end_timestamp - merged_start_timestamp).days
)

print("Starting simulation...")
# ---- Execution on the data from the CSV files ----
results = []
battery_cycle_total = {p: 0.0 for p in PHASE_KEYS}

# Initialize dictionaries to store injected and consumed energy for each phase and tariff mode
simulated_injected_energy_Wh = empty_energy_dict()
simulated_consumed_energy_Wh = empty_energy_dict()
current_injected_energy_Wh = empty_energy_dict()
current_consumed_energy_Wh = empty_energy_dict()

total_steps = len(merged_data)

for i in range(total_steps):
    print_progress_bar(i + 1, total_steps, prefix="Progress")

    # Get the current timestamp, weekday and hour
    timestamp = merged_data.iloc[i]["timestamp"]
    weekday = timestamp.weekday()
    hour = timestamp.hour

    # Get the current tarif mode
    tariff_mode = get_current_tariff_mode(hour, weekday)

    # Statistics without the battery
    house_consumption_watts = [merged_data.iloc[i][f"phase_{phase.lower()}"] for phase in PHASE_KEYS]
    # - House consumption
    energy_house_consumption_Wh_phase_a=calculate_energy_Wh(house_consumption_watts[0],1)
    energy_house_consumption_Wh_phase_b=calculate_energy_Wh(house_consumption_watts[1],1)
    energy_house_consumption_Wh_phase_c=calculate_energy_Wh(house_consumption_watts[2],1)

    # Simulation with the battery
    sim_result = simulate_battery_behavior(house_consumption_watts)

    # - House consumption
    simulated_energy_house_consumption_Wh_phase_a=calculate_energy_Wh(sim_result["A"]["new_house_grid_power_watts"],1)
    simulated_energy_house_consumption_Wh_phase_b=calculate_energy_Wh(sim_result["B"]["new_house_grid_power_watts"],1)
    simulated_energy_house_consumption_Wh_phase_c=calculate_energy_Wh(sim_result["C"]["new_house_grid_power_watts"],1)

    # - Battery energy
    battery_cycle_total["A"] += sim_result["A"]["battery_cycle"]
    battery_cycle_total["B"] += sim_result["B"]["battery_cycle"]
    battery_cycle_total["C"] += sim_result["C"]["battery_cycle"]

    # List of phases and tariff modes
    phases = PHASE_KEYS
    tariff_modes = ["HP", "HC"]

    # Process real energy consumption
    for phase, energy in zip(phases, [
        energy_house_consumption_Wh_phase_a,
        energy_house_consumption_Wh_phase_b,
        energy_house_consumption_Wh_phase_c
    ]):
        update_energy(energy, tariff_mode, current_consumed_energy_Wh, current_injected_energy_Wh, phase)

    # Process simulated energy consumption
    for phase, energy in zip(phases, [
        simulated_energy_house_consumption_Wh_phase_a,
        simulated_energy_house_consumption_Wh_phase_b,
        simulated_energy_house_consumption_Wh_phase_c
    ]):
        update_energy(energy, tariff_mode, simulated_consumed_energy_Wh, simulated_injected_energy_Wh, phase)

    # Export all info in a csv file
    results.append({
        "timestamp": timestamp,
        "tariff_mode": tariff_mode,
        "house_consumption_phase_a_Wh": energy_house_consumption_Wh_phase_a,
        "house_consumption_phase_b_Wh": energy_house_consumption_Wh_phase_b,
        "house_consumption_phase_c_Wh": energy_house_consumption_Wh_phase_c,
        "simulated_house_consumption_phase_a_Wh": simulated_energy_house_consumption_Wh_phase_a,
        "simulated_house_consumption_phase_b_Wh": simulated_energy_house_consumption_Wh_phase_b,
        "simulated_house_consumption_phase_c_Wh": simulated_energy_house_consumption_Wh_phase_c,
        "battery_energy_phase_A_Wh": sim_result["A"]["energy_in_battery_Wh"],
        "battery_energy_phase_B_Wh": sim_result["B"]["energy_in_battery_Wh"],
        "battery_energy_phase_C_Wh": sim_result["C"]["energy_in_battery_Wh"],
        "battery_status_phase_A": sim_result["A"]["battery_status"],
        "battery_status_phase_B": sim_result["B"]["battery_status"],
        "battery_status_phase_C": sim_result["C"]["battery_status"],
        "charge_power_phase_A_W": sim_result["A"]["charge_power_watts"],
        "charge_power_phase_B_W": sim_result["B"]["charge_power_watts"],
        "charge_power_phase_C_W": sim_result["C"]["charge_power_watts"],
        "discharge_power_phase_A_W": sim_result["A"]["discharge_power_watts"],
        "discharge_power_phase_B_W": sim_result["B"]["discharge_power_watts"],
        "discharge_power_phase_C_W": sim_result["C"]["discharge_power_watts"]
    })

print("\n+ Successfully simulated battery behavior")

# AUTHORITATIVE BATTERY CYCLES (GLOBAL)
battery_cycles_global = {
    "A": r(battery_cycle_total["A"]),
    "B": r(battery_cycle_total["B"]),
    "C": r(battery_cycle_total["C"])
}

# Build results DataFrame
results_df = pd.DataFrame(results)

print("**********************************")
print("******* Simulation results *******")
print("**********************************")
# Data for injected energy
data_injected = [
    ["Phase A Injected Off-Peak", int(current_injected_energy_Wh['A']['HC'] / 1000), int(simulated_injected_energy_Wh['A']['HC'] / 1000), int((simulated_injected_energy_Wh['A']['HC'] - current_injected_energy_Wh['A']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['A']['HC'] - current_injected_energy_Wh['A']['HC']) / 1000 * (-1))],
    ["Phase A Injected Peak", int(current_injected_energy_Wh['A']['HP'] / 1000), int(simulated_injected_energy_Wh['A']['HP'] / 1000), int((simulated_injected_energy_Wh['A']['HP'] - current_injected_energy_Wh['A']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['A']['HP'] - current_injected_energy_Wh['A']['HP']) / 1000 * (-1))],
    ["Phase B Injected Off-Peak", int(current_injected_energy_Wh['B']['HC'] / 1000), int(simulated_injected_energy_Wh['B']['HC'] / 1000), int((simulated_injected_energy_Wh['B']['HC'] - current_injected_energy_Wh['B']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['B']['HC'] - current_injected_energy_Wh['B']['HC']) / 1000 * (-1))],
    ["Phase B Injected Peak", int(current_injected_energy_Wh['B']['HP'] / 1000), int(simulated_injected_energy_Wh['B']['HP'] / 1000), int((simulated_injected_energy_Wh['B']['HP'] - current_injected_energy_Wh['B']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['B']['HP'] - current_injected_energy_Wh['B']['HP']) / 1000 * (-1))],
    ["Phase C Injected Off-Peak", int(current_injected_energy_Wh['C']['HC'] / 1000), int(simulated_injected_energy_Wh['C']['HC'] / 1000), int((simulated_injected_energy_Wh['C']['HC'] - current_injected_energy_Wh['C']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['C']['HC'] - current_injected_energy_Wh['C']['HC']) / 1000 * (-1))],
    ["Phase C Injected Peak", int(current_injected_energy_Wh['C']['HP'] / 1000), int(simulated_injected_energy_Wh['C']['HP'] / 1000), int((simulated_injected_energy_Wh['C']['HP'] - current_injected_energy_Wh['C']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['C']['HP'] - current_injected_energy_Wh['C']['HP']) / 1000 * (-1))]
]

# Data for consumed energy
data_consumed = [
    ["Phase A Consumed Off-Peak", int(current_consumed_energy_Wh['A']['HC'] / 1000), int(simulated_consumed_energy_Wh['A']['HC'] / 1000), int((simulated_consumed_energy_Wh['A']['HC'] - current_consumed_energy_Wh['A']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['A']['HC'] - current_consumed_energy_Wh['A']['HC']) / 1000)],
    ["Phase A Consumed Peak", int(current_consumed_energy_Wh['A']['HP'] / 1000), int(simulated_consumed_energy_Wh['A']['HP'] / 1000), int((simulated_consumed_energy_Wh['A']['HP'] - current_consumed_energy_Wh['A']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['A']['HP'] - current_consumed_energy_Wh['A']['HP']) / 1000)],
    ["Phase B Consumed Off-Peak", int(current_consumed_energy_Wh['B']['HC'] / 1000), int(simulated_consumed_energy_Wh['B']['HC'] / 1000), int((simulated_consumed_energy_Wh['B']['HC'] - current_consumed_energy_Wh['B']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['B']['HC'] - current_consumed_energy_Wh['B']['HC']) / 1000)],
    ["Phase B Consumed Peak", int(current_consumed_energy_Wh['B']['HP'] / 1000), int(simulated_consumed_energy_Wh['B']['HP'] / 1000), int((simulated_consumed_energy_Wh['B']['HP'] - current_consumed_energy_Wh['B']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['B']['HP'] - current_consumed_energy_Wh['B']['HP']) / 1000)],
    ["Phase C Consumed Off-Peak", int(current_consumed_energy_Wh['C']['HC'] / 1000), int(simulated_consumed_energy_Wh['C']['HC'] / 1000), int((simulated_consumed_energy_Wh['C']['HC'] - current_consumed_energy_Wh['C']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['C']['HC'] - current_consumed_energy_Wh['C']['HC']) / 1000)],
    ["Phase C Consumed Peak", int(current_consumed_energy_Wh['C']['HP'] / 1000), int(simulated_consumed_energy_Wh['C']['HP'] / 1000), int((simulated_consumed_energy_Wh['C']['HP'] - current_consumed_energy_Wh['C']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['C']['HP'] - current_consumed_energy_Wh['C']['HP']) / 1000)]
]

# Calculate totals for each column
totals_injected = ["Total Injected", sum(row[1] for row in data_injected), sum(row[2] for row in data_injected), sum(row[3] for row in data_injected), sum(row[4] for row in data_injected)]
totals_consumed = ["Total Consumed", sum(row[1] for row in data_consumed), sum(row[2] for row in data_consumed), sum(row[3] for row in data_consumed), sum(row[4] for row in data_consumed)]
total_gain_CHF = totals_injected[4] + totals_consumed[4]

headers = ["Phase", "Without Battery (kWh)", "With Battery (kWh)", "Delta (kWh)", "Delta (CHF)"]

print("Injected Energy:")
print("The table below shows the energy injected into the grid.") 
print("Adding a battery will reduce the amount of energy injected into the grid, as surplus energy will be stored in the battery rather than returned to the grid.")
print(tabulate(data_injected + [totals_injected], headers, tablefmt="grid"))

print("")
print("Consumed Energy:")
print("The table below shows the energy consumed from the grid.")
print("Adding a battery is expected to reduce the amount of energy consumed from the grid.")
print(tabulate(data_consumed + [totals_consumed], headers, tablefmt="grid"))

print("")
print("Rentability:")
print(f"+ Total gain: {total_gain_CHF} CHF for {duration_days_global} days or per year: {total_gain_CHF / duration_days_global * 365:.0f} CHF (extrapolated)")
if total_gain_CHF > 0:
    print(
        f"+ Amortization time: "
        f"{battery_cost / total_gain_CHF * duration_days_global / 365:.2f} years "
        f"if the cost of the battery is {battery_cost} CHF"
    )
else:
    print("+ Amortization time: not applicable (no financial gain)")


print("")
# Battery statistics
def expected_life(cycles):
    return (
        int(battery_max_cycles / cycles * duration_days_global / 365)
        if cycles > 0
        else None
    )

battery_stats_data = [
    ["Cycles",
     int(battery_cycle_total['A']),
     int(battery_cycle_total['B']),
     int(battery_cycle_total['C']),
     battery_max_cycles
    ],
    ["Expected life (years)",
     expected_life(battery_cycle_total['A']),
     expected_life(battery_cycle_total['B']),
     expected_life(battery_cycle_total['C']),
     ""
    ],
    ["Remaining energy (Wh)",
     int(energy_in_battery_Wh[0]),
     int(energy_in_battery_Wh[1]),
     int(energy_in_battery_Wh[2]),
     ""
    ]
]

headers = ["Metric", "Phase 1", "Phase 2", "Phase 3", "Max/Config"]

print("Battery Statistics:")
print("This is the statistics for the battery indicating the number of cycles, the expected life, and the remaining energy.")
print("If the cycles between the phases are different, it means that the battery is not used equally.")
print(tabulate(battery_stats_data, headers, tablefmt="grid"))


# Calculate the number of entries in results with battery status "full" or "empty"
battery_status_full_phase_A = results_df[results_df["battery_status_phase_A"] == "full"].shape[0]
battery_status_full_phase_B = results_df[results_df["battery_status_phase_B"] == "full"].shape[0]
battery_status_full_phase_C = results_df[results_df["battery_status_phase_C"] == "full"].shape[0]
battery_status_empty_phase_A = results_df[results_df["battery_status_phase_A"] == "empty"].shape[0]
battery_status_empty_phase_B = results_df[results_df["battery_status_phase_B"] == "empty"].shape[0]
battery_status_empty_phase_C = results_df[results_df["battery_status_phase_C"] == "empty"].shape[0]
battery_status_discharging_phase_A = results_df[results_df["battery_status_phase_A"] == "discharging"].shape[0]
battery_status_discharging_phase_B = results_df[results_df["battery_status_phase_B"] == "discharging"].shape[0]
battery_status_discharging_phase_C = results_df[results_df["battery_status_phase_C"] == "discharging"].shape[0]
battery_status_charging_phase_A = results_df[results_df["battery_status_phase_A"] == "charging"].shape[0]
battery_status_charging_phase_B = results_df[results_df["battery_status_phase_B"] == "charging"].shape[0]
battery_status_charging_phase_C = results_df[results_df["battery_status_phase_C"] == "charging"].shape[0]
battery_status_phase_A_total = battery_status_full_phase_A + battery_status_empty_phase_A + battery_status_discharging_phase_A + battery_status_charging_phase_A
battery_status_phase_B_total = battery_status_full_phase_B + battery_status_empty_phase_B + battery_status_discharging_phase_B + battery_status_charging_phase_B
battery_status_phase_C_total = battery_status_full_phase_C + battery_status_empty_phase_C + battery_status_discharging_phase_C + battery_status_charging_phase_C

# Create a table for battery status with value and percentage
battery_status_data = [
    ["Full", f"{battery_status_full_phase_A} ({battery_status_full_phase_A / battery_status_phase_A_total * 100:.2f}%)", f"{battery_status_full_phase_B} ({battery_status_full_phase_B / battery_status_phase_B_total * 100:.2f}%)", f"{battery_status_full_phase_C} ({battery_status_full_phase_C / battery_status_phase_C_total * 100:.2f}%)"],
    ["Empty", f"{battery_status_empty_phase_A} ({battery_status_empty_phase_A / battery_status_phase_A_total * 100:.2f}%)", f"{battery_status_empty_phase_B} ({battery_status_empty_phase_B / battery_status_phase_B_total * 100:.2f}%)", f"{battery_status_empty_phase_C} ({battery_status_empty_phase_C / battery_status_phase_C_total * 100:.2f}%)"],
    ["Discharging", f"{battery_status_discharging_phase_A} ({battery_status_discharging_phase_A / battery_status_phase_A_total * 100:.2f}%)", f"{battery_status_discharging_phase_B} ({battery_status_discharging_phase_B / battery_status_phase_B_total * 100:.2f}%)", f"{battery_status_discharging_phase_C} ({battery_status_discharging_phase_C / battery_status_phase_C_total * 100:.2f}%)"],
    ["Charging", f"{battery_status_charging_phase_A} ({battery_status_charging_phase_A / battery_status_phase_A_total * 100:.2f}%)", f"{battery_status_charging_phase_B} ({battery_status_charging_phase_B / battery_status_phase_B_total * 100:.2f}%)", f"{battery_status_charging_phase_C} ({battery_status_charging_phase_C / battery_status_phase_C_total * 100:.2f}%)"],
    ["Total", battery_status_phase_A_total, battery_status_phase_B_total, battery_status_phase_C_total]
]

headers = ["Status", "Phase A", "Phase B", "Phase C"]

print("")
print("Battery Status:")
print("This table illustrates the time the battery spends in each status.")
print("A battery that is fully charged too often may indicate that it is undersized, whereas a battery that is frequently empty may suggest that it is oversized.")
print("The distribution of charging and discharging can provide insights into whether the battery is used frequently or infrequently. If the discharging percentage exceeds the charging percentage, it indicates that the battery is frequently utilized to reduce grid consumption.")
print(tabulate(battery_status_data, headers, tablefmt="grid"))

# Statistics for charing power and discharging power when at the peak
battery_charging_max_power_phase_A = results_df[results_df["charge_power_phase_A_W"] == max_charge_power_watts[0]].shape[0]
battery_charging_max_power_phase_B = results_df[results_df["charge_power_phase_B_W"] == max_charge_power_watts[1]].shape[0]
battery_charging_max_power_phase_C = results_df[results_df["charge_power_phase_C_W"] == max_charge_power_watts[2]].shape[0]
battery_discharging_max_power_phase_A = results_df[results_df["discharge_power_phase_A_W"] == max_discharge_power_watts[0]].shape[0]
battery_discharging_max_power_phase_B = results_df[results_df["discharge_power_phase_B_W"] == max_discharge_power_watts[1]].shape[0]
battery_discharging_max_power_phase_C = results_df[results_df["discharge_power_phase_C_W"] == max_discharge_power_watts[2]].shape[0]
battery_charging_not_max_power_phase_A = results_df[(results_df["charge_power_phase_A_W"] != max_charge_power_watts[0]) & (results_df["charge_power_phase_A_W"] != 0)].shape[0]
battery_charging_not_max_power_phase_B = results_df[(results_df["charge_power_phase_B_W"] != max_charge_power_watts[1]) & (results_df["charge_power_phase_B_W"] != 0)].shape[0]
battery_charging_not_max_power_phase_C = results_df[(results_df["charge_power_phase_C_W"] != max_charge_power_watts[2]) & (results_df["charge_power_phase_C_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase_A = results_df[(results_df["discharge_power_phase_A_W"] != max_discharge_power_watts[0]) & (results_df["discharge_power_phase_A_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase_B = results_df[(results_df["discharge_power_phase_B_W"] != max_discharge_power_watts[1]) & (results_df["discharge_power_phase_B_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase_C = results_df[(results_df["discharge_power_phase_C_W"] != max_discharge_power_watts[2]) & (results_df["discharge_power_phase_C_W"] != 0)].shape[0]

def pct(part, total):
    return f"{(part / total * 100):.2f}%" if total > 0 else "n/a"

# Create a table for charging and discharging power at the peak
charging_discharging_power_data = [
    [
        "Charging at Max Power",
        pct(
            battery_charging_max_power_phase_A,
            battery_charging_max_power_phase_A + battery_charging_not_max_power_phase_A
        ),
        pct(
            battery_charging_max_power_phase_B,
            battery_charging_max_power_phase_B + battery_charging_not_max_power_phase_B
        ),
        pct(
            battery_charging_max_power_phase_C,
            battery_charging_max_power_phase_C + battery_charging_not_max_power_phase_C
        ),
    ],
    [
        "Charging Not at Max Power",
        pct(
            battery_charging_not_max_power_phase_A,
            battery_charging_max_power_phase_A + battery_charging_not_max_power_phase_A
        ),
        pct(
            battery_charging_not_max_power_phase_B,
            battery_charging_max_power_phase_B + battery_charging_not_max_power_phase_B
        ),
        pct(
            battery_charging_not_max_power_phase_C,
            battery_charging_max_power_phase_C + battery_charging_not_max_power_phase_C
        ),
    ],
    [
        "Discharging at Max Power",
        pct(
            battery_discharging_max_power_phase_A,
            battery_discharging_max_power_phase_A + battery_discharging_not_max_power_phase_A
        ),
        pct(
            battery_discharging_max_power_phase_B,
            battery_discharging_max_power_phase_B + battery_discharging_not_max_power_phase_B
        ),
        pct(
            battery_discharging_max_power_phase_C,
            battery_discharging_max_power_phase_C + battery_discharging_not_max_power_phase_C
        ),
    ],
    [
        "Discharging Not at Max Power",
        pct(
            battery_discharging_not_max_power_phase_A,
            battery_discharging_max_power_phase_A + battery_discharging_not_max_power_phase_A
        ),
        pct(
            battery_discharging_not_max_power_phase_B,
            battery_discharging_max_power_phase_B + battery_discharging_not_max_power_phase_B
        ),
        pct(
            battery_discharging_not_max_power_phase_C,
            battery_discharging_max_power_phase_C + battery_discharging_not_max_power_phase_C
        ),
    ],
]

headers = ["Metric", "Phase A", "Phase B", "Phase C"]

print("")
print("Charging and Discharging Power at Peak:")
print("The batteries are designed to charge and discharge at a specific maximum power level.")
print("Frequent charging at maximum power may suggest the need for a more powerful system or an additional battery connected in parallel. The same consideration applies to discharging.")
print(tabulate(charging_discharging_power_data, headers, tablefmt="grid"))

print("")

# Convert results to DataFrame and export to CSV
print("**********************************")
if args.export_csv:
    print("Exporting CSV simulation results...")
    results_df.to_csv(args.export_csv, index=False)
    print(f"+ CSV simulation results exported to {args.export_csv}")

# ===============================
# BUILD SIMULATION RANGES
# ===============================
print("Building simulation ranges...")

ranges = []
range_index = 0

# Prepare monthly groups and total range count (for progress bar)
results_df["year"] = results_df["timestamp"].dt.year
results_df["month"] = results_df["timestamp"].dt.month
monthly_groups = list(results_df.groupby(["year", "month"]))
total_ranges = 1 + len(monthly_groups)  # global + monthly

# ---- Global range ----
ranges.append({
    "range_index": range_index,
    "range_id": "global",
    "range_type": "global",
    "range": build_range_metadata(results_df, duration_days_global),
    "results": build_canonical_results(
        results_df,
        battery_cost,
        duration_days_global,
        cycles=battery_cycles_global
    )
})

range_index += 1
print_progress_bar(range_index, total_ranges, prefix="Ranges")

# ---- Monthly ranges ----
for (year, month), df_month in monthly_groups:
    duration_days_month = max(
        1,
        (df_month["timestamp"].max() - df_month["timestamp"].min()).days
    )

    ranges.append({
        "range_index": range_index,
        "range_id": f"{year}-{month:02d}",
        "range_type": "monthly",
        "range": build_range_metadata(df_month, duration_days_month),
        "results": build_canonical_results(
            df_month,
            battery_cost,
            duration_days_month,
            cycles=None
        )
    })

    range_index += 1
    print_progress_bar(range_index, total_ranges, prefix="Ranges")

print()  # newline after progress bar

# ===============================
# EXPORT JSON
# ===============================
if args.export_json:
    print("Exporting JSON report...")
    json_report = build_json_report(simulation_ranges=ranges)
    export_json_report(json_report, args.export_json)
