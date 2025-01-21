#! /usr/bin/python3
"""
Battery Simulation Script
This script simulates the behavior of a battery system in a house with three phases (A, B, and C). 
It reads power consumption data from CSV files, processes the data, and simulates the battery's 
charging and discharging behavior based on the house's power consumption and predefined battery 
parameters. The script also calculates the financial impact of using the battery system.
"""
import pandas as pd
import sys
from tabulate import tabulate

###################################################################
# CONFIGURATION
###################################################################
# ---- Battery parameters ----
battery_capacity_Wh = [3940, 3940, 3940]        # Battery capacity per phase (Wh)
max_charge_power_watts = [1200, 1200, 1200]     # Max charge power per phase (W)
max_discharge_power_watts = [1200, 1200, 1200]  # Max discharge power per phase (W)
battery_charge_efficiency = 0.9                 # Charge efficiency (90%)
battery_discharge_efficiency = 0.9              # Discharge efficiency (90%)
battery_max_cycles = 6000                       # Battery lifespan in cycles
battery_cost = 5847                             # Battery cost (CHF)
battery_soc = [100, 100, 100]                   # Initial state of charge in % (when the simulation starts)

# ---- Electricity tariff configuration ----
tariff_config = {
    "peak": {
        "tariff_consume": 0.34,      # CHF/kWh
        "tariff_inject": 0.10,       # CHF/kWh
        "days": [0, 1, 2, 3, 4],     # Monday to Friday
        "hours": range(17, 22)       # 5 PM to 10 PM
    },
    "off_peak": {
        "tariff_consume": 0.34,      # CHF/kWh for the rest of the time
        "tariff_inject": 0.10        # CHF/kWh for the rest of the time
    }
}

###################################################################
# FUNCTIONS
###################################################################
# ---- Function to determine the current tariff mode ----
# HP = Peak Hours
# HC = Off-Peak Hours
def get_current_tarif_mode(hour, day):
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
def update_energy(energy, tarif_mode, consumed_dict, injected_dict, phase):
    if energy < 0:
        injected_dict[phase][tarif_mode] += abs(energy)
    else:
        consumed_dict[phase][tarif_mode] += energy

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
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase] - discharge_power_watts[phase]
                battery_cycle[phase] = abs(energy_to_be_consumed_from_battery_Wh / battery_capacity_Wh[phase] / 2)
                battery_status[phase] = "discharging"
            else:
                # No energy in the battery
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase]
                battery_status[phase] = "empty"

    return {
        "phase1": {
            "new_house_grid_power_watts": new_house_grid_power_watts[0],
            "energy_in_battery_Wh": energy_in_battery_Wh[0],
            "battery_status": battery_status[0],
            "battery_cycle": battery_cycle[0],
            "charge_power_watts": charge_power_watts[0],
            "discharge_power_watts": discharge_power_watts[0]
        },
        "phase2": {
            "new_house_grid_power_watts": new_house_grid_power_watts[1],
            "energy_in_battery_Wh": energy_in_battery_Wh[1],
            "battery_status": battery_status[1],
            "battery_cycle": battery_cycle[1],
            "charge_power_watts": charge_power_watts[1],
            "discharge_power_watts": discharge_power_watts[1]
        },
        "phase3": {
            "new_house_grid_power_watts": new_house_grid_power_watts[2],
            "energy_in_battery_Wh": energy_in_battery_Wh[2],
            "battery_status": battery_status[2],
            "battery_cycle": battery_cycle[2],
            "charge_power_watts": charge_power_watts[2],
            "discharge_power_watts": discharge_power_watts
        }
    }

###################################################################
# MAIN
###################################################################
# ---- Reading CSV files from the command line ----
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python battery_simulator.py <house_phase_a.csv> <house_phase_b.csv> <house_phase_c.csv>")
    sys.exit(1)
house_phase_a_file = sys.argv[1]
house_phase_b_file = sys.argv[2]
house_phase_c_file = sys.argv[3]

# Initial battery state
energy_in_battery_Wh = [battery_capacity_Wh[i] * (battery_soc[i] / 100) for i in range(3)]
print("Initial battery state:")
print(f"+ phase 1: {energy_in_battery_Wh[0]} kWh")
print(f"+ phase 2: {energy_in_battery_Wh[1]} kWh")
print(f"+ phase 3: {energy_in_battery_Wh[2]} kWh")

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
number_of_data_days = (merged_end_timestamp - merged_start_timestamp).days

# Write the merged data to a new CSV file
#print("Exporting merged data to merged_data.csv")
#merged_data.to_csv("merged_data.csv", index=False)

print("Starting simulation...")
# ---- Execution on the data from the CSV files ----
results = []
energy_production_Wh_total = 0
battery_cycle_total = {"phase1": 0, "phase2": 0, "phase3": 0}

# Initialize dictionaries to store injected and consumed energy for each phase and tariff mode
simulated_injected_energy_Wh = {"phase_a": {"HP": 0, "HC": 0}, "phase_b": {"HP": 0, "HC": 0}, "phase_c": {"HP": 0, "HC": 0}}
simulated_consumed_energy_Wh = {"phase_a": {"HP": 0, "HC": 0}, "phase_b": {"HP": 0, "HC": 0}, "phase_c": {"HP": 0, "HC": 0}}
current_injected_energy_Wh = {"phase_a": {"HP": 0, "HC": 0}, "phase_b": {"HP": 0, "HC": 0}, "phase_c": {"HP": 0, "HC": 0}}
current_consumed_energy_Wh = {"phase_a": {"HP": 0, "HC": 0}, "phase_b": {"HP": 0, "HC": 0}, "phase_c": {"HP": 0, "HC": 0}}

for i in range(len(merged_data)):
    # Print progress bar
    progress = (i + 1) / len(merged_data) * 100
    sys.stdout.write(f"\r+ Progress: [{int(progress) * '#'}{(100 - int(progress)) * ' '}] {progress:.2f}%")
    sys.stdout.flush()

    # Get the current timestamp, weekday and hour
    timestamp = merged_data.iloc[i]["timestamp"]
    weekday = timestamp.weekday()
    hour = timestamp.hour

    # Get the current tarif mode
    tarif_mode = get_current_tarif_mode(hour, weekday)

    # Statistics without the battery
    house_consumption_watts = [merged_data.iloc[i][f"phase_{phase}"] for phase in ["a", "b", "c"]]
    # - House consumption
    energy_house_consumption_Wh_phase_a=calculate_energy_Wh(house_consumption_watts[0],1)
    energy_house_consumption_Wh_phase_b=calculate_energy_Wh(house_consumption_watts[1],1)
    energy_house_consumption_Wh_phase_c=calculate_energy_Wh(house_consumption_watts[2],1)

    # Simulation with the battery
    sim_result = simulate_battery_behavior(house_consumption_watts)

    # - House consumption
    simulated_energy_house_consumption_Wh_phase_a=calculate_energy_Wh(sim_result["phase1"]["new_house_grid_power_watts"],1)
    simulated_energy_house_consumption_Wh_phase_b=calculate_energy_Wh(sim_result["phase2"]["new_house_grid_power_watts"],1)
    simulated_energy_house_consumption_Wh_phase_c=calculate_energy_Wh(sim_result["phase3"]["new_house_grid_power_watts"],1)
   
    # - Battery energy
    battery_cycle_total["phase1"] += sim_result["phase1"]["battery_cycle"]
    battery_cycle_total["phase2"] += sim_result["phase2"]["battery_cycle"]
    battery_cycle_total["phase3"] += sim_result["phase3"]["battery_cycle"]

    # List of phases and tariff modes
    phases = ["phase_a", "phase_b", "phase_c"]
    tariff_modes = ["HP", "HC"]

    # Process real energy consumption
    for phase, energy in zip(phases, [
        energy_house_consumption_Wh_phase_a,
        energy_house_consumption_Wh_phase_b,
        energy_house_consumption_Wh_phase_c
    ]):
        update_energy(energy, tarif_mode, current_consumed_energy_Wh, current_injected_energy_Wh, phase)

    # Process simulated energy consumption
    for phase, energy in zip(phases, [
        simulated_energy_house_consumption_Wh_phase_a,
        simulated_energy_house_consumption_Wh_phase_b,
        simulated_energy_house_consumption_Wh_phase_c
    ]):
        update_energy(energy, tarif_mode, simulated_consumed_energy_Wh, simulated_injected_energy_Wh, phase)

    # Export all info in a csv file
    results.append({
        "timestamp": timestamp,
        "tarif_mode": tarif_mode,
        "house_consumption_phase_a_Wh": energy_house_consumption_Wh_phase_a,
        "house_consumption_phase_b_Wh": energy_house_consumption_Wh_phase_b,
        "house_consumption_phase_c_Wh": energy_house_consumption_Wh_phase_c,
        "simulated_house_consumption_phase_a_Wh": simulated_energy_house_consumption_Wh_phase_a,
        "simulated_house_consumption_phase_b_Wh": simulated_energy_house_consumption_Wh_phase_b,
        "simulated_house_consumption_phase_c_Wh": simulated_energy_house_consumption_Wh_phase_c,
        "battery_energy_phase1_Wh": sim_result["phase1"]["energy_in_battery_Wh"],
        "battery_energy_phase2_Wh": sim_result["phase2"]["energy_in_battery_Wh"],
        "battery_energy_phase3_Wh": sim_result["phase3"]["energy_in_battery_Wh"],
        "battery_status_phase1": sim_result["phase1"]["battery_status"],
        "battery_status_phase2": sim_result["phase2"]["battery_status"],
        "battery_status_phase3": sim_result["phase3"]["battery_status"],
        "charge_power_phase1_W": sim_result["phase1"]["charge_power_watts"],
        "charge_power_phase2_W": sim_result["phase2"]["charge_power_watts"],
        "charge_power_phase3_W": sim_result["phase3"]["charge_power_watts"],
        "discharge_power_phase1_W": sim_result["phase1"]["discharge_power_watts"],
        "discharge_power_phase2_W": sim_result["phase2"]["discharge_power_watts"],
        "discharge_power_phase3_W": sim_result["phase3"]["discharge_power_watts"]
    })

# Convert results to DataFrame and export to CSV
results_df = pd.DataFrame(results)
#results_df.to_csv("simulation_results.csv", index=False)
#print("\n+ Successfully exported simulation results to simulation_results.csv")

print("****************************************************************************************")
# Data for injected energy
data_injected = [
    ["Phase A Injected Off-Peak", int(current_injected_energy_Wh['phase_a']['HC'] / 1000), int(simulated_injected_energy_Wh['phase_a']['HC'] / 1000), int((simulated_injected_energy_Wh['phase_a']['HC'] - current_injected_energy_Wh['phase_a']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['phase_a']['HC'] - current_injected_energy_Wh['phase_a']['HC']) / 1000)],
    ["Phase A Injected Peak", int(current_injected_energy_Wh['phase_a']['HP'] / 1000), int(simulated_injected_energy_Wh['phase_a']['HP'] / 1000), int((simulated_injected_energy_Wh['phase_a']['HP'] - current_injected_energy_Wh['phase_a']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['phase_a']['HP'] - current_injected_energy_Wh['phase_a']['HP']) / 1000)],
    ["Phase B Injected Off-Peak", int(current_injected_energy_Wh['phase_b']['HC'] / 1000), int(simulated_injected_energy_Wh['phase_b']['HC'] / 1000), int((simulated_injected_energy_Wh['phase_b']['HC'] - current_injected_energy_Wh['phase_b']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['phase_b']['HC'] - current_injected_energy_Wh['phase_b']['HC']) / 1000)],
    ["Phase B Injected Peak", int(current_injected_energy_Wh['phase_b']['HP'] / 1000), int(simulated_injected_energy_Wh['phase_b']['HP'] / 1000), int((simulated_injected_energy_Wh['phase_b']['HP'] - current_injected_energy_Wh['phase_b']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['phase_b']['HP'] - current_injected_energy_Wh['phase_b']['HP']) / 1000)],
    ["Phase C Injected Off-Peak", int(current_injected_energy_Wh['phase_c']['HC'] / 1000), int(simulated_injected_energy_Wh['phase_c']['HC'] / 1000), int((simulated_injected_energy_Wh['phase_c']['HC'] - current_injected_energy_Wh['phase_c']['HC']) / 1000), int(get_current_tariff_price("HC", "inject") * abs(simulated_injected_energy_Wh['phase_c']['HC'] - current_injected_energy_Wh['phase_c']['HC']) / 1000)],
    ["Phase C Injected Peak", int(current_injected_energy_Wh['phase_c']['HP'] / 1000), int(simulated_injected_energy_Wh['phase_c']['HP'] / 1000), int((simulated_injected_energy_Wh['phase_c']['HP'] - current_injected_energy_Wh['phase_c']['HP']) / 1000), int(get_current_tariff_price("HP", "inject") * abs(simulated_injected_energy_Wh['phase_c']['HP'] - current_injected_energy_Wh['phase_c']['HP']) / 1000)]
]

# Data for consumed energy
data_consumed = [
    ["Phase A Consumed Off-Peak", int(current_consumed_energy_Wh['phase_a']['HC'] / 1000), int(simulated_consumed_energy_Wh['phase_a']['HC'] / 1000), int((simulated_consumed_energy_Wh['phase_a']['HC'] - current_consumed_energy_Wh['phase_a']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['phase_a']['HC'] - current_consumed_energy_Wh['phase_a']['HC']) / 1000)],
    ["Phase A Consumed Peak", int(current_consumed_energy_Wh['phase_a']['HP'] / 1000), int(simulated_consumed_energy_Wh['phase_a']['HP'] / 1000), int((simulated_consumed_energy_Wh['phase_a']['HP'] - current_consumed_energy_Wh['phase_a']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['phase_a']['HP'] - current_consumed_energy_Wh['phase_a']['HP']) / 1000)],
    ["Phase B Consumed Off-Peak", int(current_consumed_energy_Wh['phase_b']['HC'] / 1000), int(simulated_consumed_energy_Wh['phase_b']['HC'] / 1000), int((simulated_consumed_energy_Wh['phase_b']['HC'] - current_consumed_energy_Wh['phase_b']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['phase_b']['HC'] - current_consumed_energy_Wh['phase_b']['HC']) / 1000)],
    ["Phase B Consumed Peak", int(current_consumed_energy_Wh['phase_b']['HP'] / 1000), int(simulated_consumed_energy_Wh['phase_b']['HP'] / 1000), int((simulated_consumed_energy_Wh['phase_b']['HP'] - current_consumed_energy_Wh['phase_b']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['phase_b']['HP'] - current_consumed_energy_Wh['phase_b']['HP']) / 1000)],
    ["Phase C Consumed Off-Peak", int(current_consumed_energy_Wh['phase_c']['HC'] / 1000), int(simulated_consumed_energy_Wh['phase_c']['HC'] / 1000), int((simulated_consumed_energy_Wh['phase_c']['HC'] - current_consumed_energy_Wh['phase_c']['HC']) / 1000), int(get_current_tariff_price("HC", "consume") * abs(simulated_consumed_energy_Wh['phase_c']['HC'] - current_consumed_energy_Wh['phase_c']['HC']) / 1000)],
    ["Phase C Consumed Peak", int(current_consumed_energy_Wh['phase_c']['HP'] / 1000), int(simulated_consumed_energy_Wh['phase_c']['HP'] / 1000), int((simulated_consumed_energy_Wh['phase_c']['HP'] - current_consumed_energy_Wh['phase_c']['HP']) / 1000), int(get_current_tariff_price("HP", "consume") * abs(simulated_consumed_energy_Wh['phase_c']['HP'] - current_consumed_energy_Wh['phase_c']['HP']) / 1000)]
]

# Calculate totals for each column
totals_injected = ["Total Injected", sum(row[1] for row in data_injected), sum(row[2] for row in data_injected), sum(row[3] for row in data_injected), sum(row[4] for row in data_injected)]
totals_consumed = ["Total Consumed", sum(row[1] for row in data_consumed), sum(row[2] for row in data_consumed), sum(row[3] for row in data_consumed), sum(row[4] for row in data_consumed)]
total_gain_CHF = totals_injected[4] + totals_consumed[4]

headers = ["Phase", "Current (kWh)", "Simulated (kWh)", "Delta (kWh)", "Delta (CHF)"]

print("Injected Energy:")
print(tabulate(data_injected + [totals_injected], headers, tablefmt="grid"))

print("")
print("Consumed Energy:")
print(tabulate(data_consumed + [totals_consumed], headers, tablefmt="grid"))

print("")
print("Rentability:")
print(f"+ Total gain: {total_gain_CHF} CHF for {number_of_data_days} days or per year: {total_gain_CHF / number_of_data_days * 365:.0f} CHF (extrapolated)")
print(f"+ Amortization time: {battery_cost / total_gain_CHF * number_of_data_days / 365:.2f} years if the cost of the battery is {battery_cost} CHF")

print("")
# Battery statistics
battery_stats_data = [
    ["Cycles", int(battery_cycle_total['phase1']), int(battery_cycle_total['phase2']), int(battery_cycle_total['phase3']), battery_max_cycles],
    ["Expected life (years)", int(battery_max_cycles / battery_cycle_total['phase1'] * number_of_data_days / 365), int(battery_max_cycles / battery_cycle_total['phase2'] * number_of_data_days / 365), int(battery_max_cycles / battery_cycle_total['phase3'] * number_of_data_days / 365), ""],
    ["Remaining energy (Wh)", int(energy_in_battery_Wh[0]), int(energy_in_battery_Wh[1]), int(energy_in_battery_Wh[2]), ""]
]

headers = ["Metric", "Phase 1", "Phase 2", "Phase 3", "Max/Config"]

print("Battery Statistics:")
print(tabulate(battery_stats_data, headers, tablefmt="grid"))


# Calculate the number of entries in results with battery status "full" or "empty"
battery_status_full_phase1 = results_df[results_df["battery_status_phase1"] == "full"].shape[0]
battery_status_full_phase2 = results_df[results_df["battery_status_phase2"] == "full"].shape[0]
battery_status_full_phase3 = results_df[results_df["battery_status_phase3"] == "full"].shape[0]
battery_status_empty_phase1 = results_df[results_df["battery_status_phase1"] == "empty"].shape[0]
battery_status_empty_phase2 = results_df[results_df["battery_status_phase2"] == "empty"].shape[0]
battery_status_empty_phase3 = results_df[results_df["battery_status_phase3"] == "empty"].shape[0]
battery_status_discharging_phase1 = results_df[results_df["battery_status_phase1"] == "discharging"].shape[0]
battery_status_discharging_phase2 = results_df[results_df["battery_status_phase2"] == "discharging"].shape[0]
battery_status_discharging_phase3 = results_df[results_df["battery_status_phase3"] == "discharging"].shape[0]
battery_status_charging_phase1 = results_df[results_df["battery_status_phase1"] == "charging"].shape[0]
battery_status_charging_phase2 = results_df[results_df["battery_status_phase2"] == "charging"].shape[0]
battery_status_charging_phase3 = results_df[results_df["battery_status_phase3"] == "charging"].shape[0]
battery_status_phase1_total = battery_status_full_phase1 + battery_status_empty_phase1 + battery_status_discharging_phase1 + battery_status_charging_phase1
battery_status_phase2_total = battery_status_full_phase2 + battery_status_empty_phase2 + battery_status_discharging_phase2 + battery_status_charging_phase2
battery_status_phase3_total = battery_status_full_phase3 + battery_status_empty_phase3 + battery_status_discharging_phase3 + battery_status_charging_phase3

# Create a table for battery status with value and percentage
battery_status_data = [
    ["Full", f"{battery_status_full_phase1} ({battery_status_full_phase1 / battery_status_phase1_total * 100:.2f}%)", f"{battery_status_full_phase2} ({battery_status_full_phase2 / battery_status_phase2_total * 100:.2f}%)", f"{battery_status_full_phase3} ({battery_status_full_phase3 / battery_status_phase3_total * 100:.2f}%)"],
    ["Empty", f"{battery_status_empty_phase1} ({battery_status_empty_phase1 / battery_status_phase1_total * 100:.2f}%)", f"{battery_status_empty_phase2} ({battery_status_empty_phase2 / battery_status_phase2_total * 100:.2f}%)", f"{battery_status_empty_phase3} ({battery_status_empty_phase3 / battery_status_phase3_total * 100:.2f}%)"],
    ["Discharging", f"{battery_status_discharging_phase1} ({battery_status_discharging_phase1 / battery_status_phase1_total * 100:.2f}%)", f"{battery_status_discharging_phase2} ({battery_status_discharging_phase2 / battery_status_phase2_total * 100:.2f}%)", f"{battery_status_discharging_phase3} ({battery_status_discharging_phase3 / battery_status_phase3_total * 100:.2f}%)"],
    ["Charging", f"{battery_status_charging_phase1} ({battery_status_charging_phase1 / battery_status_phase1_total * 100:.2f}%)", f"{battery_status_charging_phase2} ({battery_status_charging_phase2 / battery_status_phase2_total * 100:.2f}%)", f"{battery_status_charging_phase3} ({battery_status_charging_phase3 / battery_status_phase3_total * 100:.2f}%)"],
    ["Total", battery_status_phase1_total, battery_status_phase2_total, battery_status_phase3_total]
]

headers = ["Status", "Phase 1", "Phase 2", "Phase 3"]

print("")
print("Battery Status:")
print(tabulate(battery_status_data, headers, tablefmt="grid"))

# Statistics for charing power and discharging power when at the peak
battery_charging_max_power_phase1 = results_df[results_df["charge_power_phase1_W"] == max_charge_power_watts[0]].shape[0]
battery_charging_max_power_phase2 = results_df[results_df["charge_power_phase2_W"] == max_charge_power_watts[1]].shape[0]
battery_charging_max_power_phase3 = results_df[results_df["charge_power_phase3_W"] == max_charge_power_watts[2]].shape[0]
battery_discharging_max_power_phase1 = results_df[results_df["discharge_power_phase1_W"] == max_discharge_power_watts[0]].shape[0]
battery_discharging_max_power_phase2 = results_df[results_df["discharge_power_phase2_W"] == max_discharge_power_watts[1]].shape[0]
battery_discharging_max_power_phase3 = results_df[results_df["discharge_power_phase3_W"] == max_discharge_power_watts[2]].shape[0]
battery_charging_not_max_power_phase1 = results_df[(results_df["charge_power_phase1_W"] != max_charge_power_watts[0]) & (results_df["charge_power_phase1_W"] != 0)].shape[0]
battery_charging_not_max_power_phase2 = results_df[(results_df["charge_power_phase2_W"] != max_charge_power_watts[1]) & (results_df["charge_power_phase2_W"] != 0)].shape[0]
battery_charging_not_max_power_phase3 = results_df[(results_df["charge_power_phase3_W"] != max_charge_power_watts[2]) & (results_df["charge_power_phase3_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase1 = results_df[(results_df["discharge_power_phase1_W"] != max_discharge_power_watts[0]) & (results_df["discharge_power_phase1_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase2 = results_df[(results_df["discharge_power_phase2_W"] != max_discharge_power_watts[1]) & (results_df["discharge_power_phase2_W"] != 0)].shape[0]
battery_discharging_not_max_power_phase3 = results_df[(results_df["discharge_power_phase3_W"] != max_discharge_power_watts[2]) & (results_df["discharge_power_phase3_W"] != 0)].shape[0]

# Create a table for charging and discharging power at the peak
charging_discharging_power_data = [
    ["Charging at Max Power", f"{battery_charging_max_power_phase1} ({battery_charging_max_power_phase1 / (battery_charging_max_power_phase1 + battery_charging_not_max_power_phase1) * 100:.2f}%)", f"{battery_charging_max_power_phase2} ({battery_charging_max_power_phase2 / (battery_charging_max_power_phase2 + battery_charging_not_max_power_phase2) * 100:.2f}%)", f"{battery_charging_max_power_phase3} ({battery_charging_max_power_phase3 / (battery_charging_max_power_phase3 + battery_charging_not_max_power_phase3) * 100:.2f}%)"],
    ["Charging Not at Max Power", f"{battery_charging_not_max_power_phase1} ({battery_charging_not_max_power_phase1 / (battery_charging_max_power_phase1 + battery_charging_not_max_power_phase1) * 100:.2f}%)", f"{battery_charging_not_max_power_phase2} ({battery_charging_not_max_power_phase2 / (battery_charging_max_power_phase2 + battery_charging_not_max_power_phase2) * 100:.2f}%)", f"{battery_charging_not_max_power_phase3} ({battery_charging_not_max_power_phase3 / (battery_charging_max_power_phase3 + battery_charging_not_max_power_phase3) * 100:.2f}%)"],
    ["Discharging at Max Power", f"{battery_discharging_max_power_phase1} ({battery_discharging_max_power_phase1 / (battery_discharging_max_power_phase1 + battery_discharging_not_max_power_phase1) * 100:.2f}%)", f"{battery_discharging_max_power_phase2} ({battery_discharging_max_power_phase2 / (battery_discharging_max_power_phase2 + battery_discharging_not_max_power_phase2) * 100:.2f}%)", f"{battery_discharging_max_power_phase3} ({battery_discharging_max_power_phase3 / (battery_discharging_max_power_phase3 + battery_discharging_not_max_power_phase3) * 100:.2f}%)"],
    ["Discharging Not at Max Power", f"{battery_discharging_not_max_power_phase1} ({battery_discharging_not_max_power_phase1 / (battery_discharging_max_power_phase1 + battery_discharging_not_max_power_phase1) * 100:.2f}%)", f"{battery_discharging_not_max_power_phase2} ({battery_discharging_not_max_power_phase2 / (battery_discharging_max_power_phase2 + battery_discharging_not_max_power_phase2) * 100:.2f}%)", f"{battery_discharging_not_max_power_phase3} ({battery_discharging_not_max_power_phase3 / (battery_discharging_max_power_phase3 + battery_discharging_not_max_power_phase3) * 100:.2f}%)"]
]
headers = ["Metric", "Phase 1", "Phase 2", "Phase 3"]

print("")
print("Charging and Discharging Power at Peak:")
print(tabulate(charging_discharging_power_data, headers, tablefmt="grid"))