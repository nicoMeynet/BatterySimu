#! /usr/bin/python3
import pandas as pd
import sys
import time

# ---- Paramètres de la batterie ----
battery_capacity_Wh = [10000, 10000, 10000]  # Capacité de la batterie par phase (Wh)
max_charge_power_watts = [3000, 3000, 3000]      # Puissance max de charge par phase (W)
max_discharge_power_watts = [3000, 3000, 3000]   # Puissance max de décharge par phase (W)
battery_charge_efficiency = 0.9                     # Rendement (90%)
battery_discharge_efficiency = 0.9                     # Rendement (90%)
#soc_min = 20                         # Capacité de décharge minimale (%) (State of Charge)
#battery_cycles = 5000                # Durée de vie de la batterie en cycles

# ---- Configuration des tarifs d'électricité ----
tarif_config = {
    "pleines": {
        "tarif": 0.25,  # CHF/kWh
        "jours": [0, 1, 2, 3, 4],  # Lundi à vendredi
        "heures": range(17, 22)  # 17h à 22h
    },
    "creuses": {
        "tarif": 0.15  # CHF/kWh pour le reste du temps
    }
}

# ---- État initial de la batterie ----
battery_soc = [100, 100, 100]  # État de charge initial en %
energy_in_battery_Wh = [battery_capacity_Wh[i] * (battery_soc[i] / 100) for i in range(3)]
print(f"Initial battery energy phase 1: {energy_in_battery_Wh[0]} kWh")
print(f"Initial battery energy phase 2: {energy_in_battery_Wh[1]} kWh")
print(f"Initial battery energy phase 3: {energy_in_battery_Wh[2]} kWh")

# ---- Fonction pour déterminer le tarif actuel ----
def get_current_tarif(hour, day):
    if day in tarif_config["pleines"]["jours"] and hour in tarif_config["pleines"]["heures"]:
        return tarif_config["pleines"]["tarif"]
    return tarif_config["creuses"]["tarif"]

# ---- Fonction pour calculer l'énergie en Wh ----
def calculate_energy_Wh(power_watts,duration_minutes):
    return power_watts * duration_minutes/60

# ---- Simulation ----
def simulate_battery_behavior(house_grid_power_watts, hour, day):
    global energy_in_battery_Wh
    # Calcul du tarif actuel
    tarif = get_current_tarif(hour, day)

    # Calcul de la production et de la consommation par phase
    new_house_grid_power_watts = [0, 0, 0]
    status = [0, 0, 0]
    battery_cycle = [0, 0, 0]
    for phase in range(3):
        if house_grid_power_watts[phase] < 0:
            # On inject sur le reseau, charger la batterie avec le surplus
            surplus_watts = abs(house_grid_power_watts[phase])
            if energy_in_battery_Wh[phase] < battery_capacity_Wh[phase]:
                # Charger la batterie si pas pleine
                charge_power_watts = min(surplus_watts, max_charge_power_watts[phase])
                energy_to_be_injected_in_battery_Wh = calculate_energy_Wh(charge_power_watts,1) * battery_charge_efficiency
                energy_in_battery_Wh[phase] += energy_to_be_injected_in_battery_Wh
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase] + charge_power_watts
                battery_cycle[phase] = abs(energy_to_be_injected_in_battery_Wh / battery_capacity_Wh[phase] /2)
                status[phase] = "charging"
            else:
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase]
                status[phase] = "full"
        else:
            # On consomme sur le réseau, décharger la batterie pour compenser
            deficit_watts = house_grid_power_watts[phase]
            discharge_power_watts = min(deficit_watts, max_discharge_power_watts[phase])
            if energy_in_battery_Wh[phase] > 0:
                # Décharger la batterie si pas vide
                energy_to_be_consummed_from_battery_Wh = calculate_energy_Wh(discharge_power_watts,1) * (1 / battery_discharge_efficiency)
                energy_in_battery_Wh[phase] -= energy_to_be_consummed_from_battery_Wh
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase] - discharge_power_watts 
                battery_cycle[phase] = abs(energy_to_be_consummed_from_battery_Wh / battery_capacity_Wh[phase] /2)
                status[phase] = "discharging"
            else:
                new_house_grid_power_watts[phase] = house_grid_power_watts[phase]
                status[phase] = "empty"

    return {
        "phase1": {
            "New_House_Grid_power_watts": new_house_grid_power_watts[0],
            "Energy_in_battery_Wh": energy_in_battery_Wh[0],
            "Status": status[0],
            "Battery_cycle": battery_cycle[0]

        },
        "phase2": {
            "New_House_Grid_power_watts": new_house_grid_power_watts[1],
            "Energy_in_battery_Wh": energy_in_battery_Wh[1],
            "Status": status[1],
            "Battery_cycle": battery_cycle[1]
        },
        "phase3": {
            "New_House_Grid_power_watts": new_house_grid_power_watts[2],
            "Energy_in_battery_Wh": energy_in_battery_Wh[2],
            "Status": status[2],
            "Battery_cycle": battery_cycle[2]
        },
        "Tarif": tarif
    }


###################################################################
# MAIN
###################################################################
# ---- Lecture des fichiers CSV depuis la ligne de commande ----
if len(sys.argv) != 5:
    print("Usage: python battery_simulator.py <solar_csv> <house_phase_a.csv> <house_phase_b.csv> <house_phase_c.csv>")
    sys.exit(1)

solar_file = sys.argv[1]
house_phase_a_file = sys.argv[2]
house_phase_b_file = sys.argv[3]
house_phase_c_file = sys.argv[4]

# Chargement des fichiers CSV
print("Chargement des fichiers CSV")
solar_production = pd.read_csv(solar_file, parse_dates=["last_changed"])
house_phase_a = pd.read_csv(house_phase_a_file, parse_dates=["last_changed"])
house_phase_b = pd.read_csv(house_phase_b_file, parse_dates=["last_changed"])
house_phase_c = pd.read_csv(house_phase_c_file, parse_dates=["last_changed"])

# Mise en forme des données (nettoyage, transformation, etc.)
print("Mise en forme des données")
house_phase_a.drop(columns=["entity_id"], inplace=True)
house_phase_a.rename(columns={"last_changed": "timestamp", "state": "phase_a"}, inplace=True)
house_phase_b.drop(columns=["entity_id"], inplace=True)
house_phase_b.rename(columns={"last_changed": "timestamp", "state": "phase_b"}, inplace=True)
house_phase_c.drop(columns=["entity_id"], inplace=True)
house_phase_c.rename(columns={"last_changed": "timestamp", "state": "phase_c"}, inplace=True)
solar_production.rename(columns={"last_changed": "timestamp", "state": "solar_production"}, inplace=True)
solar_production.drop(columns=["entity_id"], inplace=True)

# Afficher les informations sur les données
print("Informations sur les données chargées")
solar_start_timestamp = solar_production["timestamp"].min()
solar_end_timestamp = solar_production["timestamp"].max()
solar_data_quantity = len(solar_production)
print(f"+ Solaire - Timestamp: {solar_start_timestamp} to {solar_end_timestamp} with {solar_data_quantity} lines")
house_phase_a_start_timestamp = house_phase_a["timestamp"].min()
house_phase_a_end_timestamp = house_phase_a["timestamp"].max()
house_phase_a_data_quantity = len(house_phase_a)
print(f"+ Phase A - Timestamp: {house_phase_a_start_timestamp} to {house_phase_a_end_timestamp} with {house_phase_a_data_quantity} lines")
house_phase_b_start_timestamp = house_phase_c["timestamp"].min()
house_phase_b_end_timestamp = house_phase_c["timestamp"].max()
house_phase_b_data_quantity = len(house_phase_b)
print(f"+ Phase B - Timestamp: {house_phase_b_start_timestamp} to {house_phase_b_end_timestamp} with {house_phase_b_data_quantity} lines")
house_phase_c_start_timestamp = house_phase_c["timestamp"].min()
house_phase_c_end_timestamp = house_phase_c["timestamp"].max()
house_phase_c_data_quantity = len(house_phase_c)
print(f"+ Phase C - Timestamp: {house_phase_c_start_timestamp} to {house_phase_c_end_timestamp} with {house_phase_c_data_quantity} lines")

# Arrondir les timestamps à la minute
print("Arrondissement des timestamps à la minute")
solar_production["timestamp"] = solar_production["timestamp"].dt.floor("min")
house_phase_a["timestamp"] = house_phase_a["timestamp"].dt.floor("min")
house_phase_b["timestamp"] = house_phase_b["timestamp"].dt.floor("min")
house_phase_c["timestamp"] = house_phase_c["timestamp"].dt.floor("min")

# Trier les colonnes pour avoir 'timestamp' en premier
print("Tri des colonnes pour avoir 'timestamp' à gauche")
columns_order = ["timestamp"] + [col for col in solar_production.columns if col != "timestamp"]
solar_production = solar_production[columns_order]
columns_order = ["timestamp"] + [col for col in house_phase_a.columns if col != "timestamp"]
house_phase_a = house_phase_a[columns_order]
columns_order = ["timestamp"] + [col for col in house_phase_b.columns if col != "timestamp"]
house_phase_b = house_phase_b[columns_order]
columns_order = ["timestamp"] + [col for col in house_phase_c.columns if col != "timestamp"]
house_phase_c = house_phase_c[columns_order]

# Group by timestamp and calculate the mean of the phase column
print("Groupement des données par timestamp")
solar_production["solar_production"] = pd.to_numeric(solar_production["solar_production"], errors="coerce")
solar_production = solar_production.dropna(subset=["solar_production"])
solar_production = solar_production.groupby("timestamp").agg({"solar_production": "mean"}).reset_index()
house_phase_a["phase_a"] = pd.to_numeric(house_phase_a["phase_a"], errors="coerce")
house_phase_a = house_phase_a.dropna(subset=["phase_a"])
house_phase_a = house_phase_a.groupby("timestamp").agg({"phase_a": "mean"}).reset_index()
house_phase_b["phase_b"] = pd.to_numeric(house_phase_b["phase_b"], errors="coerce")
house_phase_b = house_phase_b.dropna(subset=["phase_b"])
house_phase_b = house_phase_b.groupby("timestamp").agg({"phase_b": "mean"}).reset_index()
house_phase_c["phase_c"] = pd.to_numeric(house_phase_c["phase_c"], errors="coerce")
house_phase_c = house_phase_c.dropna(subset=["phase_c"])
house_phase_c = house_phase_c.groupby("timestamp").agg({"phase_c": "mean"}).reset_index()

# Compléter les timestamps manquants dans la production solaire avec une valeur moyenne entre les valeurs avant et après le timestamp manquant
print("Compléter les timestamps manquants dans la production solaire avec une valeur moyenne entre les valeurs avant et après le timestamp manquant")
all_timestamps = pd.date_range(start=solar_start_timestamp, end=solar_end_timestamp, freq='min')
solar_production = solar_production.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
solar_production["solar_production"] = solar_production["solar_production"].interpolate(method='linear').apply(lambda x: int(x))

# Compléter les timestamps manquants dans la phase A avec une valeur moyenne entre les valeurs avant et après le timestamp manquant
print("Completer les timestamps manquants dans la phase A avec une valeur moyenne entre les valeurs avant et après le timestamp manquant")
all_timestamps = pd.date_range(start=house_phase_a_start_timestamp, end=house_phase_a_end_timestamp, freq='min')
house_phase_a = house_phase_a.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_a["phase_a"] = house_phase_a["phase_a"].interpolate(method='linear').apply(lambda x: int(x))

# Compléter les timestamps manquants dans la phase B avec une valeur moyenne entre les valeurs avant et après le timestamp manquant
print("Completer les timestamps manquants dans la phase B avec une valeur moyenne entre les valeurs avant et après le timestamp manquant")
all_timestamps = pd.date_range(start=house_phase_b_start_timestamp, end=house_phase_b_end_timestamp, freq='min')
house_phase_b = house_phase_b.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_b["phase_b"] = house_phase_b["phase_b"].interpolate(method='linear').apply(lambda x: int(x))

# Compléter les timestamps manquants dans la phase C avec une valeur moyenne entre les valeurs avant et après le timestamp manquant
print("Completer les timestamps manquants dans la phase C avec une valeur moyenne entre les valeurs avant et après le timestamp manquant")
all_timestamps = pd.date_range(start=house_phase_c_start_timestamp, end=house_phase_c_end_timestamp, freq='min')
house_phase_c = house_phase_c.set_index("timestamp").reindex(all_timestamps).rename_axis("timestamp").reset_index()
house_phase_c["phase_c"] = house_phase_c["phase_c"].interpolate(method='linear').apply(lambda x: int(x))

# Merge the 4 datasets based on the timestamp, filling missing values with 0
print("Merge des données en une seule table")
merged_data = solar_production.merge(house_phase_a, on="timestamp", how="outer")
merged_data = merged_data.merge(house_phase_b, on="timestamp", how="outer")
merged_data = merged_data.merge(house_phase_c, on="timestamp", how="outer")

# Fill missing values with 0
print("Remplir les valeurs manquantes avec 0")
merged_data.fillna(0, inplace=True)

# Afficher les informations sur les données
print("Informations sur les données fusionnées")
merged_start_timestamp = merged_data["timestamp"].min()
merged_end_timestamp = merged_data["timestamp"].max()
merged_data_quantity = len(merged_data)
print(f"+ Merged - Timestamp: {merged_start_timestamp} to {merged_end_timestamp} with {merged_data_quantity} lines")

# Write the merged data to a new CSV file
print("Export des données fusionnées vers merged_data.csv")
merged_data.to_csv("merged_data.csv", index=False)

print("Début de la simulation...")
# ---- Exécution sur les données des fichiers CSV ----
results = []
energy_consumption_Wh_total=0
energy_production_Wh_total=0
simulated_energy_consumption_Wh_total = 0
battery_cycle_total = 0

for i in range(len(merged_data)):
    timestamp = merged_data.iloc[i]["timestamp"]
    weekday = timestamp.weekday()
    hour = timestamp.hour

    #print(f"Timestamp: {timestamp}")

    # Statistics without the battery
    solar_production_watts = [int(merged_data.iloc[i]["solar_production"] / 3)] * 3
    house_consumption_watts = [int(merged_data.iloc[i][f"phase_{phase}"]) for phase in ["a", "b", "c"]]
    # - House consumption
    energy_house_consumption_Wh_phase_a=calculate_energy_Wh(house_consumption_watts[0],1)
    energy_house_consumption_Wh_phase_b=calculate_energy_Wh(house_consumption_watts[1],1)
    energy_house_consumption_Wh_phase_c=calculate_energy_Wh(house_consumption_watts[2],1)
    energy_consumption_Wh_total+=energy_house_consumption_Wh_phase_a+energy_house_consumption_Wh_phase_b+energy_house_consumption_Wh_phase_c
    #print(f"House consumption: phase A: {int(energy_house_consumption_Wh_phase_a)} Wh, phase B: {int(energy_house_consumption_Wh_phase_b)} Wh, phase C: {int(energy_house_consumption_Wh_phase_c)} Wh")

    # - Solar production
    energy_solar_production_Wh_phase_a=calculate_energy_Wh(solar_production_watts[0],1)
    energy_solar_production_Wh_phase_b=calculate_energy_Wh(solar_production_watts[1],1)
    energy_solar_production_Wh_phase_c=calculate_energy_Wh(solar_production_watts[2],1)
    energy_production_Wh_total += energy_solar_production_Wh_phase_a+energy_solar_production_Wh_phase_b+energy_solar_production_Wh_phase_c

    # Simulation with the battery
    result = simulate_battery_behavior(house_consumption_watts, hour, weekday)
    # - House consumption
    simulated_energy_house_consumption_Wh_phase_a=int(calculate_energy_Wh(result["phase1"]["New_House_Grid_power_watts"],1))
    simulated_energy_house_consumption_Wh_phase_b=int(calculate_energy_Wh(result["phase2"]["New_House_Grid_power_watts"],1))
    simulated_energy_house_consumption_Wh_phase_c=int(calculate_energy_Wh(result["phase3"]["New_House_Grid_power_watts"],1))
    simulated_energy_consumption_Wh_total += simulated_energy_house_consumption_Wh_phase_a+simulated_energy_house_consumption_Wh_phase_b+simulated_energy_house_consumption_Wh_phase_c
    #print(f"Simulated house consumption: phase A: {simulated_energy_house_consumption_Wh_phase_a} Wh, phase B: {simulated_energy_house_consumption_Wh_phase_b} Wh, phase C: {simulated_energy_house_consumption_Wh_phase_c} Wh")

    battery_cycle_total += result["phase1"]["Battery_cycle"] + result["phase2"]["Battery_cycle"] + result["phase3"]["Battery_cycle"]

    #print(f"Energy in battery: phase A: {int(result['phase1']['Energy_in_battery_Wh'])} Wh, "
    #    f"phase B: {int(result['phase2']['Energy_in_battery_Wh'])} Wh, "
    #    f"phase C: {int(result['phase3']['Energy_in_battery_Wh'])} Wh")
    #print(f"Status: phase A: {result['phase1']['Status']}, "
    #    f"phase B: {result['phase2']['Status']}, "
    #    f"phase C: {result['phase3']['Status']}")
    #print(f"Tarif: {result['Tarif']} CHF/kWh")


    # Sleep 0.5 seconds to simulate processing time
    #time.sleep(0.01)
    

print("Simulation terminée: ")
print(f"- Solar energy produced:     {int(energy_production_Wh_total)} Wh")
print(f"- Current energy consumed:   {int(energy_consumption_Wh_total)} Wh")
print(f"- Simulated energy consumed: {int(simulated_energy_consumption_Wh_total)} Wh")
print(f"- Energy saved:              {int(energy_consumption_Wh_total - simulated_energy_consumption_Wh_total)} Wh")
print("Total energy in battery:")
print(f"- phase 1: {int(energy_in_battery_Wh[0])} Wh")
print(f"- phase 2: {int(energy_in_battery_Wh[1])} Wh")
print(f"- phase 3: {int(energy_in_battery_Wh[2])} Wh")

print(f"Total battery cycles: {battery_cycle_total}")
# ---- Export des résultats ----
#pd.DataFrame(results).to_csv("battery_simulation_results.csv", index=False)
#print("Simulation terminée et exportée vers battery_simulation_results.csv")
