#! /usr/bin/python3
import pandas as pd
import sys

# ---- Paramètres de la batterie ----
battery_capacity_kwh = [10, 10, 10]  # Capacité de la batterie par phase (kWh)
max_charge_power_kw = [3, 3, 3]      # Puissance max de charge par phase (kW)
max_discharge_power_kw = [3, 3, 3]   # Puissance max de décharge par phase (kW)
efficiency = 0.9                     # Rendement (90%)
soc_min = 20                         # Capacité de décharge minimale (%) (State of Charge)
battery_cycles = 5000                # Durée de vie de la batterie en cycles

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
energy_in_battery = [battery_capacity_kwh[i] * (battery_soc[i] / 100) for i in range(3)]

# ---- Fonction pour déterminer le tarif actuel ----
def get_current_tarif(hour, day):
    if day in tarif_config["pleines"]["jours"] and hour in tarif_config["pleines"]["heures"]:
        return tarif_config["pleines"]["tarif"]
    return tarif_config["creuses"]["tarif"]

# ---- Simulation ----
def simulate_battery_behavior(prod_phase, cons_phase, hour, day):
    global energy_in_battery, battery_soc
    tarif = get_current_tarif(hour, day)

    for phase in range(3):
        surplus = prod_phase[phase] - cons_phase[phase]

        if surplus > 0:
            # Charger la batterie avec le surplus
            charge_power = min(surplus, max_charge_power_kw[phase])
            energy_in_battery[phase] += charge_power * efficiency * (1 / 3600)
        else:
            # Décharger la batterie si déficit
            deficit = abs(surplus)
            discharge_power = min(deficit, max_discharge_power_kw[phase])
            if energy_in_battery[phase] > (battery_capacity_kwh[phase] * (soc_min / 100)):
                energy_in_battery[phase] -= discharge_power * (1 / efficiency) * (1 / 3600)

        # Calcul de l'état de charge (SoC)
        battery_soc[phase] = (energy_in_battery[phase] / battery_capacity_kwh[phase]) * 100
        battery_soc[phase] = max(soc_min, min(100, battery_soc[phase]))

    return {
        "phase1": {
            "SOC": battery_soc[0],
            "Energy_in_battery_kWh": energy_in_battery[0]
        },
        "phase2": {
            "SOC": battery_soc[1],
            "Energy_in_battery_kWh": energy_in_battery[1]
        },
        "phase3": {
            "SOC": battery_soc[2],
            "Energy_in_battery_kWh": energy_in_battery[2]
        },
        "Tarif": tarif
    }

def validate_timestamp_continuity(data, max_delta_time=60):
    # Calculate the difference between consecutive timestamps
    data["timestamp_diff"] = data["timestamp"].diff().dt.total_seconds()

    # Identify missing timestamps
    missing_timestamps = data[data["timestamp_diff"] > max_delta_time]

    # Calculate the number of missing timestamps and the longest delta time
    num_missing_timestamps = len(missing_timestamps)
    longest_delta_time = missing_timestamps["timestamp_diff"].max()

    # Drop the temporary column used for validation
    data.drop(columns=["timestamp_diff"], inplace=True)

    return num_missing_timestamps, longest_delta_time

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

exit(0)

print("Début de la simulation...")
# ---- Exécution sur les données des fichiers CSV ----
results = []
for i in range(len(merged_data)):
    timestamp = merged_data.iloc[i]["timestamp"]
    day = timestamp.weekday()
    hour = timestamp.hour
    print(f"DEBUG: timestamp={timestamp}, day={day}, hour={hour}")

    production = [merged_data.iloc[i]["solar_production"] / 3] * 3

    #consumption = [merged_data.iloc[i][f"phase{j+1}"] for j in range(3)]

    #result = simulate_battery_behavior(production, consumption, hour, day)
    #results.append({"timestamp": timestamp, **result})

# ---- Export des résultats ----
#pd.DataFrame(results).to_csv("battery_simulation_results.csv", index=False)
#print("Simulation terminée et exportée vers battery_simulation_results.csv")
