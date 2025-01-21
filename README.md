# Battery Simulator

## Overview
This project is a Python-based battery simulator that models the behavior of a 3-phase battery system in a residential setup. It calculates energy usage, solar production, and battery behavior to optimize energy consumption and reduce costs. The simulation includes:

- Handling of solar energy production and consumption.
- Tracking battery charge and discharge cycles.
- Integration of time-of-use tariffs (peak and off-peak hours).
- Simulation of energy injected into or consumed from the grid.

## Features
- Models a 3-phase battery system with configurable capacity and power.
- Calculates battery efficiency and lifecycle.
- Handles time-based tariffs for energy consumption and injection.
- Provides detailed statistics on energy usage and battery behavior.

## Requirements
- Python 3.7+
- Required Python libraries:
  - `pandas`
  - `tabulate`

Install the required libraries using:
```bash
pip install pandas tabulate
```

## Usage
Run the script with the following command:
```bash
python battery_simulator.py <solar_csv> <house_phase_a.csv> <house_phase_b.csv> <house_phase_c.csv>
```

### Arguments
- `<solar_csv>`: CSV file containing solar production data.
- `<house_phase_a.csv>`: CSV file containing energy consumption data for Phase A.
- `<house_phase_b.csv>`: CSV file containing energy consumption data for Phase B.
- `<house_phase_c.csv>`: CSV file containing energy consumption data for Phase C.

### Example
```bash
python battery_simulator.py solar.csv phase_a.csv phase_b.csv phase_c.csv
```

## Input File Format
Each input CSV file should have the following structure:

### Solar CSV
| entity_id                                 | state | last_changed           |
|------------------------------------------|-------|------------------------|
| sensor.envoy_482310100274_production_d_electricite_actuelle | 0     | 2024-12-31T23:00:00Z |
| sensor.envoy_482310100274_production_d_electricite_actuelle | 0     | 2025-01-01T00:00:00Z |

### Phase CSV (A, B, C)
| entity_id                                    | state       | last_changed           |
|---------------------------------------------|-------------|------------------------|
| sensor.shellyem3_244cab435bf4_channel_a_power | 128.517028  | 2024-12-31T23:00:00Z |
| sensor.shellyem3_244cab435bf4_channel_a_power | 128.570356  | 2025-01-01T00:00:00Z |

### Preprocessing
The script performs the following preprocessing steps:
- Cleans and renames columns.
- Groups data by timestamp.
- Fills missing timestamps with interpolated values.

## Outputs
1. **Merged Data File**: A CSV file (`merged_data.csv`) containing merged solar and consumption data.
2. **Simulation Results**: Detailed energy statistics printed to the console, including:
   - Energy injected and consumed during peak (HP) and off-peak (HC) hours.
   - Delta values for energy usage and associated cost differences.
   - Battery lifecycle statistics.

### Example Output
#### Injected Energy
```plaintext
Injected Energy:
+---------------------+----------------+------------------+--------------+---------------+
| Phase               |   Current (Wh) |   Simulated (Wh) |   Delta (Wh) |   Delta (CHF) |
+=====================+================+==================+==============+===============+
| Phase A Injected HC |        50741.2 |          12809.3 |     -37931.9 |       3.79319 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase A Injected HP |            0   |              0   |          0   |       0       |
+---------------------+----------------+------------------+--------------+---------------+
| Phase B Injected HC |        57892.9 |          17238.8 |     -40654.2 |       4.06542 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase B Injected HP |            0   |              0   |          0   |       0       |
+---------------------+----------------+------------------+--------------+---------------+
| Phase C Injected HC |        59030.6 |          17433   |     -41597.6 |       4.15976 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase C Injected HP |            0   |              0   |          0   |       0       |
+---------------------+----------------+------------------+--------------+---------------+
| Total Injected      |       167665   |          47481.1 |    -120184   |      12.0184  |
+---------------------+----------------+------------------+--------------+---------------+
```

#### Consumed Energy
```plaintext
Consumed Energy:
+---------------------+----------------+------------------+--------------+---------------+
| Phase               |   Current (Wh) |   Simulated (Wh) |   Delta (Wh) |   Delta (CHF) |
+=====================+================+==================+==============+===============+
| Phase A Consumed HC |       108909   |          85912.4 |    -22996.4  |       7.81876 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase A Consumed HP |        37827.6 |          28609.3 |     -9218.28 |       3.13422 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase B Consumed HC |        79603.8 |          57703.4 |    -21900.4  |       7.44612 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase B Consumed HP |        38817.5 |          24239.1 |    -14578.5  |       4.95668 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase C Consumed HC |        73280.5 |          51744.3 |    -21536.2  |       7.32231 |
+---------------------+----------------+------------------+--------------+---------------+
| Phase C Consumed HP |        37827   |          23630.2 |    -14196.8  |       4.82691 |
+---------------------+----------------+------------------+--------------+---------------+
| Total Consumed      |       376265   |         271839   |   -104426    |      35.505   |
+---------------------+----------------+------------------+--------------+---------------+
```

#### Battery Statistics
- **Cycles**: Phase 1: 8, Phase 2: 9, Phase 3: 9
- **Expected Life**:
  - Phase 1: 23 years
  - Phase 2: 20 years
  - Phase 3: 20 years
- **Remaining Energy**:
  - Phase 1: 2284 Wh
  - Phase 2: -3 Wh
  - Phase 3: 1674 Wh

## Configuration
The following parameters can be configured in the script:

### Battery Parameters
- `battery_capacity_Wh`: Battery capacity per phase.
- `max_charge_power_watts`: Maximum charge power per phase.
- `max_discharge_power_watts`: Maximum discharge power per phase.
- `battery_charge_efficiency`: Charge efficiency (default: 90%).
- `battery_discharge_efficiency`: Discharge efficiency (default: 90%).
- `battery_max_cycles`: Maximum battery cycles (default: 5000).
- `battery_cost`: Cost of the battery (CHF).

### Tariff Configuration
- `tarif_consume`: Cost of consuming energy (CHF/kWh).
- `tarif_inject`: Cost of injecting energy (CHF/kWh).
- `jours`: Days for peak tariffs (Monday to Friday).
- `heures`: Hours for peak tariffs (17:00 to 22:00).

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Acknowledgments
This project is inspired by efforts to optimize residential energy usage with renewable energy and battery storage systems.
