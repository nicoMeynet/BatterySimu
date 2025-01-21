# Battery Simulator

## Overview
This project is a Python-based battery simulator that models the behavior of a 3-phase battery system in a residential setup. It calculates energy usage, solar production, and battery behavior to optimize energy consumption and reduce costs. The simulation includes:

- Handling of solar energy consumption.
- Tracking battery charge and discharge cycles.
- Integration of time-of-use tariffs (peak and off-peak hours).
- Simulation of energy injected into or consumed from the grid.

In recent years, plugin battery systems have become increasingly popular due to their ease of installation—requiring no electrician for setup. However, evaluating whether these systems are economically beneficial remains a challenge. The goal of this script is to help users make informed decisions by simulating and analyzing the financial impact of such systems.

Example of plug and play battery: Zendure hyper 2000

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
python battery_simulator.py <house_phase_a.csv> <house_phase_b.csv> <house_phase_c.csv>
```

### Arguments
- `<house_phase_a.csv>`: CSV file containing energy consumption data for Phase A.
- `<house_phase_b.csv>`: CSV file containing energy consumption data for Phase B.
- `<house_phase_c.csv>`: CSV file containing energy consumption data for Phase C.

### Example
```bash
python battery_simulator.py phase_a.csv phase_b.csv phase_c.csv

```

## Input File Format
The CSV file containing the measurements is sourced from Home Assistant, which collects energy consumption data for each phase using a Shelly 3EM module.
Each input CSV file should have the following structure:

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
1. **Simulation Results**: Detailed energy statistics printed to the console, including:
   - Energy injected and consumed during peak (HP) and off-peak (HC) hours.
   - Delta values for energy usage and associated cost differences.
   - Rentability of the battery
   - Battery lifecycle statistics.

### Example Output for 337 days
#### Injected Energy
```plaintext
Injected Energy:
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase                     |   Current (kWh) |   Simulated (kWh) |   Delta (kWh) |   Delta (CHF) |
+===========================+=================+===================+===============+===============+
| Phase A Injected Off-Peak |            1782 |               843 |          -939 |            93 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase A Injected Peak     |               0 |                 0 |             0 |             0 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase B Injected Off-Peak |            2169 |              1315 |          -854 |            85 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase B Injected Peak     |               0 |                 0 |             0 |             0 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase C Injected Off-Peak |            2374 |              1620 |          -753 |            75 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase C Injected Peak     |               3 |                 3 |             0 |             0 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Total Injected            |            6328 |              3781 |         -2546 |           253 |
+---------------------------+-----------------+-------------------+---------------+---------------+
```

#### Consumed Energy
```plaintext
Consumed Energy:
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase                     |   Current (kWh) |   Simulated (kWh) |   Delta (kWh) |   Delta (CHF) |
+===========================+=================+===================+===============+===============+
| Phase A Consumed Off-Peak |            1424 |               953 |          -471 |           160 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase A Consumed Peak     |             495 |               202 |          -293 |            99 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase B Consumed Off-Peak |             891 |               449 |          -441 |           150 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase B Consumed Peak     |             363 |               111 |          -251 |            85 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase C Consumed Off-Peak |             708 |               303 |          -404 |           137 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Phase C Consumed Peak     |             299 |                89 |          -209 |            71 |
+---------------------------+-----------------+-------------------+---------------+---------------+
| Total Consumed            |            4180 |              2107 |         -2069 |           702 |
+---------------------------+-----------------+-------------------+---------------+---------------+
```

#### Rentability
- **Total gain**: 955 CHF for 337 days or per year: 1034 CHF (extrapolated)
- **Amortization time**: 5.80 years if the cost of the battery is 6000 CHF

#### Battery Statistics
- **Cycles**: phase 1: 215, phase 2: 195, phase 3: 172
- **Expected life based on cycles**: phase 1: 21 years, phase 2: 23 years, phase 3: 26 years
- **Remaining Energy**: phase 1: -2 Wh, phase 2: 2870 Wh, phase 3: 0 Wh

## Configuration
The following parameters can be configured in the script:

### Battery Parameters
- `battery_capacity_Wh`: Battery capacity per phase.
- `max_charge_power_watts`: Maximum charge power per phase.
- `max_discharge_power_watts`: Maximum discharge power per phase.
- `battery_charge_efficiency`: Charge efficiency (default: 90%).
- `battery_discharge_efficiency`: Discharge efficiency (default: 90%).
- `battery_max_cycles`: Maximum battery cycles (default: 5000).
- `battery_cost`: Total cost of the battery (CHF).

### Tariff Configuration
- `tariff_consume`: Cost of consuming energy (CHF/kWh).
- `tariff_inject`: Cost of injecting energy (CHF/kWh).
- `days`: Days for peak tariffs (Monday to Friday).
- `hours`: Hours for peak tariffs (17:00 to 22:00).

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Acknowledgments
This project is inspired by efforts to optimize residential energy usage with renewable energy and battery storage systems.
