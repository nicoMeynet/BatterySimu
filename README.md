# Battery Simulator

## Overview
This project is a Python-based battery simulator that models the behavior of a 3-phase battery system in a residential setup. It calculates energy usage, solar production, and battery behavior to optimize energy consumption and reduce costs. The simulation includes:

- Handling of solar energy consumption.
- Tracking battery charge and discharge cycles.
- Integration of time-of-use tariffs (peak and off-peak hours).
- Simulation of energy injected into or consumed from the grid.

In recent years, plugin battery systems have become increasingly popular due to their ease of installation—requiring no electrician for setup. However, evaluating whether these systems are economically beneficial remains a challenge. The goal of this script is to help users make informed decisions by simulating and analyzing the financial impact of such systems.

Example of plug and play battery: "Zendure hyper 2000" ou "Sunology Storey"

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

| Phase                     | Current (kWh) | Simulated (kWh) | Delta (kWh) | Delta (CHF) |
|---------------------------|---------------|-----------------|-------------|-------------|
| Phase A Injected Off-Peak | 50            | 12              | -37         | 3           |
| Phase A Injected Peak     | 0             | 0               | 0           | 0           |
| Phase B Injected Off-Peak | 57            | 17              | -40         | 4           |
| Phase B Injected Peak     | 0             | 0               | 0           | 0           |
| Phase C Injected Off-Peak | 59            | 17              | -41         | 4           |
| Phase C Injected Peak     | 0             | 0               | 0           | 0           |
| **Total Injected**        | **166**       | **46**          | **-118**    | **11**      |

#### Consumed Energy

| Phase                     | Current (kWh) | Simulated (kWh) | Delta (kWh) | Delta (CHF) |
|---------------------------|---------------|-----------------|-------------|-------------|
| Phase A Consumed Off-Peak | 108           | 85              | -22         | 7           |
| Phase A Consumed Peak     | 37            | 28              | -9          | 3           |
| Phase B Consumed Off-Peak | 79            | 57              | -21         | 7           |
| Phase B Consumed Peak     | 38            | 24              | -14         | 4           |
| Phase C Consumed Off-Peak | 73            | 51              | -21         | 7           |
| Phase C Consumed Peak     | 37            | 24              | -13         | 4           |
| **Total Consumed**        | **372**       | **269**         | **-100**    | **32**      |

#### Rentability

- **Total gain:** 43 CHF for 15 days or per year: 1046 CHF (extrapolated)
- **Amortization time:** 5.59 years if the cost of the battery is 5847 CHF

#### Battery Statistics

| Metric                | Phase 1 | Phase 2 | Phase 3 | Max/Config |
|-----------------------|---------|---------|---------|------------|
| Cycles                | 8       | 9       | 9       | 6000       |
| Expected life (years) | 27      | 25      | 25      |            |
| Remaining energy (Wh) | 2287    | -17     | 1945    |            |

#### Battery Status

| Status      | Phase 1    | Phase 2    | Phase 3    |
|-------------|------------|------------|------------|
| Full        | 507 (2.20%)| 737 (3.20%)| 662 (2.88%)|
| Empty       | 11785 (51.22%)| 10355 (45.00%)| 9815 (42.66%)|
| Discharging | 6994 (30.40%)| 7851 (34.12%)| 8425 (36.61%)|
| Charging    | 3724 (16.18%)| 4067 (17.67%)| 4108 (17.85%)|
| **Total**   | **23010**  | **23010**  | **23010**  |

#### Charging and Discharging Power at Peak

| Metric                       | Phase 1        | Phase 2        | Phase 3        |
|------------------------------|----------------|----------------|----------------|
| Charging at Max Power        | 1014 (27.23%)  | 991 (24.37%)   | 1067 (25.97%)  |
| Charging Not at Max Power    | 2710 (72.77%)  | 3076 (75.63%)  | 3041 (74.03%)  |
| Discharging at Max Power     | 1402 (7.47%)   | 1035 (5.69%)   | 0 (0.00%)      |
| Discharging Not at Max Power | 17360 (92.53%) | 17146 (94.31%) | 23010 (100.00%)|


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
