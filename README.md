# Battery Simulator

## Overview
The Battery Simulator is a Python-based tool designed to optimize residential energy consumption using a 3-phase battery system. By modeling battery behavior and calculating energy flows, the simulator provides insights into energy usage, cost savings, and the financial viability of battery systems in real-world scenarios. This tool empowers homeowners to make data-driven decisions about integrating battery systems into their energy setups, including determining the optimal sizing of the system, the number of batteries per phase, and the charging and discharging power required per phase.

Key benefits include:
- **Technical:** Simulates energy flows and battery performance with precision.
- **Financial:** Provides detailed analysis of cost savings and rentability.
- **Operational:** Easily integrates with Home Assistant CSV data for seamless modeling.

### Why Plug-and-Play Batteries?
In recent years, plug-and-play battery systems have surged in popularity due to their ease of installation—no electrician required. Technological advancements and decreasing costs have made these systems more accessible for residential use. However, understanding their economic feasibility remains challenging.

#### Examples of plug-and-play batteries:
- **Zendure Hyper 2000:** Known for its portability and user-friendly setup.
- **Sunology Storey:** Offers scalability and efficient energy storage for small households.

The Battery Simulator aims to simplify this evaluation process by analyzing the potential savings, operational impact, and optimal configuration of these systems.

## Features
### Technical Features
- Models a 3-phase battery system with configurable capacity and power.
- Handles time-based tariffs for energy consumption and injection.
- Provides detailed statistics on energy usage and battery behavior.
- Includes preprocessing of input data with timestamp interpolation.

### Financial Features
- Calculates battery efficiency and lifecycle.
- Estimates cost savings and rentability of battery systems.
- Accounts for peak and off-peak energy tariffs to optimize savings.

### Operational Features
- Supports CSV-based input from Home Assistant for seamless integration.
- Offers configurability for battery parameters and tariff structures.
- Simulates charging and discharging cycles for real-world scenarios.

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

### Example Output with a dataset of 337 days
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

#### Example Configuration
Below is an example of configuring the script for a 10 kWh battery system:
```python
battery_capacity_Wh = [3940, 3940, 3940]        # Battery capacity per phase (Wh)
max_charge_power_watts = [1200, 1200, 1200]     # Max charge power per phase (W)
max_discharge_power_watts = [1200, 1200, 1200]  # Max discharge power per phase (W)
battery_charge_efficiency = 0.9                 # Charge efficiency (90%)
battery_discharge_efficiency = 0.9              # Discharge efficiency (90%)
battery_max_cycles = 6000                       # Battery lifespan in cycles
battery_cost = 5847                             # Battery cost (CHF)
```

### Tariff Configuration
- `tariff_consume`: Cost of consuming energy (CHF/kWh).
- `tariff_inject`: Cost of injecting energy (CHF/kWh).
- `days`: Days for peak tariffs (Monday to Friday).
- `hours`: Hours for peak tariffs (17:00 to 22:00).

#### Example Tariff Configuration
```python
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
```

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Acknowledgments
This project is inspired by efforts to optimize residential energy usage with renewable energy and battery storage systems.
