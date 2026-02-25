## 1. Input Data
- Real 3-phase household consumption data (Phase A, B, C)
- Timestamp-level power measurements
- Historical seasonal load patterns
- Exported data from Home Assistant
Data granularity ensures realistic modeling of daily and seasonal variability.

## 2. Simulation Engine
- Timestamp-by-timestamp battery charge/discharge decisions
- SOC (State of Charge) tracking with min/max constraints
- Power limitation enforcement (charge and discharge limits)
- Efficiency modeling (charge and discharge losses)
- Priority rules for self-consumption optimization
Battery behavior is evaluated under physical and electrical constraints.

## 3. Tariff Model
- Peak and off-peak consumption tariffs
- Injection tariffs for exported energy
- Net financial gain calculation versus no battery
- Cost comparison per scenario
All financial gains are calculated relative to the no-battery baseline.

## 4. Comparison Baseline
- Identical household load data
- Same solar production profile
- No-battery reference case
- Identical tariff assumptions
This ensures consistent and fair scenario comparison.

## 5. Key Performance Indicators (KPIs)
### Financial Indicators
- Monthly and seasonal net financial gain
- Grid import reduction
- Grid export reduction
### Energy Indicators
- Battery energy throughput
- Equivalent full cycles
- Energy shifting volume
### Structural Sizing Indicators
- Battery full SOC share (oversizing indicator)
- Battery empty SOC share (undersizing indicator)
- Daily structural undersizing
- Evening structural undersizing
### Power Indicators
- Active power saturation
- Idle missed opportunities
- Power state distribution
These indicators together provide a multi-dimensional sizing assessment.
