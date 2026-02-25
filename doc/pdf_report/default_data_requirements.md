To ensure accurate and meaningful results, the simulation requires:

- Historical household consumption measurements without a battery installed
- Grid import and export power data
- 3-phase measurements (A, B, C) with timestamp granularity
- A defined battery configuration for each tested scenario

Measurements can be collected using:

- A 3-phase energy meter such as Shelly 3EM
- Integration with Home Assistant for data logging and export

The simulator uses this real-world baseline to perform controlled charge/discharge modeling under defined tariff rules.

Accurate input data is essential, as simulation quality directly depends on measurement quality.
