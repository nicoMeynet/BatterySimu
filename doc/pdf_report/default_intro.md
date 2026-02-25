This report evaluates multiple residential battery configurations to determine the optimal storage size for the household. The analysis is based on real 3-phase grid measurements collected prior to battery installation, ensuring that all results reflect actual consumption and injection behavior.

Using timestamp-level import and export data, the simulator models tariff-aware charge and discharge decisions under defined battery constraints. Each configuration is evaluated against an identical no-battery baseline, enabling a fair and consistent comparison.

To generate meaningful results, the user must provide:
- Historical grid import and export measurements (without battery)
- 3-phase power data with adequate time resolution
- The battery configurations to be tested
- Applicable tariff parameters

Measurements can be obtained using a 3-phase energy meter (for example Shelly 3EM) integrated with Home Assistant for data collection and export.

The objective is to support a data-driven investment decision by balancing financial return, energy adequacy, and power limitations, selecting the smallest configuration that delivers robust and sustainable performance across the full year.
