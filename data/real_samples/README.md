# Real PV Data Samples

This folder stores small real photovoltaic data samples used for project research.

## Current Sample

- File: `pvdaq_system_10_2023_01_01.csv`
- Source: NREL / DOE PVDAQ public dataset
- System: PVDAQ system 10, Golden, Colorado, USA
- Date: 2023-01-01
- Resolution: 1 minute
- Rows: 1,440

Important columns:

- `measured_on`: timestamp
- `ac_power__423`: AC power
- `dc_power__422`: DC power
- `ambient_temp__428`: ambient temperature
- `module_temp_1__429`: module temperature
- `poa_irradiance__421`: plane-of-array irradiance

The project still uses `data/clean_dataset.csv` for baseline training because the real sample needs additional field mapping and may not include all current demo features such as humidity and wind speed.
