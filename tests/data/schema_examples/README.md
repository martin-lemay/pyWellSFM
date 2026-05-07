# Schema Examples

This folder contains loadable examples used by `tests/test_schema_examples_io.py`.

## Mapping: schema -> example files

- `jsonSchemas/AccumulationModelSchema.json`
  - `accumulation_model_example.json`
  - CSV-reference variant: `accumulation_model_curve_ref_csv_example.json` (uses `curve_water_depth_reduction.csv`)
- `jsonSchemas/CurveSchema.json`
  - `curve_example.json`
  - CSV variant: `curve_age_value.csv`
- `jsonSchemas/DepositionalEnvironmentModelSchema.json`
  - `depositional_environment_model_example.json`
- `jsonSchemas/DESimulationSchema.json`
  - `desimulation_example.json`
- `jsonSchemas/EnvironmentConditionModelSchema.json`
  - `environment_condition_model_example.json`
  - CSV-reference variant: `environment_condition_model_curve_ref_csv_example.json` (uses `curve_water_depth_reduction.csv`)
- `jsonSchemas/EnvironmentConditionsModelSchema.json`
  - `environment_conditions_model_example.json`
  - CSV-reference variant: `environment_conditions_model_curve_ref_csv_example.json` (uses `curve_water_depth_reduction.csv`)
- `jsonSchemas/FaciesModelSchema.json`
  - `facies_model_example.json`
- `jsonSchemas/FSSimulationDataSchema.json`
  - `fssimulation_data_example.json`
- `jsonSchemas/RealizationDataSchema.json`
  - `realization_data_example.json`
  - CSV-reference variant: `realization_data_curve_ref_csv_example.json` (uses `curve_age_value.csv`)
- `jsonSchemas/ScenarioSchema.json`
  - `scenario_example.json`
  - CSV-reference variant: `scenario_curve_ref_csv_example.json` (uses `curve_age_value.csv`)
- `jsonSchemas/TabulatedFunctionSchema.json`
  - `tabulated_function_example.json`
  - CSV variant: `tabulated_function.csv`
- `jsonSchemas/UncertaintyCurveSchema.json`
  - `uncertainty_curve_example.json`
  - CSV variant: `uncertainty_curve.csv`
- `jsonSchemas/WellSchema.json`
  - `well_example.json`
  - CSV-reference variant: `well_ref_csv_example.json` (uses `striplog_lithology.csv`, `well_continuous_logs.csv`)

## Support files

Shared CSV helper files used by reference variants:

- `curve_water_depth_reduction.csv`
- `curve_age_value.csv`
- `striplog_lithology.csv`
- `well_continuous_logs.csv`
