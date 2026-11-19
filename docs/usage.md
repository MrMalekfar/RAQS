# Running profiles and interpreting outputs

## Selecting and executing a profile

`main.py` and `cli.py` call `cli.main`. `--profile` is required. Legacy switches such as `--n-elements`, `--target-aoa`, `--rf-frequency`, `--lo-rf-frequency`, `--spacing` and `--cell-length` remain visible in help, but combining them with a profile is rejected. Edit a copy of the input JSON.

`load_run_profile` strictly validates schema keys, duplicate keys, finite values and complete declarations. Schema 1 supports prescribed diagnostics and experimental sensing. Schema 2 adds explicit `environment`, `spectrum` and `execution`, including prescribed sensing. `spectrum` may be null. A thermal environment alone leaves temporal Doppler disabled. [Profile parser](../rydberg_aoa/config/run_profile.py)

In `prescribed` mode you declare the ladder. In `experimental` mode the channel is selected from RF, preparation and bounds. Values in `proposed_wavelengths_nm_not_applied` are proposals. A filename containing `reference` can have large optical detuning, as in `prescribed_reference`.

The [main README](../README.md) contains verified starting commands. `--diagnostics-only` skips ordinary sensing workers and DoA and is rejected for studies. Specialized diagnostics may still perform numerical work. `--no-plots` disables CLI figure generation/display. Supply `--output-dir` to save artifacts. Library messages may accompany the printed report on stdout.

## Reading a result

Use a fresh or empty output directory. `profile.json` retains the bytes consumed by the program. `manifest.json` is published last and records input/output SHA-256 values, dependency versions and Git state. Its absence means publication is incomplete. Choose another directory for a retry. Unavailable Git fields are recorded as null.

| Report section | Interpretation |
| --- | --- |
| `atom`, `levels`, `applied_fields` | Species, resolved levels and applied fields to compare against the profile |
| `detunings_rad_per_us`, `rabi_rad_per_us` | Probe/coupling/RF detunings and probe/coupling/LO/signal Rabi frequencies |
| `rf_orientation`, `signed_if_mhz` | RF energy direction and signed signal−LO difference |
| `decay_rate_basis`, `radiative_model` | Rate basis and radiative model scope |
| `diagnostics_only`, `sensing` | Diagnostic status and temporal sensing results |
| `sensing` | True/estimated angle, path/sample counts and effective execution settings |
| `sensing.estimated_aoa_grid_deg`, `sensing.doa_peak_refined` | Original grid peak and refinement status for grid estimators |
| `if_measurement` | IF metrics with their declared definitions and units |
| `noise_status` and `included` flags | Effects actually executed |
| `temporal_ode` | Configured/effective solver settings, source and execution status |

The starting sensing example has noise, Zeeman and Doppler ensembles disabled. `target_azimuth_deg` supplies simulation truth. Agreement under this ideal single-source model establishes phase preservation through the simulated pipeline. Statistical accuracy and experimental agreement require separate evidence.

`save_profile_results` writes single-run arrays to `simulation_arrays.npz`. [Result writer](../rydberg_aoa/simulation/profile_execution.py)

- `tlist` uses µs. `time_traces` is ordered by path, then time.
- `complex_responses` contains one complex response per path. `array_positions` contains matching positions.
- `if_snapshots` has shape sensors × windows. `sample_covariance` is its covariance.
- `fft_frequency_path_*` and `fft_complex_path_*` hold per-path FFT data. Precise IF extraction uses its own fit.
- `angles` and `power` are present for estimators with a spectrum. Search-free methods have their own outputs.
- `spectrum_*`, `trial_*` and detector arrays appear on their associated paths. Read `sensing_trace_units` or the detector receipt for trace units.

Load NPZ files with `np.load` and inspect names and shapes first. IF-only has a separate output contract. [IF-only implementation](../rydberg_aoa/simulation/if_only.py)

## Studies and data retention

Study profiles dispatch to `execute_study`. The study writer stores `study_report.json`, `study_profile.json` and point/trial subdirectories appropriate to the study type. Some paths have separate calibration and null-validation outputs. Existing manifests are protected from mixing with a fresh study. Always choose a fresh destination. Preserve the original input, code/environment version and numerical settings with the result. [Study execution](../rydberg_aoa/simulation/study_execution.py) · [Study writer](../rydberg_aoa/simulation/study_output.py) · [Reproducibility](README.md)

## Troubleshooting

- ARC construction with `readonly database` requires writable user atomic data. Fix access to the declared data source.
- `PhysicsConfigurationError` requires checking declarations, q, mJ, bounds and option compatibility.
- Preserve the exception chain and settings when a solver or worker fails. Numerical tolerances and baselines must follow their documented contracts.
- Runtime grows with paths, samples, velocity classes, study points and trials. Start with a bounded example.
- For OpenBLAS allocation errors, retry with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. These are process-resource settings.

## Profile and noise versions

The top-level profile version and `noise.schema_version` are independent. Tracked executable examples use profile version 2. Version 1 remains readable. In version 2, `environment: null` preserves explicit temporal rates without a thermal environment, `spectrum: null` leaves spectrum calculation unrequested, and a prescribed diagnostics profile may set `execution: null`. Spectrum computation requires an explicit environment and execution. A thermal budget without execution is rejected. The report records `input_schema_version`.

Tracked noise examples use version 4. Versions 1–3 remain readable.

- `enabled` controls the whole noise block. Sources share `seed` and `trials`.
- `drive` is null or contains `settings` and `sources`. Each RF/laser source retains its own `enabled` key and parameters.
- `readout` is null or selects `model: "gaussian"` or `model: "pin"` with the corresponding sources. PIN requires `detector`. Combined drive/readout uses `readout.enabled` independently. Standalone readout uses `noise.enabled`.
- An enabled PIN detector still converts the signal when its noise sources are disabled. Gaussian and PIN are alternative readout models within a profile.

The newer JSON maps to the existing execution models while preserving report formats and random-stream identifiers. [Gaussian example](../examples/noise002/prescribed_gaussian.json) · [Drive with PIN](../examples/noise005/prescribed_rf_pin.json)
