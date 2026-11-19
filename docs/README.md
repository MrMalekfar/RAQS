# Research guide

[Getting started](../README.md) · [Execution and outputs](usage.md) · [Examples](../examples/README.md) · [Tests](../tests/README.md) · [Architecture](../architecture.md)

The [Research Workbench guide](gui.md), [architecture and roadmap](gui-plan.md), and [verification report](gui-verification.md) describe the desktop workflow and its limits. Historical audit records retain the commits, dates and measured results of their original reviews.

## Model and interpretation

`create_sensor_model` builds the four-level ground → excited → rydberg1 ↔ rydberg2 model in the rotating-wave approximation with ℏ=1. Lindblad evolution and conversion of probe coherence to transmission feed IF processing. The ordinary model assumes a prepared magnetic channel. Sublevel populations produced by preparation require a separate model. [Hamiltonian source](../rydberg_aoa/physics/hamiltonian/create_sensor_model.py)

QN-001 distinguishes HFS states labeled by F/mF from FS states labeled by j/mJ. Spherical polarization q and the quantization axis are explicit. The declared channel determines the dipole. `fs_centroid` is the optical energy convention of the starting example. An HFS label alone does not establish inclusion of every hyperfine energy shift. [Laboratory contract](../rydberg_aoa/config/laboratory.py) · [Prepared channel](../rydberg_aoa/arc/prepared_channel.py)

ARC supplies energies, dipoles and rates. Rb85, Rb87 and Cs133 are recognized. Omitting `atom` selects pure Rb85. `isotope_fraction` is a declared fraction, without dynamical isotope-mixture evolution. The collision approximation has species-specific limits. ARC failures and invalid channels propagate explicitly. [Species configuration](../rydberg_aoa/config/atom_species.py)

`decay_rate_basis` records the default rate model. The optional `selected_internal_spontaneous_e1` model replaces the selected internal E1 graph. Population return through all omitted electronic states requires additional modeling. Public magnetic-core sensing is rejected by `execute_profile`. Diagnostics can run a requested specialized temporal path and may be computationally expensive. [Radiative model](../rydberg_aoa/config/profile_radiative.py) · [Profile execution](../rydberg_aoa/simulation/profile_execution.py)

## Units and conventions

| Quantity | Convention |
| --- | --- |
| Time | µs for `duration_us` and `tlist` |
| Input RF frequency and laser wavelength | Hz and nm |
| Laser offset and IF | Cyclic MHz, with signed IF = (signal − LO)/10⁶ |
| Solver Rabi frequency and detuning | rad/µs, converted from Hz as 2πf/10⁶ |
| Event, decay and transit rates | µs⁻¹, converted from s⁻¹ by division by 10⁶ |
| Power, field and distance | mW, V/m and m for sensor positions and cell length |
| `beam_waist_mm` | Gaussian 1/e² intensity radius in mm |

Check each function independently. Angular linewidth and event rate have distinct conversion rules. Transit resets populations and coherences through dedicated collapse operators and must remain separate from pure dephasing. Read `rf_orientation` for RF energy direction and preserve the declared detuning signs and Hamiltonian factor of 1/2. [Transit operators](../rydberg_aoa/physics/lindblad/create_transit_reset_collapse_ops.py)

Azimuth runs from +y toward +x, with horizontal direction `(sin az, cos az, 0)`. Arrival phase is k·r and steering is exp(-j k·r). Snapshot matrices have shape `(sensors, windows)` and rows follow `positions_m`. The final incomplete window is discarded. Window length uses |IF| and projection retains its sign. ESPRIT and Root-MUSIC require compatible ULA geometry. The public pipeline models one RF source at zero elevation. Grid estimators refine a stationary point between neighboring scan angles and retain `grid_azimuth_deg` in diagnostics. [Steering](../rydberg_aoa/doa/steering.py) · [Snapshots](../rydberg_aoa/doa/snapshots.py) · [Profile options](../rydberg_aoa/config/doa_options.py)

## Effects and reproducibility

Spectral and temporal Doppler are independent. Inspect `doppler_ensemble_included` and `temporal_doppler_ensemble_included` separately. In `lab006`, IF/DoA has no Doppler averaging. In `lab007`, the stationary spectrum includes Doppler while IF/DoA does not. Readout, PIN, RF drive, laser-frequency noise and RIN have separate execution paths. `noise_status` and execution receipts record what actually ran. A reference spectrum can differ in noise content from a temporal trial.

Undeclared temporal ODE controls use `rtol=1e-10`, `atol=1e-12`, `nsteps=5000`, with output normalization disabled. Explicit controls take precedence. Density certification thresholds remain independent of solver error requests. Numerical convergence still requires refinement studies.

The main four-level pipeline uses the existing per-level linewidth budget. The independent comparison script evaluates field phase diffusion through `create_laser_phase_diffusion_ops`, removing optical linewidth from the per-level budget first. The reported comparison covers two `lab007` profiles at 1 kHz and 0.5 MHz. Its IF calculation excludes temporal Doppler and physical readout. Identify the actual thesis profiles before extending those conclusions. [Linewidth comparison](LINEWIDTH-MODEL-COMPARISON-2026-11-17.md)

For every result retain the profile, commit and local diff, Python/package versions, ARC species/version and atomic-data provenance, seeds and `noise_random_plan`, solver settings, time/velocity grids and trial count. Single-profile receipts archive input/artifact hashes, available Git information and major dependency versions. They do not archive the entire ARC database or lock all transitive dependencies. Fixed seeds support replay. Convergence and statistical replication require independent checks.

`outputs/` contains fresh results outside Git. Reference fixtures require provenance, version, units and a hash. Preserve both root regression baselines. Literature manifests define the comparison scope. Experimental validation, general optical pumping and a complete multilevel receiver remain separate research tasks.

## Review records

- [2026-11-12 audit and remediation](AUDIT-2026-11-12.md) records initial findings and their follow-up acceptance.
- [2026-11-17 external review](REVIEW-2026-11-17.md) describes the numerical fixes and historical measurements.
- [Laser linewidth comparison](LINEWIDTH-MODEL-COMPARISON-2026-11-17.md) records model definitions, data and limits.
- [Current test evidence](../tests/README.md#verified-review-branches-2026-11-19) records actual Windows/Linux CI and local slow-test results.
