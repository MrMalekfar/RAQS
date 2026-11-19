# Execution architecture

The public entry point `main.py` calls `cli.main()`. The CLI loads a JSON profile through `config.run_profile.load_run_profile()` and dispatches to `simulation.profile_execution.execute_profile()` or, for a multipoint study, `simulation.study_execution.execute_study()`.

| Package | Responsibility |
| --- | --- |
| `config/` | Profile contracts, configuration and explicit physical initialization |
| `arc/` | Atomic data and declared transitions |
| `physics/` | Hamiltonians, collapse operators, Doppler and transmission |
| `simulation/` | Solvers, orchestration, detection and multipoint studies |
| `doa/` | Geometry, snapshots, covariance and estimators |
| `signal_processing/` and `tools/` | Helpers used by the main package |
| `visualization/` | Figures generated from results |
| `analytical/` and `level_finder/` | Analytical tools and level selection |

All package paths in this table are under `rydberg_aoa/`. Profile parsers and executors validate effect settings and their compatibility. Invalid channels, models and initialization requests fail explicitly.

Ordinary inputs are declared in profiles. `outputs/` contains generated results. Active tests use declared fixtures independently of historical runs. The two root baselines support regression contracts and must retain their original values when investigating failures.

Temporal workers certify density matrices before consuming observables. The direct steady-state solver and the adaptive Doppler affine kernel share the trace-constrained solve and NUM-SS-001 certification. The optimized Doppler path compares reference coherences at every scan point before using the kernel. DoA grid estimators retain the discrete peak in diagnostics alongside the refined estimate.
