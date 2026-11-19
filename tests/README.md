# Testing and development

Run from the repository root in an environment installed from `requirements-dev.txt`.

```powershell
& ./.venv/Scripts/python.exe -m pytest --collect-only -q
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_live_entrypoint.py -q -rA -s
& ./.venv/Scripts/python.exe -m pytest tests -m "not slow" -q -ra
& ./.venv/Scripts/python.exe -m pytest tests -m slow -q -ra -s
& ./.venv/Scripts/python.exe -m pytest tests -q -ra
```

Collection discovers tests. Successful execution requires pass/fail/skip results. The `not slow` expression selects markers and does not guarantee a short runtime. Large studies can remain expensive.

## Coverage

| Entry point | Contract |
| --- | --- |
| `test_lab006_live_entrypoint.py` | Actual subprocesses, ARC, diagnostics and spawned sensing |
| `test_lab006_run_profile.py`, `test_lab007_profile_integration.py` | Schema and profile integration |
| `test_arch001*`, `test_mode001*` | Explicit initialization, rollback and state boundaries |
| `test_c001_rabi_contract.py`, `test_beam002_radius_contract.py` | Rabi, dipole and beam-radius conventions |
| `test_num_td001_density_validation.py`, `test_num_ode002_profile.py` | Density certification and solver settings |
| `test_doa_*`, `test_if_*` | Steering, snapshots, covariance and IF extraction |
| `test_doa_peak_refinement.py` | Off-grid angle accuracy, diagnostics and pipeline integration |
| `test_doppler_affine_kernel.py` | Direct-solver parity and reference fallback |
| `test_linewidth_model_comparison.py` | Linewidth operators, single counting and replay |
| `test_noise*`, `test_dop*`, `test_dph*` | Effect, study and transit contracts |

Atomic fixtures in `conftest.py` are opt-in. ARCH-001 keeps imports independent of physical initialization. `atomic_physics` requests real initialization. `published_atomic_state` supplies controlled state to consumer unit tests with synthetic inputs. [Fixtures](conftest.py)

`atomic_init_support.py` converts only the documented QN-001A HFS→FS missing-Rydberg-mJ boundary to `BLOCKED` skips. Other errors propagate. Always report skips separately. This suite depends on its versioned profiles and baselines. `outputs/` remains generated evidence. Synthetic and live numerical tests require independent experimental validation before making hardware claims. [Atomic support](atomic_init_support.py)

## Development entry points

The public route is `main.py` → `cli.main` → `load_run_profile` → `execute_profile` or `execute_study`. Read `config/run_profile.py` and parser tests for new inputs, `physics/` and `simulation/` for solving, and `doa/` with `simulation/doa_pipeline.py` for array processing. `run_simulation_with_config` is a legacy helper. [Architecture](../architecture.md)

Check ordering, trace, Hermiticity, positivity and relevant convergence for solver changes. Preserve sensor order, phase and snapshot/covariance shapes for DoA changes. Run affected behavior and report environmental failures separately. Preserve baseline, seed and acceptance-tolerance contracts. [Development instructions](../AGENTS.md)

## Verified review branches, 2026-11-19

GitHub Actions ran Python 3.11 core tests on Ubuntu and Windows, with GUI tests on Windows. The reviewed branches contain separate numerical fixes and a dependent linewidth comparison.

| Revision | Core on each OS | Windows GUI |
| --- | --- | --- |
| `faa1341`, numerical fixes | 2724 passed, 0 failed, 82 skipped, 12 slow deselected | 34 passed |
| `2f8a575`, combined changes | 2728 passed, 0 failed, 82 skipped, 12 slow deselected | 34 passed |

The 12 slow tests ran separately on local Windows with Python 3.11.0 and QuTiP 5.2.1. Seven passed and five skipped at QN-001A. Pytest also reported four collection skips from obsolete modules, producing a command summary of 7 passed, 9 skipped and 2806 deselected. Those collection skips overlap with CI and must be counted once when combining coverage. [Fixes CI](https://github.com/MrMalekfar/RAQS/actions/runs/44521453340) · [Combined CI](https://github.com/MrMalekfar/RAQS/actions/runs/44521511673)

The local duplicate full-suite run stopped at about 63% after Windows CI completed, with no failure observed up to that point. It has no complete result. An initial sandbox-only targeted run had 73 passes and 12 ARC read-only-database failures. Final verification used writable ARC data. `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` were used for successful local sensing and numerical checks. GUI verification used `QT_QPA_PLATFORM=offscreen`.

The README help, diagnostics and sensing commands completed. Sensing stored `time_traces` with shape `(3, 601)`, `if_snapshots` with shape `(3, 2)` and covariance `(3, 3)`. `pip check` and `git diff --check` succeeded. Both 41-point `lab007` linewidth comparisons were reproduced on Windows at the profile linewidth and 0.5 MHz. Maximum spectral difference from the supplied review data was `1.1e-16`, and IF replay error was zero in all four cases.

## Earlier documentation verification, 2026-11-08

The existing Windows/Python 3.11.0 environment passed `pip check` and a requirements dry run. Pytest collected 2742 tests. Clean installation and other platforms were outside that historical check. A sandbox attempt failed before collection on ARC's read-only SQLite database. Another captured run was interrupted after three passes and has no complete result. Running the five live entry-point tests with writable ARC and `-s` produced 5 passes, 0 failures and 0 skips.

The profile integration selection below produced 85 passes, 0 failures and 0 skips.

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_run_profile.py tests/test_lab007_profile_integration.py -q -ra
```

Both README commands and `--help` succeeded. Array shapes were `(3, 601)`, `(3, 2)` and `(3, 3)`. Local links and whitespace were checked. The full suite and `not slow` selection were outside that historical 90-test verification. QuTiP FutureWarnings were recorded at that time.
