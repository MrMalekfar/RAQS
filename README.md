# RAQS — Rydberg AoA Simulation

RAQS is research software for modeling the RF response of a Rydberg-atom receiver and estimating angle of arrival (AoA) from multiple sensing paths. Its main pipeline connects probe, coupling, LO and signal fields to an atomic model, Lindblad evolution, heterodyne detection and DoA processing. It supports numerical studies of EIT, IF response, array geometry and noise. Experimental validation and hardware accuracy require independent evidence.

## Getting started

On Windows, extract the complete repository ZIP to a writable folder. Install 64-bit Python 3.11 with Python Launcher, then double-click **Launch RAQS GUI.cmd**. The launcher creates the project environment and installs its dependencies. The first launch needs internet access and may take several minutes. [Python for Windows](https://www.python.org/downloads/release/python-3119/) · [Desktop guide](docs/gui.md)

The **Windows installer** GitHub Actions workflow builds a standalone installer. [Build, test and download instructions](docs/gui.md#building-and-testing-the-installer)

### Command line

Run these commands in PowerShell from the repository root with Python 3.11.

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe main.py --help
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/prescribed_reference.json --diagnostics-only --no-plots --output-dir outputs/diagnostics/run01
& ./.venv/Scripts/python.exe main.py --profile examples/lab006/experimental_near_resonant.json --no-plots --output-dir outputs/sensing/run01
```

`requirements.txt` is sufficient for runtime use. On Linux, create the environment with `python3.11` and use `.venv/bin/python`. Core tests run on Windows and Linux in GitHub Actions. GUI tests run on Windows. Direct dependencies are pinned. Transitive dependencies are not fully locked. ARC needs writable atomic data in the user directory.

The first profile performs configuration diagnostics and atomic resolution. The second runs sensing with three paths, 601 samples per path, a 0.1 MHz IF and a target angle of -30°. The refined estimate is approximately -30.000038° in the verified example. This checks the simulation path and phase preservation under its assumptions. It does not establish general receiver accuracy.

Choose a fresh or empty output directory for each profile run. The consumed input is saved as `profile.json`, arrays as `simulation_arrays.npz`, and a final `manifest.json` records successful artifact publication. If OpenBLAS reports allocation errors, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before running. These thread settings were used for local verification.

## Capabilities and scientific scope

- `prescribed` profiles declare a ladder. `experimental` profiles select a channel within hardware bounds. The latter name denotes a simulation mode.
- Four-level prepared-channel response, EIT spectra, IF extraction and array processing. Doppler, transit and noise have independent switches and execution limits.
- Phase beamforming, Bartlett, MVDR, MUSIC, ESPRIT and Root-MUSIC, each with geometry and snapshot requirements. Grid estimators refine the peak between scan points and retain the grid estimate in diagnostics.
- Multipoint studies, noise campaigns, detection and convergence checks. The parser and execution path determine supported combinations.

The public profile pipeline models one RF source at zero elevation. The effective four-level model assumes a prepared magnetic channel. General optical pumping, full multilevel vapor-cell dynamics and dynamical isotope mixtures require additional modeling. The magnetic-core path provides specialized diagnostics. Check each run's declared scope before interpreting it as a receiver result.

Undeclared temporal ODE controls use `rtol=1e-10`, `atol=1e-12` and `nsteps=5000`. Explicit profile controls take precedence. The field phase-diffusion linewidth comparison is a separate analysis tool. The main pipeline retains its existing per-level linewidth model. [Research guide](docs/README.md) · [Execution and outputs](docs/usage.md) · [Example profiles](examples/README.md)

## Tests and development

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_lab006_live_entrypoint.py -q -rA -s
```

See the [test guide](tests/README.md) for wider checks, CI results and skip semantics. Start development with the [architecture](architecture.md) and [development instructions](AGENTS.md).

This repository contains code, tests and example profiles. Historical research tools and thesis inputs have separate provenance. The software is licensed under [MIT](LICENSE).
