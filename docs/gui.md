# RAQS Research Workbench

The workbench is a local English-language PySide6 desktop interface over the existing RAQS profile and simulation APIs. It improves configuration, execution visibility and measurement review without changing the physical model, units, baselines, seeds or solver tolerances.

## Install and launch

From the repository root, using the project Python environment:

```powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-gui.txt
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe -m raqs_gui
```

The optional GUI requirements use PySide6-Essentials 6.10.3. CLI-only installations do not need Qt. The application still runs from this checkout; no standalone installer is provided.

## First complete workflow

1. In Configure, use the near-resonant AoA template or Import JSON. A valid example is loaded at startup so the interface is usable without a personal profile. Template values are simulation assumptions, not hardware calibration.
2. Select a scientific section, such as Probe Laser or Array & Acquisition. Edit values directly. Live JSON shows the same current draft. Importing another file replaces the relevant controls instead of retaining old input.
3. Read the eligibility line below the command bar. Run Diagnostics resolves declared atomic inputs; Start Calculation executes the current AoA or IF-only measurement. Ctrl+Enter starts a runnable calculation. A parsed profile has not necessarily passed ARC or solver execution.
4. In Run monitor, choose the output parent folder before starting. Each run receives a new independent subfolder and a byte-exact input snapshot. During execution the configuration is locked, while the monitor and other pages remain accessible.
5. Results opens on Figures when a scientific image is available; Summary and Technical report are adjacent tabs. Open Saved Run verifies the saved artifacts before displaying them. Open Result Folder exposes the complete data for further research.

Use the Cs133 IF-only template for a single-path IF example. The summary explicitly says DoA was not requested. For a prescribed diagnostics-only template, calculation is disabled because no sensing execution is declared; diagnostics remains available.

## Configuration and optional effects

The section selector organizes fields by scientific task. Values accept decimal/scientific notation without display rounding entering the run data. Labels distinguish F/mF from j/mJ and show declared units. Numeric syntax errors appear beside their control; scientific incompatibilities are reported by the core parser in the visible message and eligibility area.

Add / replace effect lists sections and explicit starting values from the current examples. You select both the section and its source example. Replacing an existing section requires confirmation. These presets do not establish scientific compatibility with the remaining configuration. Remove effect deletes the chosen optional section, or sets a required version-2 spectrum declaration to null. Invalid combinations remain drafts until corrected.

Numeric matrices support Duplicate row and Remove row. Vectors support Duplicate value and Remove value without silently losing entries. Advanced structure provides native tree editing of every JSON value, array entry and object member, including optional/null/composite structures that do not yet have a dedicated form. No source JSON typing is required. The parser continues to decide valid types, names and combinations.

If atom is absent, the form shows the effective legacy species/fraction without silently adding an atom block. Intentionally changing it makes the declaration explicit. Changing species never retunes fields or rewrites the prepared channel.

Save As creates a new file and refuses an existing destination. An untouched import retains its original bytes. After editing, serialization may change whitespace but preserves unedited data. Reload, replacement and closing protect unsaved edits with a discard/cancel prompt. The live JSON is an inspector, not a separate competing editor.

## Research documents are not run profiles

A design.json, manifest or source guard may describe a research workflow instead of declaring fields/mode. Such documents remain visible in preview but are not converted to a different model. Choose referenced profile is enabled when a declared referenced input exists. Missing historical tools, outputs or inputs are reported rather than replaced.

## Execution semantics

Each worker is a fresh Python process using execute_profile and the existing array/figure writers. Qt never sets global atomic parameters directly. The worker is allowed to create the production spawn workers. The input is fixed before computation and a separate folder is used for every job.

The monitor shows starting/running/saving and actual elapsed wall time. It does not estimate remaining time or manufacture a completion percentage. Follow latest output is enabled by default; uncheck it to inspect an earlier position without being pulled back to the bottom. The compact command bar stays visible, while eligibility and messages are at the bottom.

On Windows, Stop Calculation (Ctrl+Shift+X) force-stops the current owned worker process group, including its calculation descendants. A startup handshake establishes ownership before any physics initialization. If the GUI cannot establish ownership, computation does not start. This is process termination, not a graceful solver cancellation or a partial scientific result. Stopping remains an active state until the owned group is empty and its launcher has exited; the window cannot close during cleanup. Interrupted files are retained with interrupted.json and are rejected by the result reader, even if a completion manifest was written in a finish/stop race. A metadata-write failure is reported explicitly; do not interpret that folder as a completed result. Verified Stop is Windows-only; on other platforms it is disabled. Artifact verification is not a calculation and is not cancelled by Stop.

Worker, solver, rendering and storage errors are failures. A complete manifest is published only after the required artifacts are written. A report-write failure does not become a successful partial measurement. The log is preserved on disk; the monitor displays its recent tail.

Study and magnetic-core profiles can be imported, edited and saved, but their GUI execution is deliberately disabled with a visible reason. Ordinary AoA and IF-only execution are supported. A control being editable does not mean every combination is scientifically executable.

## Reading and exporting results

The Summary table records the receiver mode, species, IF, acquisition shape, estimator where applicable, active effects, rate basis and declared scientific scope. AoA estimates are shown only for AoA measurements. No angular spectrum is synthesized for a search-free estimator.

Figures displays the PNGs made by the existing scientific plotting functions. Fit Page is the default and keeps the whole figure visible; Fit Width and fixed image zoom are also available. Focus Figure hides navigation, metadata and footer to reclaim space; Escape restores the full workbench. Save Figure copies the original PNG without changing the calculation or overwriting an existing file. These controls are not a solver or interactive numerical reanalysis.

The run folder includes profile.json, run_report.json, simulation_arrays.npz for sensing, scientific PNGs, status.json, run.log and manifest.json. The manifest records input/artifact SHA-256 values, Python and major package versions, UTC start/finish times, inherited BLAS/OMP thread settings and available Git information. It does not sign authenticity, archive the entire ARC database or lock every transitive dependency.

Saved results are verified in a background task. Missing, modified or unsupported receipts are rejected and an older successful result is not substituted. CLI output without the GUI receipt is not automatically presented as a verified GUI run. The synchronous read_result helper is still available for non-UI callers.

## Tests

Install requirements-dev.txt and requirements-gui.txt first:

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_gui_jobs.py -q -ra -s
& ./.venv/Scripts/python.exe -m pytest gui_tests -q -ra -s
```

GUI tests require Qt explicitly; a missing optional GUI installation is not silently counted as a pass. Live tests require writable ARC data and exercise real subprocesses. They cover independent runs, exact AoA/IF-only equivalence to CLI examples, import/edit/save behavior, invalid drafts, effect preservation and report-write failure. Thread-limited test commands and actual results for this delivery are recorded in the [verification report](gui-verification.md).

## Limits

This is a research workbench, not experimental validation or a guarantee of arbitrary-profile convergence. Study/magnetic-core execution, graceful solver cancellation, non-Windows process-tree stopping, standalone packaging, multiuser service, exhaustive DPI/accessibility verification and interactive numerical plotting remain outside the implemented scope. Other operating systems have not been certified.

Advanced Structure uses clearly bordered in-cell editors with explicit row height; click a scalar value or use F2. Editing preserves numeric text/scientific notation until the core parser validates the resulting profile. The large application heading has been removed; the operating-system window title identifies the application. The JSON preview can be hidden via the preview checkbox.

Process ownership uses Windows Job Objects with no breakaway permission and kill-on-last-handle-close. See the [Microsoft Job Objects documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects). Other application processes are not members of this calculation job.
