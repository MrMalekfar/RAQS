# Workbench verification — 2026-11-09

Branch: `codex/research-workbench`. All changes remain local. No commit, push or merge was performed. Existing uncommitted changes, chapter-three profiles and the user's profile copy were retained.

## Audit and scope

The audit read the actual frontend, profile and output contracts, worker, examples and GUI tests, then exercised the existing application. The baseline real ARC/spawn tests completed with 9 passes before redesign. Native accessibility inspection exposed the nested form/tab hierarchy; native screenshot capture timed out.

The highest-priority defects were JSON-key-driven forms, hidden optional effects, unclear execution eligibility, report-first results, fixed-width figures and synchronous artifact verification. The implementation now uses task-based configuration, a persistent status/command bar, direct effect controls, a measurement summary, image navigation/export and background artifact verification. The first filled review also found a persistent error tint and dense result rows; these were fixed together. Vector duplication was corrected to preserve all values.

## Completed verification runs

Environment: Windows, project Python 3.11.0, optional PySide6-Essentials 6.10.3. `pip check`, syntax compilation, local documentation links and `git diff --check` passed.

| Completed command | Pass | Fail | Skip |
| --- | ---: | ---: | ---: |
| `python -m pytest gui_tests -q -ra -s` | 22 | 0 | 0 |
| `python -m pytest gui_tests/test_workbench.py -q -ra -s` after the final stale-read regression was added | 7 | 0 | 0 |
| `python -m pytest tests/test_gui_jobs.py tests/test_lab006_run_profile.py tests/test_lab007_profile_integration.py -q -ra -s` | 92 | 0 | 0 |

The focused workbench rerun overlaps six tests in the full GUI run. This is **115 distinct passing tests**, not 121 different tests. The whole repository suite was not run.

Real checks include two fresh AoA jobs, Cs133 IF-only diagnostics/calculation, exact report/array comparison against CLI for the same input, an actual report-write failure, process-start failure, incomplete artifacts, invalid drafts, optional-effect preservation, vector edits and late-read invalidation. Button-driven Qt interaction covers import, numeric editing, saving and reopening; file-dialog paths are supplied by tests. Synthetic fixtures test UI/file contracts and do not replace the live ARC/spawn checks.

The accepted resource-bounded runs used these settings only in the test command environment:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& ./.venv/Scripts/python.exe -m pytest gui_tests -q -ra -s
```

An earlier concurrent attempt encountered an explicit OpenBLAS allocation failure and did not produce a complete test result; it is not counted as passing. Another preliminary run returned 19 pass / 1 fail because a UI title-case assertion expected `Probe laser` instead of the actual `Probe Laser`; the assertion was corrected. No physical input, baseline, seed or solver tolerance was changed for either issue. The separate user calculation was not intentionally terminated.

## Filled render and interaction review

Actual production-generated artifacts were verified and reopened for the review. The review covered configuration, invalid input, diagnostics, a running monitor, measurement summary, scientific figures and a compact 1000×680 window. Normal review size was 1380×880. An additional Qt offscreen render at a simulated scale factor of 1.5 also succeeded; it is not a test of native Windows DPI switching. The screenshot source was a real numerical run, not illustrative scientific data.

Local captures and their SHA-256/provenance are in `outputs/workbench-review/`. They include `configure.png`, `invalid-draft.png`, `measurement-summary.png`, `scientific-figure.png`, `compact-result.png`, `diagnostics.png`, `running.png` and `live-diagnostics.png`. The last two were captured during/after an additional real diagnostic job through the new button. Its final receipt was checked for UTC start time and inherited thread settings.

Qt event-driven interaction and offscreen rendering succeeded. Full native pointer interaction/capture, real Windows DPI changes, assistive-technology coverage and a clean-machine installer remain **unverified**. These checks must not be inferred from a passing offscreen test or accessibility-tree discovery.

## Remaining limits

Study and magnetic-core execution remain visibly unavailable in this GUI. Figure zoom is image navigation; no interactive numerical plotting or graceful solver cancellation is claimed. Windows forced Stop was added in the follow-up below. Packaging and other operating systems remain separate work. Existing core/CLI code, runtime dependencies, baselines and scientific settings were not modified.


## Stop and space-use follow-up — 2026-11-09

Active branch: `codex/workbench-stop-layout`, created before this follow-up. No staging, commit, push or merge was performed. The earlier local work and user examples remain intact.

The follow-up fixes the invisible Advanced Structure scalar editor, removes the oversized application heading, moves eligibility and notifications to the footer, opens numerical results on Figures, defaults to Fit Page and adds Focus Figure / Escape. The JSON preview can be hidden. Run Monitor follows the latest log output by default; disabling Follow latest output preserves manual scrolling.

On Windows the real interpreter establishes a Job Object before physics startup and waits for GUI authorization. Stop terminates only this owned group and waits for zero active members and launcher exit. Tests cover an actual profile, a real descendant process, an unrelated process surviving Stop, failed ownership preventing startup, interrupted-result rejection and a subsequent real diagnostics job. This is forced termination, not graceful solver cancellation. Partial files are retained. Non-Windows process-tree Stop remains unsupported.

| Completed command for this follow-up | Pass | Fail | Skip |
| --- | ---: | ---: | ---: |
| `python -m pytest gui_tests -q -ra -s` | 29 | 0 | 0 |
| `python -m pytest tests/test_gui_jobs.py tests/test_lab006_run_profile.py tests/test_lab007_profile_integration.py -q -ra -s` | 92 | 0 | 0 |

These are **121 distinct passing tests**. Earlier focused reruns overlap and are not added. Logs are `outputs/stop-layout-gui-suite.txt` and `outputs/stop-layout-core-suite.txt`. The same test-only BLAS thread caps documented above were used. Live AoA / IF-only ARC and spawn checks and CLI equivalence remained in the GUI suite. The full repository suite was not run.

Filled Qt captures were inspected together at 1380×880 and 1000×680, using a verified real numerical result. `outputs/stop-layout-review/` contains configuration, the active structural value editor, scientific figure, focused and compact figure views, and a real running/stopped monitor. `provenance.json` records source and image hashes. Escape was exercised after restoring the main window's active focus from the modal editor. A second real profile was stopped after its worker had written log output. The captured stop state is not presented as a completed calculation.

This review used QTest events and Qt window grabs. Native pointer/screenshot tooling and real Windows DPI changes remain unverified; successful Qt rendering does not certify them. Physics/core, CLI, dependencies, baseline, seed and tolerance were unchanged.

The additional sandboxed monitor-capture attempt encountered ARC SQLite `attempt to write a readonly database` and is not a successful calculation. Repeating the capture outside that filesystem restriction produced real ARC log output and verified owned-process stopping. No atomic fallback or fabricated result was used. Final `pip check`, syntax compilation, local documentation link checks and `git diff --check` succeeded.


## Review fixes

The branch `codex/workbench-review-fixes` starts at `cfa280d`, the merge of PR #4. The review fixes preserve an unsaved draft when Import or Reload fails, validate Advanced Structure before replacing the form, keep verified results available after edits, and bind the displayed and exported PNG to the verified bytes. A diagnostics report for a different snapshot opens in Results.

On Windows with the documented test-only BLAS thread caps, `python -m pytest gui_tests tests/test_gui_jobs.py -q -ra` finished with **41 passed, 0 failed, 0 skipped** in 130.93 seconds. The run includes live ARC AoA and Cs133 IF-only paths and comparison with the CLI. The earlier full attempt had 38 passes and three failures. A delayed log scrollbar range caused one failure. The other two involved an obsolete default-preview assertion and a test that changed the active flag without refreshing its controls. The final run included the fixes. The restricted-shell attempt stopped during ARC setup before test collection.

The Windows Qt backend was used at the host's existing 125% scale. At 1090×610 logical pixels, a keyboard section change, the optional JSON pane, a prior result with a different input snapshot, and visible export feedback in Focus Figure were checked. Local captures are under `outputs/workbench-review-fixes/`. The filled figure was reopened from a prior real run whose manifest passed verification. QTest supplied keyboard events and the export path. OS scaling changes and NVDA were outside this check. The whole repository suite was not run.
