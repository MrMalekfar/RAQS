# GUI architecture and remaining work

This document separates the implemented desktop architecture from future work. The active user guide is [gui.md](gui.md); visual conventions are in [DESIGN.md](../DESIGN.md).

## Implemented architecture

```text
Qt workbench / task-oriented editor
    -> canonical profile parsing and fixed byte snapshot
    -> QProcess: fresh Python worker
    -> execute_profile / existing result and figure writers
    -> complete artifact receipt
    -> background receipt/hash verification
    -> measurement summary, saved figures and original report
```

The UI owns presentation, draft state, file selection and process lifecycle. The core owns scientific validation, atomic resolution and numerical execution. No scientific rules, fallback atomic channels or Qt imports were added to the core. Rendering consumes completed outputs and is not run in the UI thread as part of the solver.

## Configuration contract

The editor represents one draft. JSON preview, Save As and the run snapshot read from that same draft; an invalid edit clears the executable snapshot. Import rebuilds controls. Unhandled composite values are preserved and remain accessible in Advanced structure. Optional effects are copied only from a selected explicit example or edited in the tree; no valid-looking physical parameters are invented.

Original file bytes are preserved before edits. After edits JSON formatting may change. Source files, baselines and prior outputs are not overwritten by normal GUI operations.

## Execution and result contract

Ordinary AoA and IF-only profiles use execute_profile in an independent process. Diagnostics is a distinct operation. Windows jobs establish an owned process group before physics startup; Stop terminates that group and waits for cleanup. Interrupted folders are explicitly rejected. A job is not accepted as completed merely because the solver returned: array/figure/report writing and the final receipt must succeed. The visible eligibility line distinguishes editable/parsed/runnable state and whether the current input matches a displayed verified result.

Opening and verifying saved artifacts happens outside the UI thread. A generation token prevents a late read from replacing results after the input changes. Scientific image zoom/export does not recompute data. Missing curves, unavailable angles and unexecuted effects are not fabricated.

## Remaining work, not current capabilities

- Study and magnetic-core execution require explicit routing and measurement-specific result review; until then they remain visibly unavailable in this GUI.
- Windows forced Stop is implemented and tested with descendants and an unrelated process. Graceful solver/study cancellation and non-Windows stopping remain separate work.
- A standalone Windows package requires a separate ARC data, spawn, font and clean-machine validation pass.
- Numerical interactive plots may be added using saved arrays, independently of solver execution. Current figure zoom is image navigation only, with Fit Page and a focus mode.
- Exhaustive native keyboard/pointer, assistive-technology and Windows scaling verification remain release gates. Other operating systems require independent validation.

Do not add a web service, plugin framework, schema abstraction or new dependency until a concrete requirement justifies it. Adding a model input starts with its core parser/tests; the GUI then adds labels, units, controls and result presentation for the supported contract.
