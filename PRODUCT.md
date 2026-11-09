# RAQS Research Workbench

RAQS is research software for simulating Rydberg atomic sensing, heterodyne IF measurements, and array angle-of-arrival estimation. The desktop workbench serves researchers who need to configure, inspect, run and revisit a traceable simulation.

## Confirmed product constraints

- Local desktop application, initially verified on Windows; PySide6 / Qt Widgets remains the UI platform.
- English interface and GUI documentation. Existing scientific and project documents retain their own scope.
- The canonical profile parser and physics APIs own scientific validation. Qt is optional and never imported by the core or CLI.
- Inputs may be created from explicit example templates or imported from JSON. Unsaved drafts, current execution snapshots and saved results are distinct.
- Ordinary AoA and IF-only calculations use a fresh Python process for each job. This is numerical research software, not experimental certification.
- Study and magnetic-core documents can be edited, but their execution is explicitly unavailable in this GUI. No implicit fallback model is allowed.
- Windows offers an explicit forced Stop with verified owned-process cleanup. Graceful solver cancellation, non-Windows stopping, installer, web service and multiuser operation are not claimed.

## Success criteria

A researcher can start without a personal JSON file; edit scientific inputs without losing precision; identify why a draft is not runnable; execute diagnostics or a measurement without freezing the UI; and reopen a verified result with its actual method, units and model limitations.

See [the user guide](docs/gui.md), [the design system](DESIGN.md) and [architecture and remaining work](docs/gui-plan.md).
