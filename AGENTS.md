# Release development guide

Write documentation in English and respond in the user's requested language. Preserve imported Chapter 3 research documentation in Persian. Before editing, inspect Git status and the relevant source. Preserve the user's existing changes. Make the smallest complete, relevant change. Keep the model, baselines, seeds and acceptance tolerances fixed when diagnosing test failures.

The main entry points are `main.py`, `cli.py` and `rydberg_aoa/`. Active tests are in `tests/` and profiles in `examples/`. Use Codebase Memory for structure when available, and source/configuration/tests for behavioral evidence. This repository excludes the research archive and thesis inputs.

Check each function's units independently. Time is usually µs, detuning and Rabi frequency rad/µs, and event rates µs⁻¹. Convert Hz to rad/µs with `2πf/10⁶`. Convert event rates from s⁻¹ by dividing by 10⁶. Preserve basis ordering, detuning signs, phase, propagation direction and the factor of 1/2 unless justified and tested. Keep transit reset separate from pure dephasing.

Preserve ARCH-001. Imports must leave physical initialization explicit. ARC and configuration failures must propagate. Preserve QN-001, the declared F/mF and j/mJ states, and polarization. Keep mF and mJ distinct. Channel selection must respect the declaration without scanning for the largest dipole. Context7 discovery may help with ARC. Verify ambiguous APIs against official ARC source.

Solver changes require checks of ordering, trace, Hermiticity, positivity and relevant convergence. In DoA, keep sensor ordering, steering, phase, snapshot shapes and covariance consistent with the pipeline. Atomic fixtures must remain explicit. A skip is permitted only at the documented QN-001A boundary and must be reported separately from PASS.

`outputs/` contains fresh run results. Program and test inputs must come from declared profiles or fixtures. Reference fixtures require provenance, version, units and a hash. Add dependencies only for a concrete need. Runtime dependencies belong in `requirements.txt` and test dependencies in `requirements-dev.txt`.

After editing, run tests close to the affected behavior and check the diff and whitespace. README example commands require separate execution verification. Report the change, reason, tests actually run, pass/fail/skip counts and remaining limitations. Connect implementation claims to a path and function. Label future proposals clearly.
