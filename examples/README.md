# Example profile map

The local `ch3_source_profiles/README.md`, when present, records recovered inputs, differences between same-named versions and their SHA-256 provenance. Import a run profile JSON through the GUI. Research `design.json` files describe studies and have separate execution requirements.

[Getting started](../README.md) · [Execution and outputs](../docs/usage.md) · [Scientific assumptions](../docs/README.md)

Each JSON run profile is an independent input. Preserve the original and edit a copy for a study. Read `schema_version`, `execution` and `study` before running. Some inputs require specialized execution or substantial computation.

| Directory | Purpose and entry point |
| --- | --- |
| [lab006](lab006/) | Start here. `prescribed_reference.json` provides diagnostics, `experimental_near_resonant.json` bounded IF/AoA sensing, and `experimental_reference.json` channel diagnostics |
| [lab007](lab007/) | Schema 2, environment/spectrum declarations, prescribed/experimental sensing and uniform/nonuniform scans |
| [atom001](atom001/), [atom002](atom002/) | Rb85, Rb87 and Cs133 profiles with explicit species contracts |
| [dop_td001](dop_td001/), [dop_profile001](dop_profile001/), [dop_conv001](dop_conv001/) | Temporal Doppler, adaptive quadrature and velocity-grid checks |
| [num_ode002](num_ode002/) | Explicit ODE controls for three species and a bounded warm case |
| [relax002](relax002/) | The `selected_internal_spontaneous_e1` radiative model |
| [noise002](noise002/), [noise003](noise003/) | Gaussian readout and repeated noise campaigns |
| [noise004](noise004/), [noise005](noise005/) | PIN readout and RF drive noise |
| [noise006](noise006/), [noise007](noise007/) | Laser frequency noise and RIN |
| [noise008](noise008/), [noise009](noise009/) | Noise sweeps and study statistics |
| [noise010](noise010/), [noise011](noise011/), [noise012](noise012/) | Grid convergence, IF studies and field response |
| [noise013](noise013/), [noise014](noise014/) | IF-only signal/null runs and detection studies |
| [noise015](noise015/), [noise016](noise016/), [noise018](noise018/) | Sampled sensitivity, detection numerics and operating-point/candidate studies |
| [literature001](literature001/) | Literature-related profiles and a `manifest.json` defining comparison scope |

The `literature001` manifest is a reference test input and must retain its provenance. A profile named after a paper alone cannot establish reproduction of its experiment. Profile parsers and executors define supported behavior.

Locally imported Chapter 3 research records retain their original Persian documentation. Their README files and source manifests identify archive dependencies. The imported records may cite historical tools and outputs absent from the release repository. Follow the source-profile manifest when selecting executable inputs.
