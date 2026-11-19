# Original LAB-006 inputs

The original bytes came from commit `b757ca9`, under `examples/lab006/`. Units and settings are declared in each profile. Their SHA-256 values are listed below.

- `experimental_reference.json` — `f201db181a9efb54b60aa5b8cf17f7d850cd48dddd33b209aa7fe75d6bcf0317`
- `experimental_near_resonant.json` — `51dd30930e34b59114a5ade36370562207322baa68c55a88d2439aa2a3a46e3b`
- `prescribed_reference.json` — `4188cbebe7c62d9b08a1e19f73db73bea18694c8470161b3343847a74dd5819b`

The prepared-channel regression checks the last input's hash and resolves it with the ARC dependency in `requirements.txt` (3.9.0 when the audit findings were fixed). An independent NumPy solution uses the rotating-frame Hamiltonian and trace-one Lindblad equation to obtain the reference density matrix. Rabi frequencies and detunings use rad/µs, and event rates use µs⁻¹. This solution uses shared ARC inputs and independent linear algebra, without QuTiP operators or historical generated project outputs. The test establishes numerical agreement under those assumptions. Experimental agreement requires separate evidence.
