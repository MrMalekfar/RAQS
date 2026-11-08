"""RF-004 seam-caller integration regression tests.

Root cause covered here: commit 84b8e74 introduced ``apply_noise_seam``
(documented contract: returns exactly the 2-tuple ``(H, c_ops)``) but wired
both production callers with a 3-target unpack, so ``python main.py`` crashed
at the first steady-state solve with
``ValueError: not enough values to unpack (expected 3, got 2)``.

The unit suite missed it because ``tests/test_noise_seam_rf004.py`` tests only
the seam helper with synthetic Qobjs (its "caller contract" test inspects
signatures via ``inspect`` without ever calling the functions), and
``tests/test_simulation_runner.py`` mocks ``simulate_spectrum`` entirely.

These tests execute the REAL production caller bodies end-to-end:
no mock on ``apply_noise_seam``, no mock on ``create_sensor_model``,
no mock on the QuTiP solvers.

Known separate follow-up defect (NOT fixed here, out of scope): the
time-domain worker returns a 6-element error tuple ``(None, None, None, i,
None, error_msg)`` on exception versus a 5-element success tuple — a
structurally inconsistent return contract inside ``run_single_heterodyne_worker``.
"""

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.config import parameters as params
from rydberg_aoa.simulation import steady_state as ss
from rydberg_aoa.simulation import time_domain as td
from rydberg_aoa.simulation.noise_seam import apply_noise_seam
from tests.atomic_init_support import ensure_resolved_atomic_physics


@pytest.fixture(scope="module")
def real_physics():
    """Real ARC-derived physics bundle, identical to the runner's inputs.

    ARCH-001T: opts into the explicit ARCH-001 seam through the shared
    support helper (idempotent; skipped with a ``BLOCKED: QN-001A`` reason
    while the coupling mJ contract is open).
    """
    ensure_resolved_atomic_physics()

    from rydberg_aoa.config import atomic_config as acfg
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates,
    )

    dp, dc, drf = acfg.get_cached_detunings()
    return {
        "omegas": (acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE),
        "deltas": (dp, dc, drf),
        "gammas": [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43],
        "dephasing": list(calculate_dephasing_rates()),
    }



# ---------------------------------------------------------------------------
# Test C — seam contract pin (fail fast if the seam shape ever changes)
# ---------------------------------------------------------------------------
def test_seam_none_path_returns_exactly_two_items():
    """``apply_noise_seam`` must return exactly (H, c_ops) — a 2-tuple.

    Pinning this shape here means any future seam-side contract change fails
    a seam test instead of a production run (the failure mode of 84b8e74).
    """
    H = qt.basis(2, 0) * qt.basis(2, 0).dag()
    c_ops = [np.sqrt(0.5) * (qt.basis(2, 0) * qt.basis(2, 1).dag())]

    out = apply_noise_seam(H, c_ops, obs=None, noise_model=None)

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] is H          # strict identity, per the RF-004 no-op contract
    assert out[1] is c_ops


# ---------------------------------------------------------------------------
# Test A — steady-state seam caller (the exact production crash site)
# ---------------------------------------------------------------------------
def test_solve_probe_transmission_executes_through_seam(real_physics):
    """``_solve_probe_transmission`` must run the seam unpack and reach
    ``steadystate`` for real, returning a finite transmission.

    Before the fix this raised
    ``ValueError: not enough values to unpack (expected 3, got 2)``
    at steady_state.py:51 — before any solver was ever invoked.
    """
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf = real_physics["omegas"]

    # Resonant point: the very first scan point class main.py evaluates.
    transmission = ss._solve_probe_transmission(
        dp, dc, omega_p, omega_c, omega_rf, drf,
        real_physics["gammas"], real_physics["dephasing"],
        noise_model=None,
    )

    # A finite scalar transmission proves the full chain executed:
    # seam unpack -> steadystate solve -> observable trace -> susceptibility.
    assert np.isfinite(transmission)
    assert transmission > 0.0



def test_simulate_spectrum_minimal_scan_executes_through_seam(
        real_physics, monkeypatch):
    """Minimal real ``simulate_spectrum`` sweep exercises the production loop
    that crashed in main.py, without mocking anything.

    NOTE (R-2A update): ``simulate_spectrum`` now honors its
    ``detuning_list_mhz`` argument EXACTLY (values, ordering, length); the
    historical defect of silently rebuilding the scan from
    ``params.SCAN_RANGE_MHZ``/``params.N_SCAN`` is fixed. The scan grid is
    therefore supplied explicitly here.
    """
    scan_points = np.array([-1.0, 0.0, 1.0])

    dp, dc, drf = real_physics["deltas"]

    scan, trans_num, trans_ana = ss.simulate_spectrum(
        real_physics["omegas"][2],
        scan_points,                           # honored exactly (R-2A)
        real_physics["omegas"],
        (dp, dc, drf),
        real_physics["gammas"],
        real_physics["dephasing"],
        absorption_const=1.0,                  # value irrelevant to this path
        noise_model=None,
    )

    assert len(scan) == 3
    assert np.allclose(scan, scan_points, rtol=0, atol=0)
    assert np.all(np.isfinite(trans_num))
    assert np.all(np.isfinite(trans_ana))


# ---------------------------------------------------------------------------
# Test B — time-domain seam caller (latent second crash site, silent path)
# ---------------------------------------------------------------------------
def test_heterodyne_worker_executes_through_seam_success_shape(real_physics):
    """``run_single_heterodyne_worker`` must return the 5-element SUCCESS
    tuple with a non-None signal trace when the seam call succeeds.

    Before the fix the ValueError was swallowed by the worker's broad
    ``except Exception`` and returned the 6-element error tuple — a silent
    production failure. A reduced time grid keeps this fast while still
    executing seam -> steadystate -> mesolve -> FFT extraction for real.
    """
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_lo = real_physics["omegas"]

    # Reduced grid: same span as production (15 IF cycles) but 101 steps
    # instead of 1001 — fast, yet spans multiple IF periods so the FFT
    # extraction window is populated.
    tlist = np.linspace(0.0, params.T_MAX, 101)

    complex_amp, xf, yf, i, signal_trace = td.run_single_heterodyne_worker(
        0,                                     # element index
        real_physics["omegas"][2] * (1.0 + 0.0j),   # signal Rabi (complex)
        tlist,
        params.IF_FREQ,
        omega_p, omega_c, omega_lo,
        dp, dc, drf,
        real_physics["gammas"],
        real_physics["dephasing"],
        noise_model=None,
    )

    # Success-path contract: exactly 5 elements, all real data (no error tail).
    assert signal_trace is not None
    assert len(signal_trace) == len(tlist)
    assert np.all(np.isfinite(signal_trace))
    assert np.iscomplexobj(complex_amp) or isinstance(complex_amp, complex)
    assert np.isfinite(complex_amp)
    assert xf is not None and yf is not None
    assert i == 0
