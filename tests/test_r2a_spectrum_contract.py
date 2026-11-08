"""R-2A — simulate_spectrum parameter-contract regression tests.

Contract established by R-2A (ONE source of truth, no silent overwrite):

1. ``omega_rf`` (first positional argument) is the SINGLE authoritative RF
   Rabi frequency of the run. The retired ``p_omegas[2]`` aggregate-bundle
   slot is NEVER assigned over it. Historical defect (reproduced at runtime
   before the fix): with explicit ``omega_rf=1.234`` and
   ``p_omegas[2]=5.678``, every solve silently received 5.678.
2. ``detuning_list_mhz`` is honored EXACTLY (values, ordering, length) when
   provided; ``None`` regenerates the configured default linspace.
   Historical defect: the explicit list was silently ignored.
3. ``absorption_const`` is a DEAD/STALE compatibility argument (accepted,
   never read; transmission constants are re-derived from configuration
   inside ``calculate_transmission_from_rho21``). Pinned here so any silent
   semantic change is caught. Cleanup deferred to R-2B.

RF-state-to-Hamiltonian validation uses the REAL ``create_sensor_model``
(model-level oracle: H_34 = -Omega_rf/2), not transmission-only checks.
Fast tests: single-point scans, Doppler disabled or solver mocked.
"""

import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.simulation import steady_state as ss
from tests.atomic_init_support import ensure_resolved_atomic_physics


A_EXPLICIT = 1.234   # explicit omega_rf argument (deliberately conflicting)
B_BUNDLE = 5.678     # retired p_omegas[2] bundle slot


@pytest.fixture(scope="module")
def real_physics():
    """Real ARC-derived physics bundle, identical to the runner's inputs.

    ARCH-001T: resolved quantities exist only after the explicit ARCH-001
    seam ran, so this fixture opts in through the shared support helper
    (idempotent; pytest skips with an explicit ``BLOCKED: QN-001A`` reason
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
        "acfg": acfg,
    }


@pytest.fixture
def cold_path(monkeypatch):
    """Disable Doppler averaging: exactly one v=0 class per scan point."""
    monkeypatch.setattr(params, "DOPPLER_ENABLED", False, raising=True)


@pytest.fixture
def mocked_solver(monkeypatch):
    """Replace the per-point solver with a constant complex coherence."""
    monkeypatch.setattr(
        ss, "_solve_probe_coherence", lambda *a, **kw: complex(0.05, -0.02)
    )


# ---------------------------------------------------------------------------
# 1. RF source of truth — the exact historical overwrite defect
# ---------------------------------------------------------------------------
def test_explicit_omega_rf_is_authoritative_over_bundle(
        real_physics, cold_path, monkeypatch):
    """The bundle slot p_omegas[2] must NEVER overwrite explicit omega_rf.

    Pre-fix runtime reproduction: all 201 solves received B_BUNDLE while the
    caller requested A_EXPLICIT. Post-fix the model must see A_EXPLICIT.
    """
    captured = []
    real_csm = ss.create_sensor_model

    def spy_csm(omega_p, omega_c, omega_rf_lo, *a, **kw):
        captured.append(omega_rf_lo)
        return real_csm(omega_p, omega_c, omega_rf_lo, *a, **kw)

    monkeypatch.setattr(ss, "create_sensor_model", spy_csm)

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    scan, t_num, t_ana = ss.simulate_spectrum(
        A_EXPLICIT,                          # explicit request = A
        np.array([0.0]),
        (omega_p, omega_c, B_BUNDLE),        # bundle slot = B (conflicting)
        (dp, dc, drf),
        real_physics["gammas"],
        real_physics["dephasing"],
        None,
    )

    assert len(scan) == 1
    assert captured == [A_EXPLICIT]          # exactly one build, exact value


def test_rf_off_request_reaches_hamiltonian_without_runner_compensation(
        real_physics, cold_path, monkeypatch):
    """RF-off via the production call shape, WITHOUT zeroing p_omegas[2].

    Pre-fix, this call (omega_rf=0.0 with the UNCOMPENSATED bundle
    (Omega_p, Omega_c, OMEGA_RF_BASE)) silently ran the RF-ON physics — the
    defect that forced runner.py to zero p_omegas[2]. Post-fix the requested
    0.0 must reach create_sensor_model unchanged.
    """
    captured = []
    real_csm = ss.create_sensor_model

    def spy_csm(omega_p, omega_c, omega_rf_lo, *a, **kw):
        captured.append(omega_rf_lo)
        return real_csm(omega_p, omega_c, omega_rf_lo, *a, **kw)

    monkeypatch.setattr(ss, "create_sensor_model", spy_csm)

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf_base = real_physics["omegas"]

    # Exactly the runner's RF-off call, minus the p_omegas[2]=0 compensation:
    ss.simulate_spectrum(
        0.0,
        np.array([0.0]),
        (omega_p, omega_c, omega_rf_base),   # bundle NOT compensated
        (dp, dc, drf),
        real_physics["gammas"],
        real_physics["dephasing"],
        None,
        transit_rate_per_us=0.0,
    )
    assert captured == [0.0]


def test_rf_on_request_reaches_hamiltonian_unchanged(
        real_physics, cold_path, monkeypatch):
    """RF-on production call: OMEGA_RF_BASE must reach create_sensor_model."""
    captured = []
    real_csm = ss.create_sensor_model

    def spy_csm(omega_p, omega_c, omega_rf_lo, *a, **kw):
        captured.append(omega_rf_lo)
        return real_csm(omega_p, omega_c, omega_rf_lo, *a, **kw)

    monkeypatch.setattr(ss, "create_sensor_model", spy_csm)

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf_base = real_physics["omegas"]

    ss.simulate_spectrum(
        omega_rf_base,
        np.array([0.0]),
        (omega_p, omega_c, omega_rf_base),
        (dp, dc, drf),
        real_physics["gammas"],
        real_physics["dephasing"],
        None,
        transit_rate_per_us=0.0,
    )
    assert captured == [omega_rf_base]
    assert omega_rf_base > 0.0               # genuinely the configured RF-on value


# ---------------------------------------------------------------------------
# 2. Model-level Hamiltonian oracle (not transmission-only)
# ---------------------------------------------------------------------------
def test_hamiltonian_rf_coupling_oracle(real_physics):
    """H_34 = -Omega_rf/2 exactly, for RF-off (0) and RF-on (OMEGA_RF_BASE).

    Basis |1>..|4> -> indices 0..3; s43 = |4><3| sits at [3, 2] (0-based).
    The probe element [1, 0] = -Omega_p/2 pins the indexing convention.
    """
    from rydberg_aoa.physics.hamiltonian.create_sensor_model import (
        create_sensor_model,
    )

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf_base = real_physics["omegas"]

    for omega_rf_requested in (0.0, omega_rf_base):
        H, c_ops, obs = create_sensor_model(
            omega_p, omega_c, omega_rf_requested,
            dp, dc, drf,
            real_physics["gammas"], real_physics["dephasing"],
            transit_rate_per_us=0.0,
        )
        h = H.full()                         # QuTiP Qobj → complex ndarray
        # Indexing sanity: probe coupling must be present and exact.
        assert h[1, 0] == pytest.approx(-omega_p / 2.0, abs=0.0)
        assert h[0, 1] == pytest.approx(-omega_p / 2.0, abs=0.0)
        # RF coupling carries the REQUESTED value, unchanged:
        assert h[3, 2] == pytest.approx(-omega_rf_requested / 2.0, abs=0.0)
        assert h[2, 3] == pytest.approx(-omega_rf_requested / 2.0, abs=0.0)

    # RF-off vs RF-on Hamiltonians must genuinely differ in the RF block.
    H_off, _, _ = create_sensor_model(
        omega_p, omega_c, 0.0, dp, dc, drf,
        real_physics["gammas"], real_physics["dephasing"],
        transit_rate_per_us=0.0)
    H_on, _, _ = create_sensor_model(
        omega_p, omega_c, omega_rf_base, dp, dc, drf,
        real_physics["gammas"], real_physics["dephasing"],
        transit_rate_per_us=0.0)
    assert H_off.full()[3, 2] != H_on.full()[3, 2]


def test_rf_state_changes_real_coherence(real_physics):
    """The requested RF state changes the physical response (unmocked solves).

    rho21 at the same detunings must differ between Omega_rf = 0 and
    OMEGA_RF_BASE — the RF coupling is not a silent no-op anywhere.
    """
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf_base = real_physics["omegas"]

    rho_off = ss._solve_probe_coherence(
        dp, dc, omega_p, omega_c, 0.0, drf,
        real_physics["gammas"], real_physics["dephasing"],
        transit_rate_per_us=0.0)
    rho_on = ss._solve_probe_coherence(
        dp, dc, omega_p, omega_c, omega_rf_base, drf,
        real_physics["gammas"], real_physics["dephasing"],
        transit_rate_per_us=0.0)

    assert rho_off != rho_on
    assert abs(rho_on - rho_off) > 1e-6


# ---------------------------------------------------------------------------
# 3. detuning_list_mhz honored EXACTLY
# ---------------------------------------------------------------------------
def test_custom_irregular_list_honored_exactly(
        real_physics, cold_path, mocked_solver):
    """TEST A: custom irregular list -> output length 4, values exact."""
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]
    custom = [-3.1, -0.7, 0.2, 2.9]

    scan, t_num, t_ana = ss.simulate_spectrum(
        0.0, custom,
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"], None,
    )

    assert len(scan) == 4
    assert np.allclose(scan, np.array(custom), rtol=0, atol=0)


def test_each_solve_receives_corresponding_detuning(
        real_physics, cold_path, monkeypatch):
    """TEST B: every solve receives delta_p_center + 2*pi*delta[MHz], in order."""
    captured = []

    def spy_spc(*args, **kw):
        captured.append(args[0])             # delta_p_rad, first positional
        return complex(0.05, -0.02)

    monkeypatch.setattr(ss, "_solve_probe_coherence", spy_spc)

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]
    custom = [-3.1, -0.7, 0.2, 2.9]

    ss.simulate_spectrum(
        0.0, custom,
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"], None,
    )

    expected = [dp + 2.0 * np.pi * dp_mhz for dp_mhz in custom]
    assert captured == pytest.approx(expected, rel=1e-12)


def test_reversed_input_ordering_remains_reversed(
        real_physics, cold_path, mocked_solver):
    """TEST C: ordering is preserved, never re-sorted."""
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]
    custom = [2.9, 0.2, -0.7, -3.1]          # reversed version of TEST A

    scan, _, _ = ss.simulate_spectrum(
        0.0, custom,
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"], None,
    )

    assert np.allclose(scan, np.array(custom), rtol=0, atol=0)


def test_single_point_list_works(real_physics, cold_path, mocked_solver):
    """TEST D: an explicit single-point list yields exactly one solve."""
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    scan, t_num, t_ana = ss.simulate_spectrum(
        0.0, [0.42],
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"], None,
    )

    assert len(scan) == 1
    assert scan[0] == 0.42
    assert len(np.atleast_1d(t_num)) == 1
    assert len(np.atleast_1d(t_ana)) == 1


def test_none_preserves_configured_default_scan(
        real_physics, cold_path, mocked_solver, monkeypatch):
    """TEST E: None regenerates the configured default linspace exactly."""
    monkeypatch.setattr(params, "N_SCAN", 5, raising=True)
    monkeypatch.setattr(params, "SCAN_RANGE_MHZ", 7.0, raising=True)

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    scan, _, _ = ss.simulate_spectrum(
        0.0, None,
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"], None,
    )

    assert np.allclose(
        scan,
        np.linspace(-params.SCAN_RANGE_MHZ, params.SCAN_RANGE_MHZ, params.N_SCAN),
        rtol=0, atol=0,
    )


def test_empty_explicit_list_fails_loud(real_physics, cold_path, mocked_solver):
    """An explicitly provided EMPTY list is a caller error, not a silent run."""
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    with pytest.raises(ValueError):
        ss.simulate_spectrum(
            0.0, [],
            (omega_p, omega_c, 0.0), (dp, dc, drf),
            real_physics["gammas"], real_physics["dephasing"], None,
        )


# ---------------------------------------------------------------------------
# 4. absorption_const — dead/stale argument pinned as inert
# ---------------------------------------------------------------------------
def test_absorption_const_is_inert(real_physics, cold_path, mocked_solver):
    """absorption_const is accepted for compatibility but never read.

    (R-2A investigation: calculate_transmission_from_rho21 re-derives its
    constants from configuration internally. Cleanup candidate: R-2B.)
    """
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    kwargs = dict(
        p_omegas=(omega_p, omega_c, 0.0),
        p_deltas=(dp, dc, drf),
        p_gammas=real_physics["gammas"],
        p_dephasing=real_physics["dephasing"],
    )
    _, t_sentinel, _ = ss.simulate_spectrum(
        0.0, [0.0], absorption_const=12345.0, **kwargs)
    _, t_other, _ = ss.simulate_spectrum(
        0.0, [0.0], absorption_const=-1e9, **kwargs)

    assert t_sentinel == t_other             # identical output => provably unread


# ---------------------------------------------------------------------------
# 5. Caller compatibility (legacy positional / keyword forms)
# ---------------------------------------------------------------------------
def test_legacy_runner_style_positional_call(
        real_physics, cold_path, mocked_solver):
    """The exact runner.py call shape (7 positional + transit kwarg) works."""
    acfg = real_physics["acfg"]
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, omega_rf_base = real_physics["omegas"]
    dp_scan_mhz = np.linspace(-params.SCAN_RANGE_MHZ, params.SCAN_RANGE_MHZ,
                              params.N_SCAN)

    out = ss.simulate_spectrum(
        0.0,
        dp_scan_mhz,
        (omega_p, omega_c, 0.0), (dp, dc, drf),
        real_physics["gammas"], real_physics["dephasing"],
        acfg,                                 # runner passes a bundle; inert (R-2A)
        transit_rate_per_us=0.0,
    )
    assert len(out) == 3
    assert len(out[0]) == params.N_SCAN


def test_keyword_call_form(real_physics, cold_path, mocked_solver):
    """All-keyword call form resolves identically."""
    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]

    scan, t_num, t_ana = ss.simulate_spectrum(
        omega_rf=0.0,
        detuning_list_mhz=[0.0],
        p_omegas=(omega_p, omega_c, 0.0),
        p_deltas=(dp, dc, drf),
        p_gammas=real_physics["gammas"],
        p_dephasing=real_physics["dephasing"],
        absorption_const=None,
    )
    assert len(scan) == 1
    assert np.isfinite(t_num)
    assert np.isfinite(t_ana)
