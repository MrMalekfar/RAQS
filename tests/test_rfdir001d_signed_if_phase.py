"""RF-DIR-001D - signed IF / heterodyne coefficient / AoA phase convention.

Scope (the D ticket ONLY):

* the laboratory IF is SIGNED: f_IF = f_signal - f_LO (MHz);
  ``abs()`` is permitted only for non-phase-sensitive uses (record length,
  display), never inside coherent dynamics;
* one phase convention, one source of truth: every sign reversal derives
  from ``rf_orientation.direction`` through ``time_domain.atomic_if_polarity``
  (no ``if_sign`` / ``phase_sign`` / ``downward_flag`` style flags);
* derived coefficient convention (independent RWA oracle, lab field
  E ~ exp(-i(w t - phi)) + exp(+i(w t - phi)), anchored to the historical
  upward convention as the hard regression of section 14):

      direction = +1 / None : H43(t) = w_s * exp(+i 2 pi f_IF t)   (legacy)
      direction = -1        : H43(t) = w_s * exp(-i 2 pi f_IF t)   (corrected)

  the injected / spatial phase phi inside w_s = |Omega| exp(+i phi) NEVER
  flips; H34(t) = conj(H43(t)) at every t (Hermitian pairing, section 15);
* the physics -> AoA boundary converts the atomic polarity ONCE per
  complex quantity (runner.process_simulation_results for the vector,
  doa_pipeline.estimate_doa_from_results for the snapshot block) so every
  estimator receives the physical +Delta_phi polarity; estimators, array
  geometry, MUSIC/ESPRIT/MVDR and the static LO Hamiltonian are untouched.

No symbolic derivation engine runs here (RF-DIR-001D section 17).
"""

import pathlib

import numpy as np
import pytest

from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa.steering import arrival_phases
from rydberg_aoa.simulation.time_domain import (
    atomic_if_polarity,
    signal_coefficient,
    signal_coefficient_dag,
    signal_phase_factor,
)

# Production pair (test-only prescribed ladder, pinned from RF-DIR-001A):
NU_R1_TO_R2_HZ = -1.3264910772876007e11  # ARC signed: R1=26D5/2 -> R2=27P3/2
DOWNWARD = resolve_rf_orientation(NU_R1_TO_R2_HZ)  # direction = -1 (production)
UPWARD = resolve_rf_orientation(-NU_R1_TO_R2_HZ)   # direction = +1 (synthetic)

F_IF_SIGNED_MHZ = 0.1          # (132.6491e9 - 132.649e9) / 1e6 = +100 kHz
PHASE = 0.37                   # rad, injected signal phase (section 9)
OMEGA_S = 0.8 * (np.cos(PHASE) + 1j * np.sin(PHASE))
T_GRID = np.linspace(0.0, 30.0, 301)


def _arc_atom():
    """Live ARC atom or a clean skip (mirrors test_rfdir001a/001b/001c)."""
    arc = pytest.importorskip("arc")
    try:
        return arc.Rubidium85()
    except Exception as exc:
        pytest.skip(f"ARC atomic database unavailable: {exc}")


def _ula(n_elements=4, az_deg=12.0):
    """Production-spacing ULA + physical arrival phases (phi = k . r)."""
    d = 299792458.0 / params.RF_SIGNAL_FREQUENCY / 8.0   # lambda / 8
    pos = np.zeros((n_elements, 2))
    pos[:, 0] = np.arange(n_elements) * d
    phi = arrival_phases(pos, params.RF_SIGNAL_FREQUENCY, az_deg)
    return pos, phi


# ---------------------------------------------------------------------------
# A. Signed IF, no abs() in the coherent path, zero-IF stationarity
# ---------------------------------------------------------------------------


def test_signed_if_is_signal_minus_lo():
    """f_IF = f_signal - f_LO = +100 kHz for the current pair (section 2)."""
    assert params.RF_SIGNAL_FREQUENCY == 132649100000.0
    assert params.RF_LO_FREQUENCY == 132649000000.0
    expected = (132649100000.0 - 132649000000.0) / 1e6
    assert params.IF_FREQ == expected          # exact, same expression
    assert params.IF_FREQ == pytest.approx(F_IF_SIGNED_MHZ, rel=1e-12)
    assert params.IF_FREQ == pytest.approx(+0.1, rel=1e-12)
    assert params.IF_FREQ > 0.0
    # record length stays positive (magnitude use of the signed IF)
    assert params.T_MAX == pytest.approx(15.0 / abs(params.IF_FREQ), rel=1e-12)
    assert params.T_MAX > 0.0


def test_no_abs_in_the_coherent_if_path():
    """abs() may appear only in the non-phase-sensitive record length (18)."""
    root = pathlib.Path(params.__file__).parent.parent
    parameters_src = (root / "config" / "parameters.py").read_text(encoding="utf-8")
    if_line = next(l for l in parameters_src.splitlines()
                   if l.startswith("IF_FREQ = "))
    assert "abs(" not in if_line              # definition itself is signed
    tmax_line = next(l for l in parameters_src.splitlines()
                     if l.startswith("T_MAX = "))
    assert "abs(" in tmax_line                # magnitude use, documented
    td_src = (root / "simulation" / "time_domain.py").read_text(encoding="utf-8")
    assert "abs(" not in td_src               # no magnitude use in dynamics


def test_zero_if_is_time_independent_for_both_orientations():
    """f_signal == f_LO -> static coefficient for BOTH orientations (7)."""
    t = T_GRID
    w = complex(OMEGA_S)
    for orientation in (None, UPWARD, DOWNWARD):
        coeff = signal_coefficient(t, w, 0.0, orientation)
        np.testing.assert_array_equal(coeff, np.full(t.shape, w, dtype=complex))
        # and the factor itself is exactly 1 for every t
        np.testing.assert_array_equal(
            signal_phase_factor(t, 0.0, orientation),
            np.ones(t.shape, dtype=complex),
        )
    # the polarity rule itself
    assert atomic_if_polarity(None) == +1.0
    assert atomic_if_polarity(UPWARD) == +1.0
    assert atomic_if_polarity(DOWNWARD) == -1.0


# ---------------------------------------------------------------------------
# B. +/- 100 kHz reversal and the injected signal phase
# ---------------------------------------------------------------------------


def test_plus_minus_100khz_reverses_the_time_exponent_exactly():
    """phi(t) = phi0 +/- 2*pi*(100 kHz)*t inspected on the COMPLEX
    coefficient directly - no FFT magnitude as oracle (section 8)."""
    f = params.IF_FREQ
    t = T_GRID
    pred_p = np.cos(2 * np.pi * f * t) + 1j * np.sin(2 * np.pi * f * t)
    pred_m = np.conj(pred_p)
    for orientation in (None, UPWARD, DOWNWARD):
        sigma = atomic_if_polarity(orientation)
        fac_p = signal_phase_factor(t, +f, orientation)    # +100 kHz
        fac_m = signal_phase_factor(t, -f, orientation)    # -100 kHz
        expected_p = pred_p if sigma > 0 else pred_m
        expected_m = pred_m if sigma > 0 else pred_p
        np.testing.assert_allclose(fac_p, expected_p, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(fac_m, expected_m, rtol=1e-13, atol=1e-15)
        # exact reversal between +IF and -IF
        np.testing.assert_allclose(fac_m, np.conj(fac_p), rtol=1e-14, atol=1e-15)

    # unwrapped phase of the production coefficient: +/- 2*pi*(0.1 MHz)*t
    for sigma, orientation in ((+1.0, UPWARD), (-1.0, DOWNWARD)):
        c_p = signal_coefficient(t, OMEGA_S, +f, orientation)
        c_m = signal_coefficient(t, OMEGA_S, -f, orientation)
        ph_p = np.unwrap(np.angle(c_p))
        ph_m = np.unwrap(np.angle(c_m))
        assert ph_p[0] == pytest.approx(PHASE, abs=1e-12)
        np.testing.assert_allclose(ph_p, PHASE + sigma * 2 * np.pi * f * t,
                                   rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(ph_m, PHASE - sigma * 2 * np.pi * f * t,
                                   rtol=0.0, atol=1e-9)


def test_injected_phase_is_never_conjugated():
    """phi_s = +0.37 rad enters as exp(+i phi) on H43 for BOTH directions
    and must be distinguished from exp(-i phi) (section 9)."""
    predicted = 0.8 * (np.cos(PHASE) + 1j * np.sin(PHASE))
    conj_predicted = np.conj(predicted)
    t = T_GRID
    for orientation in (None, UPWARD, DOWNWARD):
        sigma = atomic_if_polarity(orientation)
        # t = 0 isolates the injected phase completely
        c0 = signal_coefficient(0.0, OMEGA_S, F_IF_SIGNED_MHZ, orientation)
        assert c0 == pytest.approx(predicted, rel=1e-15, abs=1e-15)
        assert abs(c0 - conj_predicted) > 0.1 * abs(predicted)   # not -phi
        # full time trace: independent cos/sin reconstruction
        ang = 2 * np.pi * sigma * F_IF_SIGNED_MHZ * t
        reconstructed = OMEGA_S * (np.cos(ang) + 1j * np.sin(ang))
        np.testing.assert_allclose(
            signal_coefficient(t, OMEGA_S, F_IF_SIGNED_MHZ, orientation),
            reconstructed, rtol=1e-13, atol=1e-15,
        )


# ---------------------------------------------------------------------------
# C. Synthetic upward regression: legacy coefficient bit-identical (14)
# ---------------------------------------------------------------------------


def test_upward_and_legacy_are_bit_identical_to_the_historical_expression():
    t = T_GRID
    for w in (OMEGA_S, 0.8, complex(0.6, 0.0)):
        for f in (F_IF_SIGNED_MHZ, -F_IF_SIGNED_MHZ, 0.0):
            historical = w * np.exp(1j * 2 * np.pi * f * t)
            historical_dag = w.conjugate() * np.exp(-1j * 2 * np.pi * f * t)
            for orientation in (None, UPWARD):
                np.testing.assert_array_equal(
                    signal_coefficient(t, w, f, orientation), historical)
                np.testing.assert_array_equal(
                    signal_coefficient_dag(t, w, f, orientation),
                    historical_dag)


# ---------------------------------------------------------------------------
# D. Current pair, real ARC: downward golden case (section 13)
# ---------------------------------------------------------------------------


def test_current_pair_downward_matches_the_independent_rwa_derivation():
    atom = _arc_atom()
    nu = atom.getTransitionFrequency(26, 2, 2.5, 27, 1, 1.5)
    assert nu < 0.0                              # signed R1 -> R2 frequency
    orientation = resolve_rf_orientation(nu)
    assert orientation.is_downward and orientation.direction == -1

    f = (params.RF_SIGNAL_FREQUENCY - params.RF_LO_FREQUENCY) / 1e6
    assert f == pytest.approx(F_IF_SIGNED_MHZ, rel=1e-12) and f > 0.0
    assert atomic_if_polarity(orientation) == -1.0

    t = T_GRID
    sigma = -1.0
    ang = 2 * np.pi * sigma * f * t
    derived = OMEGA_S * (np.cos(ang) + 1j * np.sin(ang))   # cos/sin oracle
    production = signal_coefficient(t, OMEGA_S, f, orientation)
    np.testing.assert_allclose(production, derived, rtol=1e-13, atol=1e-15)

    # the correction is REAL: the historical upward-form coefficient differs
    historical = OMEGA_S * np.exp(1j * 2 * np.pi * f * t)
    assert np.max(np.abs(production - historical)) > 0.5 * abs(OMEGA_S)


# ---------------------------------------------------------------------------
# E. Hermitian pairing of the time-dependent drive (section 15)
# ---------------------------------------------------------------------------


def test_time_dependent_rf_drive_is_hermitian_for_every_orientation():
    w = 0.6 + 0.35j
    f = F_IF_SIGNED_MHZ
    t = T_GRID
    for orientation in (None, UPWARD, DOWNWARD):
        c43 = signal_coefficient(t, w, f, orientation)       # H43(t)
        c34 = signal_coefficient_dag(t, w, f, orientation)   # H34(t)
        np.testing.assert_array_equal(c34, np.conj(c43))
        # non-vacuous: the drive really carries an imaginary part
        assert np.max(np.abs(np.imag(c43))) > 0.1

    # independent arithmetic at a fixed t (downward, sigma = -1)
    t0 = 0.4
    ang = 2 * np.pi * (-1.0) * f * t0
    fr, fi = np.cos(ang), np.sin(ang)
    pr = w.real * fr - w.imag * fi
    pim = w.real * fi + w.imag * fr
    c43_0 = signal_coefficient(t0, w, f, DOWNWARD)
    c34_0 = signal_coefficient_dag(t0, w, f, DOWNWARD)
    assert c43_0 == complex(pr, pim)
    assert c34_0 == complex(pr, -pim)
    assert c34_0 == np.conj(c43_0)               # H(t) = H(t)^dagger
    assert c34_0 != c43_0                        # genuinely complex drive


# ---------------------------------------------------------------------------
# F. Two-channel phase difference and the physics -> AoA boundary (10-12)
# ---------------------------------------------------------------------------


def test_two_channel_phase_difference_propagation_contract():
    """phi = k.r is imposed physically; the ATOMIC chain returns
    -Delta_phi for the downward pair; the ONE boundary conversion hands
    +Delta_phi to the estimators (explicit contract, no estimator hacks)."""
    pos, phi = _ula(4, 12.0)
    dphi_phys = float(phi[1] - phi[0])
    assert dphi_phys != 0.0

    sigma = atomic_if_polarity(DOWNWARD)
    assert sigma == -1.0

    # (1) physical field phase difference from phi_i = k . r_i
    # (2) atomic/coherent convention: the raw coherent projection of
    #     Re{K * coeff(t)} at +f_IF equals A * exp(+i * sigma * phi)
    X_atomic = np.exp(1j * sigma * phi)
    dphi_atomic = float(np.angle(X_atomic[1] * np.conj(X_atomic[0])))
    assert dphi_atomic == pytest.approx(sigma * dphi_phys, abs=1e-12)
    assert dphi_atomic == pytest.approx(-dphi_phys, abs=1e-12)

    # (3) the boundary rule -> phase passed to the AoA estimator
    polarity = atomic_if_polarity(DOWNWARD)
    X_est = np.conj(X_atomic) if polarity < 0 else X_atomic
    dphi_est = float(np.angle(X_est[1] * np.conj(X_est[0])))
    assert dphi_est == pytest.approx(dphi_phys, abs=1e-12)


def test_runner_vector_boundary_applies_the_polarity_once():
    """runner.process_simulation_results normalizes the vector exactly once
    and keeps the raw atomic worker values inspectable (section 11)."""
    from rydberg_aoa.simulation.runner import SimulationRunner

    amps = np.array([0.7 + 0.4j, 0.3 - 0.9j])
    xf = np.array([0.0, 0.1])
    yf = np.array([1.0, 2.0])
    traces = [np.ones(5), np.zeros(5)]

    runner = SimulationRunner()
    runner.results = [(amps[0], xf, yf, 0, traces[0]),
                      (amps[1], xf, yf, 1, traces[1])]

    vec_legacy, _, _, tr = runner.process_simulation_results()
    np.testing.assert_array_equal(vec_legacy, amps)              # None: legacy
    vec_up, _, _, _ = runner.process_simulation_results(
        rf_orientation=UPWARD)
    np.testing.assert_array_equal(vec_up, amps)                  # +1: legacy
    vec_down, xf2, yf2, tr_down = runner.process_simulation_results(
        rf_orientation=DOWNWARD)
    np.testing.assert_array_equal(vec_down, np.conj(amps))       # -1: corrected
    np.testing.assert_array_equal(xf2, xf)
    np.testing.assert_array_equal(yf2, yf)
    np.testing.assert_array_equal(tr_down, tr)                   # real traces raw
    # the raw atomic measurement stays available in the worker tuples
    assert runner.results[0][0] == amps[0]
    assert runner.results[1][0] == amps[1]


def test_doa_boundary_conjugates_snapshots_once_for_downward():
    """Snapshots are normalized at their own production point; with the
    boundary rule the estimator input carries +Delta_phi (section 11/12)."""
    from rydberg_aoa.simulation import doa_pipeline

    pos, phi = _ula(4, 12.0)
    dphi_phys = float(phi[1] - phi[0])
    t = np.arange(0.0, 100.0, 0.1)
    # RAW downward-atomic observable: X(+f) = exp(-i phi) <=> cos(2 pi f t - phi)
    traces = [np.cos(2 * np.pi * params.IF_FREQ * t - ph) for ph in phi]
    responses = np.exp(1j * phi)   # vector branch: physical (runner's output)

    doa_down, snap_down = doa_pipeline.estimate_doa_from_results(
        array_positions=pos,
        complex_responses=responses,
        tlist_us=t,
        time_traces=traces,
        method="phase_beamforming",
        rf_orientation=DOWNWARD,
    )
    x = snap_down.X[:, 0]
    dphi_in = float(np.angle(x[1] * np.conj(x[0])))
    assert dphi_in == pytest.approx(dphi_phys, abs=1e-6)   # physical after boundary
    assert doa_down.azimuth_deg == pytest.approx(12.0, abs=1.0)

    # legacy (orientation None): snapshots keep the RAW atomic convention
    _, snap_raw = doa_pipeline.estimate_doa_from_results(
        array_positions=pos,
        complex_responses=responses,
        tlist_us=t,
        time_traces=traces,
        method="phase_beamforming",
    )
    xr = snap_raw.X[:, 0]
    dphi_raw = float(np.angle(xr[1] * np.conj(xr[0])))
    assert dphi_raw == pytest.approx(-dphi_phys, abs=1e-6)  # documented -Delta_phi


# ---------------------------------------------------------------------------
# G. Scope guards: no estimator hacks; static H / decay untouched (10, 12, 16)
# ---------------------------------------------------------------------------


def test_no_forbidden_flags_and_no_orientation_hacks_in_estimators():
    root = pathlib.Path(__file__).parents[1] / "rydberg_aoa"
    forbidden_flags = ("if_sign", "phase_sign", "downward_flag",
                       "rf_conjugate_flag")
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for flag in forbidden_flags:
            assert flag not in text, f"{flag!r} found in {py}"

    doa_dir = root / "doa"
    tokens_001d = ("RF-DIR-001D", "atomic_if_polarity",
                   "signal_coefficient", "rf_orientation")
    for py in doa_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for tok in tokens_001d:
            assert tok not in text,                 f"{tok!r} reached an estimator/snapshot module: {py}"


def test_static_hamiltonian_decay_and_atomic_config_untouched_by_001d():
    root = pathlib.Path(__file__).parents[1] / "rydberg_aoa"
    targets = [root / "physics" / "hamiltonian" / "create_sensor_model.py",
               root / "arc" / "decay_rates.py",
               root / "config" / "atomic_config.py"]
    tokens = ("RF-DIR-001D", "atomic_if_polarity",
              "signal_phase_factor", "signed_if")
    for py in targets:
        text = py.read_text(encoding="utf-8")
        for tok in tokens:
            assert tok not in text, f"{tok!r} found in {py}"
