"""Phase-0 frozen-baseline regression tests (RF-006 coverage migration).

Protects the physics-critical scalar chain end-to-end through the PUBLIC
pipeline, using exactly the same accessor paths as _phase0_capture.py so the
test and the frozen snapshot can never diverge in methodology.

Reference: phase0_baseline.json (Phase 0, P0 refactor plan). Repeated Phase-0
captures were bitwise identical and post-RF captures compared IDENTICAL
(max_abs=0.000e+00); exact equality is therefore the observationally
established criterion. Relative tolerance 1e-9 (1e-6 for the solver-derived
steady-state block) guards only JSON/float round-trip drift -- never physics.
"""
import json
import os

import numpy as np
import pytest

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "phase0_baseline.json")

LEVEL_FIELDS = ('n', 'l', 'j', 'f', 'mj', 'mf', 'energy', 'label')


def _load_baseline():
    """Load the frozen baseline snapshot; fail loudly if missing."""
    with open(os.path.abspath(BASELINE_PATH), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def baseline():
    return _load_baseline()


def _atomic_config():
    # ARCH-001T: importing atomic_config publishes nothing; every test that
    # calls this helper requests the explicit ``atomic_physics`` fixture,
    # which runs the ARCH-001 seam (or skips with BLOCKED: QN-001A).
    from rydberg_aoa.config import atomic_config as acfg
    return acfg


def test_level_selection_unchanged(baseline, atomic_physics):
    """ATOMIC_LEVELS (n, l, j, f, mj, mf, E, label) match the frozen snapshot."""
    acfg = _atomic_config()
    exp = baseline['levels']
    for key in ('ground', 'excited', 'rydberg1', 'rydberg2'):
        for field in LEVEL_FIELDS:
            got = acfg.ATOMIC_LEVELS[key].get(field)
            want = exp[key][field]
            if isinstance(want, float):
                assert got == pytest.approx(want, rel=1e-9), (
                    f"ATOMIC_LEVELS[{key}][{field}] drifted")
            else:
                assert got == want, f"ATOMIC_LEVELS[{key}][{field}] drifted"


def test_wavelengths_unchanged(baseline, atomic_physics):
    """lambda_p [m] from SI constants; lambda_c [nm] from optimized LASER_PARAMS."""
    _atomic_config()
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
    # lambda_p: authoritative dynamic value updated during import (capture path)
    assert float(params.SI_Constants.LAMBDA_P) == pytest.approx(
        baseline['probe_lambda_m'], rel=1e-9)
    # lambda_c: overwritten at import with the optimizer-calculated coupling value
    assert float(LASER_PARAMS['coupling']['wavelength_nm']) == pytest.approx(
        baseline['coupling_wavelength_nm'], rel=1e-9)


def test_dipoles_unchanged(baseline, atomic_physics):
    """ARC dipole matrix mu_ij [C*m] from the public self-healing cache."""
    acfg = _atomic_config()
    dm = acfg.get_cached_dipole_matrix()
    assert dm, "dipole matrix cache empty"
    for t in ('probe', 'coupling', 'rf'):
        assert dm[t]['dipole_si'] == pytest.approx(
            baseline['dipoles_si'][t], rel=1e-9), f"mu_{t} drifted"


def test_decay_and_rabi_unchanged(baseline, atomic_physics):
    """Gamma_21/32/43 and Omega_P/C/RF_BASE/SIGNAL [rad/us] vs frozen values."""
    acfg = _atomic_config()
    for name in ('GAMMA_21', 'GAMMA_32', 'GAMMA_43'):
        assert getattr(acfg, name) == pytest.approx(
            baseline['gammas_rad_per_us'][name], rel=1e-9), f"{name} drifted"
    for name in ('OMEGA_P', 'OMEGA_C', 'OMEGA_RF_BASE', 'OMEGA_SIGNAL_RF_BASE'):
        assert getattr(acfg, name) == pytest.approx(
            baseline['omegas_rad_per_us'][name], rel=1e-9), f"{name} drifted"


def test_detunings_including_valid_zero_unchanged(baseline, atomic_physics):
    """Delta_p/Delta_rf match; Delta_c == 0.0 is a valid zero (RF-003 contract)."""
    acfg = _atomic_config()
    dp, dc, drf = acfg.get_cached_detunings()
    exp = baseline['detunings_rad_per_us']
    assert dp == pytest.approx(exp['delta_p'], rel=1e-9)
    assert dc == pytest.approx(exp['delta_c'], abs=0.0)  # valid zero preserved
    assert drf == pytest.approx(exp['delta_rf'], rel=1e-9)


def test_dephasing_unchanged(baseline, atomic_physics):
    """gamma_2/3/4 dephasing rates [rad/us] via the public Lindblad helper."""
    from rydberg_aoa.physics.lindblad import calculate_dephasing_rates
    g2, g3, g4 = calculate_dephasing_rates()
    exp = baseline['dephasing_rates_rad_per_us']
    assert g2 == pytest.approx(exp[0], rel=1e-9)
    assert g3 == pytest.approx(exp[1], rel=1e-9)
    assert g4 == pytest.approx(exp[2], rel=1e-9)


def test_absorption_constants_unchanged(baseline, atomic_physics):
    """(od_factor, K_chi) from the susceptibility absorption helper."""
    from rydberg_aoa.physics.susceptibility.absorption_const import (
        calculate_absorption_constant)
    od, kchi = calculate_absorption_constant()
    exp = baseline['absorption']
    assert od == pytest.approx(exp['od_factor'], rel=1e-6)
    assert kchi == pytest.approx(exp['K_chi'], rel=1e-6)


@pytest.mark.slow
def test_steady_state_observables_unchanged(baseline, atomic_physics):
    """End-to-end Liouvillian check at Delta_p + {0, 2*pi*6 MHz} (capture method)."""
    from qutip import steadystate
    from rydberg_aoa.physics.hamiltonian import create_sensor_model
    from rydberg_aoa.physics.lindblad import calculate_dephasing_rates
    from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import (
        calculate_transmission_from_rho21)
    _atomic_config()
    om = baseline['omegas_rad_per_us']
    de = baseline['detunings_rad_per_us']
    ga = baseline['gammas_rad_per_us']
    dephasing = list(calculate_dephasing_rates())
    for label, dr_mhz in (('res', 0.0), ('off', 6.0)):
        # Delta_p offset: resonance run uses the cached Delta_p, off-resonance
        # shifts by 2*pi*6 MHz (angular conversion, as in the frozen capture).
        dr_rad = 2 * np.pi * dr_mhz
        H, c_ops, obs = create_sensor_model(
            om['OMEGA_P'], om['OMEGA_C'], om['OMEGA_RF_BASE'],
            de['delta_p'] + dr_rad, de['delta_c'], de['delta_rf'],
            [ga['GAMMA_21'], ga['GAMMA_32'], ga['GAMMA_43']],
            dephasing,
        )
        rho_ss = steadystate(H, c_ops)
        r21 = (obs * rho_ss).tr()
        T = calculate_transmission_from_rho21(r21)
        exp = baseline['steady_state'][label]
        assert np.real(r21) == pytest.approx(exp['rho21_re'], rel=1e-6)
        assert np.imag(r21) == pytest.approx(exp['rho21_im'], rel=1e-6)
        assert T == pytest.approx(exp['transmission'], rel=1e-6)