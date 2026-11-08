"""BEAM-002 â€” laser beam-size (1/eÂ² radius) convention contract tests.

BEAM-001 established that the historical 0.160/0.244 mm values used as ARC
``laserWaist`` were actually the 1/eÂ² DIAMETERS of the Simons experimental
FWHM beams (94/144 Âµm), i.e. a factor-of-two beam-radius error.

BEAM-002 establishes ONE source of truth:

    BEAM_FWHM_DIAMETER_UM      (experimental FWHM full-width diameters, Âµm)
        -> gaussian_fwhm_diameter_to_1e2_radius_mm  (the ONLY conversion)
        -> LASER_PARAMS['probe|coupling']['beam_waist_mm']  (1/eÂ² RADIUS, mm)
        -> ARC ``laserWaist`` (radius [m] after Ã—1e-3)

The derived radii (79.836/122.302 Âµm) are never hard-coded anywhere else.
"""

import math
import numpy as np
import pytest

from rydberg_aoa.config.laser_rf_config import (
    LASER_PARAMS,
    BEAM_FWHM_DIAMETER_UM,
    gaussian_fwhm_diameter_to_1e2_radius_mm,
)

S2LN2 = math.sqrt(2.0 * math.log(2.0))
HBAR = 1.054571817e-34
EPS0 = 8.8541878128e-12
C_L = 299792458.0


def e_peak_laser(power_w, w0_m):
    """I0 = 2P/(Ï€ w0Â²), E0 = sqrt(2 I0 / (c Îµ0)) â€” 1/eÂ² radius convention."""
    i0 = 2.0 * power_w / (math.pi * w0_m ** 2)
    return math.sqrt(2.0 * i0 / (C_L * EPS0))


def e_peak_oracle_4p(power_w, w0_m):
    """Algebraically equivalent form: sqrt(4P / (Ï€ w0Â² c Îµ0))."""
    return math.sqrt(4.0 * power_w / (math.pi * w0_m ** 2 * C_L * EPS0))


# ---------------------------------------------------------------------------
# 1. Independent conversion tests (TEST A-D)
# ---------------------------------------------------------------------------
def test_probe_fwhm_to_radius():
    """TEST A: 94 Âµm FWHM -> 79.836... Âµm 1/eÂ² radius."""
    w0_um = gaussian_fwhm_diameter_to_1e2_radius_mm(94.0) * 1000.0
    assert w0_um == pytest.approx(94.0 / S2LN2, rel=0, abs=1e-9)
    assert w0_um == pytest.approx(79.8362, rel=0, abs=5e-4)


def test_coupling_fwhm_to_radius():
    """TEST B: 144 Âµm FWHM -> 122.302... Âµm 1/eÂ² radius."""
    w0_um = gaussian_fwhm_diameter_to_1e2_radius_mm(144.0) * 1000.0
    assert w0_um == pytest.approx(144.0 / S2LN2, rel=0, abs=1e-9)
    assert w0_um == pytest.approx(122.3023, rel=0, abs=5e-4)


def test_reverse_relation_radius_to_fwhm():
    """TEST C: w0Â·sqrt(2 ln 2) == supplied FWHM diameter."""
    for fwhm in (94.0, 144.0):
        w0_mm = gaussian_fwhm_diameter_to_1e2_radius_mm(fwhm)
        assert w0_mm * 1000.0 * S2LN2 == pytest.approx(fwhm, rel=0, abs=1e-9)


def test_diameter_guard():
    """TEST D: 2Â·w0 == ~159.67 Âµm / ~244.60 Âµm â€” radius/diameter regression guard."""
    assert 2.0 * gaussian_fwhm_diameter_to_1e2_radius_mm(94.0) * 1000.0 == \
        pytest.approx(159.6725, rel=0, abs=5e-4)
    assert 2.0 * gaussian_fwhm_diameter_to_1e2_radius_mm(144.0) * 1000.0 == \
        pytest.approx(244.6047, rel=0, abs=5e-4)


# ---------------------------------------------------------------------------
# 2. Configuration contract â€” one source of truth, no hard-coded radii
# ---------------------------------------------------------------------------
def test_config_radii_are_derived_from_fwhm_single_source():
    """LASER_PARAMS radii must equal the centralized helper output (no drift)."""
    for key in ("probe", "coupling"):
        derived = gaussian_fwhm_diameter_to_1e2_radius_mm(
            BEAM_FWHM_DIAMETER_UM[key])
        assert LASER_PARAMS[key]["beam_waist_mm"] == pytest.approx(
            derived, rel=0, abs=0)
    # corrected radii are RADII (~0.08 / ~0.12 mm), not the old 0.16/0.244
    assert LASER_PARAMS["probe"]["beam_waist_mm"] < 0.1
    assert LASER_PARAMS["coupling"]["beam_waist_mm"] < 0.15
    assert BEAM_FWHM_DIAMETER_UM == {"probe": 94.0, "coupling": 144.0}

# ---------------------------------------------------------------------------
# 3. ARC Rabi path — corrected radii reach the production chain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    from arc import Rubidium85
    from rydberg_aoa.config import atomic_config as acfg
    return {"atom": Rubidium85(), "acfg": acfg}


def test_optical_rabi_arc_chain_matches_independent_oracle(bundle, atomic_physics):
    """Production \u03a9 (ARC path, corrected radii) == mu·E0(P, w0)/\u0127 oracle."""
    acfg, atom = bundle["acfg"], bundle["atom"]
    dm = acfg.get_cached_dipole_matrix()
    from rydberg_aoa.config.laser_rf_config import calculate_laser_rabi_frequency

    for name, lvl1, lvl2, mu, power_mw in (
        ("probe", acfg.ATOMIC_LEVELS["ground"], acfg.ATOMIC_LEVELS["excited"],
         dm["probe"]["dipole_si"], LASER_PARAMS["probe"]["power_mw"]),
        ("coupling", acfg.ATOMIC_LEVELS["excited"], acfg.ATOMIC_LEVELS["rydberg1"],
         dm["coupling"]["dipole_si"], LASER_PARAMS["coupling"]["power_mw"]),
    ):
        w0_m = LASER_PARAMS[name]["beam_waist_mm"] * 1e-3
        assert e_peak_laser(power_mw * 1e-3, w0_m) == pytest.approx(
            e_peak_oracle_4p(power_mw * 1e-3, w0_m), rel=1e-12)
        omega_chain = calculate_laser_rabi_frequency(
            atom, lvl1, lvl2, LASER_PARAMS[name]) * 1e-6
        omega_oracle = mu * e_peak_laser(power_mw * 1e-3, w0_m) / HBAR * 1e-6
        assert omega_chain == pytest.approx(omega_oracle, rel=1e-9)


def test_optical_rabi_now_in_corrected_regime(bundle, atomic_physics):
    """Magnitude window guarding reversion to the old (half) values."""
    acfg = bundle["acfg"]
    assert acfg.OMEGA_P == pytest.approx(15.7585, rel=0, abs=0.01)
    assert acfg.OMEGA_C == pytest.approx(113.0316, rel=0, abs=0.1)


def test_rf_rabi_unchanged_by_beam_correction(bundle, atomic_physics):
    """RF Rabi depends only on (mu, E-field) — beam-independent."""
    acfg = bundle["acfg"]
    dm = acfg.get_cached_dipole_matrix()
    mu_rf = dm["rf"]["dipole_si"]
    from rydberg_aoa.config.laser_rf_config import LO_RF_PARAMS, SIGNAL_RF_PARAMS

    for const, params in ((acfg.OMEGA_RF_BASE, LO_RF_PARAMS),
                          (acfg.OMEGA_SIGNAL_RF_BASE, SIGNAL_RF_PARAMS)):
        expected = mu_rf * params["electric_field_vm"] / HBAR * 1e-6
        assert const == pytest.approx(expected, rel=1e-9)

# ---------------------------------------------------------------------------
# 4. Transit rate — corrected radius, independent oracle
# ---------------------------------------------------------------------------
def test_transit_rate_matches_independent_oracle():
    """Gamma_t == sqrt(2/pi)*sqrt(kB*T/m)/w0_probe with the corrected radius."""
    from rydberg_aoa.config import constants as const
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.config import atomic_config as acfg
    from rydberg_aoa.physics.lindblad import calculate_transit_rate_per_us

    w0 = LASER_PARAMS["probe"]["beam_waist_mm"] * 1e-3
    expected = (np.sqrt(2 / np.pi) * np.sqrt(const.k_B * params.VAPOR_CELL_TEMPERATURE_K
                / acfg.ATOM.mass) / w0 * 1e-6)
    got = calculate_transit_rate_per_us()
    assert got == pytest.approx(expected, rel=1e-9)
    assert got == pytest.approx(1.7129, rel=0, abs=5e-3)


def test_transit_proportional_to_1_over_w0():
    """Doubling the radius halves the rate (DPH-002B \u221d 1/w0 structure)."""
    from rydberg_aoa.physics.lindblad import calculate_transit_rate_per_us
    w0 = LASER_PARAMS["probe"]["beam_waist_mm"]
    assert calculate_transit_rate_per_us(beam_waist_mm=2.0 * w0) == pytest.approx(
        calculate_transit_rate_per_us(beam_waist_mm=w0) / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Pure-dephasing invariance (transit must NOT leak into the \u03b3\u03c6 budget)
# ---------------------------------------------------------------------------
# ARCH-001T: the dephasing budget's lazy ``rydberg_n`` default resolves
# through ATOMIC_LEVELS, so this resolved-runtime check opts in explicitly
# (the pure Gaussian-beam formula tests above need no atomic physics).
def test_pure_dephasing_invariant_under_beam_change(atomic_physics):
    """\u03b3\u03c6_{2,3,4} must not depend on the beam radius (DPH-002B split)."""
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)

    base = calculate_dephasing_rates()
    injected = calculate_dephasing_rates(beam_waist_mm=2.0)
    assert injected == pytest.approx(base, rel=0, abs=0)


# ---------------------------------------------------------------------------
# 6. DOP-003S direct-solver parity spot check (corrected operating point)
# ---------------------------------------------------------------------------
def test_direct_backend_parity_at_corrected_operating_point(bundle, atomic_physics):
    """Direct dense solver \u2261 qutip.steadystate at the new-beam point."""
    acfg = bundle["acfg"]
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)
    from rydberg_aoa.simulation.steady_state import _solve_probe_coherence

    dp, dc, drf = acfg.get_cached_detunings()
    gam = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    dep = list(calculate_dephasing_rates())
    rho_dir = _solve_probe_coherence(
        dp, dc, acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE, drf,
        gam, dep, transit_rate_per_us=0.0, steady_state_backend="direct")
    rho_qutip = _solve_probe_coherence(
        dp, dc, acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE, drf,
        gam, dep, transit_rate_per_us=0.0, steady_state_backend="qutip")
    assert abs(rho_dir - rho_qutip) <= 1e-12
