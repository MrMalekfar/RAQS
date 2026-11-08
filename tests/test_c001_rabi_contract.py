"""C-001 regression tests: Rabi unit contract and ARC argument contract.

Independent oracles — the production Rabi helper is NEVER used to generate
an expected value:

    Omega_expected [rad/us] = mu * E_peak / hbar * 1e-6

with mu taken from the independently verified dipole cache (PV-3: probe =
hyperfine-resolved 5S1/2 F=3,mF=0 -> 5P3/2 F'=4,mF'=0 element) and E_peak
computed in this file from the documented laser/RF field parameters.

These tests fail if (a) a spurious 2pi is reintroduced anywhere in the
chain, (b) the ARC getRabiFrequency positional contract drifts again,
or (c) the configured probe HFS element is silently replaced by the
generic fine-structure value.
"""
import inspect

import numpy as np
import pytest
from scipy import constants as sc

from rydberg_aoa.config.laser_rf_config import (
    LASER_PARAMS,
    LO_RF_PARAMS,
    SIGNAL_RF_PARAMS,
    calculate_laser_rabi_frequency,
    calculate_rf_rabi_frequency,
    get_signal_rf_rabi_frequency,
)

HBAR = sc.hbar
EA0_SI = 8.478e-30  # e*a0 in C*m (dipole.py conversion)


def _e_peak_laser(power_w, waist_m):
    """Independent peak field of a Gaussian beam: E = sqrt(2 I0 / (eps0 c))."""
    I0 = 2.0 * power_w / (np.pi * waist_m ** 2)
    return np.sqrt(2.0 * I0 / (sc.epsilon_0 * sc.c))


@pytest.fixture(scope="module")
def bundle():
    from arc import Rubidium85
    from rydberg_aoa.config import atomic_config as acfg
    return {"atom": Rubidium85(), "acfg": acfg}


# ---------------------------------------------------------------------------
# TEST 1 + 6 — unit-contract oracle / production Rabi scale
# ---------------------------------------------------------------------------
def test_rabi_chain_matches_independent_mu_E_over_hbar(bundle, atomic_physics):
    """Every production Rabi value must equal mu*E_peak/hbar with exactly
    ONE s->us conversion. A reintroduced 2pi makes this ratio 2pi."""
    acfg = bundle["acfg"]
    atom = bundle["atom"]
    dm = acfg.get_cached_dipole_matrix()

    cases = [
        ("probe",
         calculate_laser_rabi_frequency(atom, acfg.ATOMIC_LEVELS['ground'],
                                        acfg.ATOMIC_LEVELS['excited'],
                                        LASER_PARAMS['probe']) * 1e-6,
         dm['probe']['dipole_si'],
         _e_peak_laser(LASER_PARAMS['probe']['power_mw'] * 1e-3,
                       LASER_PARAMS['probe']['beam_waist_mm'] * 1e-3),
         acfg.OMEGA_P),
        ("coupling",
         calculate_laser_rabi_frequency(atom, acfg.ATOMIC_LEVELS['excited'],
                                        acfg.ATOMIC_LEVELS['rydberg1'],
                                        LASER_PARAMS['coupling']) * 1e-6,
         dm['coupling']['dipole_si'],
         _e_peak_laser(LASER_PARAMS['coupling']['power_mw'] * 1e-3,
                       LASER_PARAMS['coupling']['beam_waist_mm'] * 1e-3),
         acfg.OMEGA_C),
        ("RF_LO",
         calculate_rf_rabi_frequency(atom, acfg.ATOMIC_LEVELS['rydberg1'],
                                     acfg.ATOMIC_LEVELS['rydberg2'],
                                     LO_RF_PARAMS) * 1e-6,
         dm['rf']['dipole_si'],
         LO_RF_PARAMS['electric_field_vm'],
         acfg.OMEGA_RF_BASE),
        ("RF_signal",
         get_signal_rf_rabi_frequency(atom, acfg.ATOMIC_LEVELS['rydberg1'],
                                      acfg.ATOMIC_LEVELS['rydberg2'],
                                      SIGNAL_RF_PARAMS) * 1e-6,
         dm['rf']['dipole_si'],
         SIGNAL_RF_PARAMS['electric_field_vm'],
         acfg.OMEGA_SIGNAL_RF_BASE),
    ]
    for name, omega_rad_us, mu, E, omega_stored in cases:
        expected = mu * E / HBAR * 1e-6
        ratio = omega_rad_us / expected
        assert ratio == pytest.approx(1.0, rel=1e-9), (
            f"{name}: chain returned {omega_rad_us:.6f} rad/us but the "
            f"independent mu*E/hbar oracle is {expected:.6f} rad/us "
            f"(ratio {ratio:.6f} — 2pi would give {2*np.pi:.6f})")
        # the stored config value must equal the chain value (no second
        # conversion anywhere between helper and atomic_config)
        assert omega_stored == pytest.approx(omega_rad_us, rel=1e-9), (
            f"{name}: stored value {omega_stored:.6f} != chain value "
            f"{omega_rad_us:.6f} — an extra conversion exists downstream")


def test_display_relation_mhz_is_omega_over_2pi(bundle, atomic_physics):
    """Stored values are ANGULAR (rad/us); f[MHz] = Omega/(2pi) is display
    only. The independent frequency oracle is (mu E/hbar)/(2pi)."""
    acfg = bundle["acfg"]
    mu = acfg.get_cached_dipole_matrix()['rf']['dipole_si']
    omega_signal = mu * SIGNAL_RF_PARAMS['electric_field_vm'] / HBAR * 1e-6
    f_mhz_oracle = omega_signal / (2.0 * np.pi)
    f_mhz_display = acfg.OMEGA_SIGNAL_RF_BASE / (2.0 * np.pi)
    assert f_mhz_display == pytest.approx(f_mhz_oracle, rel=1e-9)
    # stored value must be the angular quantity, not the MHz display value
    assert acfg.OMEGA_SIGNAL_RF_BASE == pytest.approx(omega_signal, rel=1e-9)
    assert acfg.OMEGA_SIGNAL_RF_BASE > f_mhz_display


# ---------------------------------------------------------------------------
# TEST 3 — probe HFS preservation
# ---------------------------------------------------------------------------
def test_probe_uses_configured_hyperfine_element(bundle, atomic_physics):
    """The probe Rabi must come from the configured hyperfine transition
    5S1/2 F=3,mF=0 -> 5P3/2 F'=4,mF'=0 — NOT from the generic fine-structure
    ARC call (~8% difference is angular-momentum structure, not error)."""
    acfg = bundle["acfg"]
    atom = bundle["atom"]

    mu_hfs = abs(atom.getDipoleMatrixElementHFS(
        5, 0, 0.5, 3.0, 0, 5, 1, 1.5, 4.0, 0, 0)) * EA0_SI
    cached = acfg.get_cached_dipole_matrix()['probe']['dipole_si']
    assert cached == pytest.approx(mu_hfs, rel=1e-9)

    P = LASER_PARAMS['probe']['power_mw'] * 1e-3
    w = LASER_PARAMS['probe']['beam_waist_mm'] * 1e-3
    omega_hfs = mu_hfs * _e_peak_laser(P, w) / HBAR * 1e-6
    assert acfg.OMEGA_P == pytest.approx(omega_hfs, rel=1e-6)

    # generic fine-structure element (mj=+1/2 pair) is a DIFFERENT physical
    # transition element; the production probe value must NOT match it
    mu_fine = atom.getDipoleMatrixElement(
        5, 0, 0.5, 0.5, 5, 1, 1.5, 0.5, 0) * EA0_SI
    omega_fine = mu_fine * _e_peak_laser(P, w) / HBAR * 1e-6
    assert abs(acfg.OMEGA_P - omega_fine) / acfg.OMEGA_P > 0.05, (
        "probe Rabi silently switched to the generic fine-structure value")


# ---------------------------------------------------------------------------
# TEST 4 — ARC argument contract
# ---------------------------------------------------------------------------
def test_arc_getrabi_frequency_contract(bundle):
    """Pin the installed ARC signature (no mj2 positional parameter) and
    prove the corrected keyword call succeeds for a valid element."""
    atom = bundle["atom"]
    sig = inspect.signature(atom.getRabiFrequency)
    names = list(sig.parameters)
    assert "mj2" not in names
    for required in ("n1", "l1", "j1", "mj1", "n2", "l2", "j2",
                     "q", "laserPower", "laserWaist"):
        assert required in names

    # keyword call identical to the production call shape (valid mj pair)
    om = atom.getRabiFrequency(
        n1=5, l1=0, j1=0.5, mj1=0.5,
        n2=5, l2=1, j2=1.5,
        q=0, laserPower=1e-7, laserWaist=1.6e-4)
    assert om > 0

    # the production source must use keyword arguments and must not pass
    # to_level.mj into getRabiFrequency (positional-drift guard)
    from rydberg_aoa.arc import rabi_frequency as rf
    src = inspect.getsource(rf.calculate_rabi_frequency)
    assert "laserWaist=waist_m" in src
    call_region = src.split("atom.getRabiFrequency(")[1].split(")")[0]
    assert "to_level.mj" not in call_region


# ---------------------------------------------------------------------------
# TEST 5 — ARC vs mu*E/hbar parity for the SAME matrix element
# ---------------------------------------------------------------------------
def test_arc_rabi_matches_mu_E_over_hbar_same_element(bundle):
    """For the identical (n,l,j,mj,q,P,w) element, ARC's getRabiFrequency
    and mu*E_peak/hbar must agree tightly (same element, same E convention)."""
    atom = bundle["atom"]
    P, w, q = 1e-7, 1.6e-4, 0
    d_si = abs(atom.getDipoleMatrixElement(
        5, 0, 0.5, 0.5, 5, 1, 1.5, 0.5, q)) * EA0_SI
    expected = d_si * _e_peak_laser(P, w) / HBAR          # rad/s
    got = atom.getRabiFrequency(
        n1=5, l1=0, j1=0.5, mj1=0.5,
        n2=5, l2=1, j2=1.5,
        q=q, laserPower=P, laserWaist=w)                   # rad/s
    assert got == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# TEST 7 — no double conversion into the Hamiltonian
# ---------------------------------------------------------------------------
def test_hamiltonian_receives_angular_rabi_exactly_once(bundle, atomic_physics):
    """create_sensor_model must receive the stored rad/us value unchanged:
    the probe off-diagonal of H equals -Omega_P/2 with Omega_P as stored."""
    acfg = bundle["acfg"]
    from rydberg_aoa.physics.hamiltonian.create_sensor_model import (
        create_sensor_model)
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)
    dp0, dc0, drf0 = acfg.get_cached_detunings()
    gam = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    deph = calculate_dephasing_rates()
    H, c_ops, obs = create_sensor_model(
        acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE,
        dp0, dc0, drf0, gam, deph)
    h_full = H.full()
    assert h_full[1, 0] == pytest.approx(-acfg.OMEGA_P / 2.0)
    assert h_full[2, 1] == pytest.approx(-acfg.OMEGA_C / 2.0)
    assert h_full[3, 2] == pytest.approx(-acfg.OMEGA_RF_BASE / 2.0)

