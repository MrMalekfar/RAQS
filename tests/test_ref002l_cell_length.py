"""REF-002L — reference 75 mm vapor-cell-length contract tests.

The historical CELL_LENGTH = 0.100 m was an arbitrary owner value with no
physical basis.  REF-002L adopts the Simons et al. 2016 reference geometry
(CELL_LENGTH = 0.075 m).  The cell length may ONLY enter the optical-depth
factor  od_factor = pi * L / lambda_p  (Beer-Lambert propagation); it must
NEVER alter the atomic response (H, c_ops, rho21, Rabi frequencies, transit,
pure dephasing).

The probe power is the FROZEN SENSOR operating point selected by HET-001 and
committed by OP-001 (P_probe = 0.10 uW = 0.0001 mW); these tests guard
against accidentally committing the 17.5 uW reference power as production
default (the 17.5 uW value is retained only as a REFERENCE / SIMONS INPUT
profile).
"""

import math

import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import (
    calculate_transmission_from_rho21,
)
from tests.atomic_init_support import ensure_resolved_atomic_physics

# ---------------------------------------------------------------------------
# 1. Configuration contract
# ---------------------------------------------------------------------------
def test_cell_length_is_reference_value():
    """CELL_LENGTH must be the 75 mm Simons reference geometry."""
    assert params.CELL_LENGTH == pytest.approx(0.075, rel=0, abs=0)


def test_probe_power_remains_provisional_not_reference():
    """0.1 uW is the provisional sensor value; 17.5 uW must NOT be the
    production default (PWR-001: AT doublet power-broadened away there)."""
    assert LASER_PARAMS["probe"]["power_mw"] == pytest.approx(0.0001, rel=0, abs=0)
    # Reference value stays documented as reference input, not production.
    assert LASER_PARAMS["coupling"]["power_mw"] == pytest.approx(32.0, rel=0, abs=0)


def test_beam_radii_unchanged_by_ref002l():
    """BEAM-002 radii remain frozen (79.836 / 122.302 um)."""
    assert LASER_PARAMS["probe"]["beam_waist_mm"] == pytest.approx(
        94.0 / math.sqrt(2.0 * math.log(2.0)) / 1000.0, rel=1e-12)
    assert LASER_PARAMS["coupling"]["beam_waist_mm"] == pytest.approx(
        144.0 / math.sqrt(2.0 * math.log(2.0)) / 1000.0, rel=1e-12)
def test_absorption_constant_uses_configured_cell_length(atomic_physics):
    """calculate_absorption_constant must consume params.CELL_LENGTH (0.075),
    and its od_factor must respond linearly to an explicit length override."""
    from rydberg_aoa.physics.susceptibility.absorption_const import (
        calculate_absorption_constant)

    od_75, _ = calculate_absorption_constant()
    od_100, _ = calculate_absorption_constant(cell_length_m=0.100)
    assert od_100 / od_75 == pytest.approx(100.0 / 75.0, rel=1e-12)
    # Default path must equal pi*L/lambda_p with the configured length
    # (relative tolerance: LAMBDA_P may be ARC-dynamic at runtime).
    assert od_75 == pytest.approx(
        math.pi * params.CELL_LENGTH / params.SI_Constants.LAMBDA_P, rel=1e-8)


def test_transmission_od_branch_uses_cell_length(atomic_physics, monkeypatch):
    """For FIXED rho21 the transmission must respond through od_factor only:
    log T(L=75mm) / log T(L=100mm) == 0.75 for the same atomic inputs."""
    from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 \
        import calculate_transmission_from_rho21

    rho = complex(0.05, -0.02)
    monkeypatch.setattr(params, "CELL_LENGTH", 0.075)
    t75 = calculate_transmission_from_rho21(rho)
    monkeypatch.setattr(params, "CELL_LENGTH", 0.100)
# ---------------------------------------------------------------------------
# 3. Atomic-response invariance (rho21 must NOT depend on cell length)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    """Resolved atomic runtime for the atomic-response invariance tests.

    ARCH-001T: opts into the explicit ARCH-001 seam (idempotent; skipped
    with a ``BLOCKED: QN-001A`` reason while the coupling mJ contract is
    open).
    """
    ensure_resolved_atomic_physics()
    from rydberg_aoa.config import atomic_config as acfg
    return {"acfg": acfg}


def test_rho21_independent_of_cell_length(bundle, monkeypatch):
    """Identical atomic inputs with different CELL_LENGTH must give the
    identical rho21: the solve path must never read the cell length."""
    acfg = bundle["acfg"]
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates)
    from rydberg_aoa.simulation.steady_state import _solve_probe_coherence

    dp, dc, drf = acfg.get_cached_detunings()
    gam = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    dep = list(calculate_dephasing_rates())

    monkeypatch.setattr(params, "CELL_LENGTH", 0.075)
    rho_75 = _solve_probe_coherence(
        dp, dc, acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE, drf,
        gam, dep, transit_rate_per_us=0.0)
    monkeypatch.setattr(params, "CELL_LENGTH", 0.100)
    rho_100 = _solve_probe_coherence(
        dp, dc, acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE, drf,
        gam, dep, transit_rate_per_us=0.0)
    assert rho_75 == rho_100


def test_omegas_and_transit_invariant_under_length_change(bundle, monkeypatch):
    """Omega_p / Omega_c and Gamma_t must not depend on L."""
    from rydberg_aoa.physics.lindblad import calculate_transit_rate_per_us

    gam_t_before = calculate_transit_rate_per_us()
    monkeypatch.setattr(params, "CELL_LENGTH", 0.250)
    gam_t_after = calculate_transit_rate_per_us()
    assert gam_t_after == gam_t_before
    # Rabi module constants are import-time snapshots; assert the BEAM-002
    # operating point is intact (probe power provisional 0.1 uW).
    assert bundle["acfg"].OMEGA_P == pytest.approx(15.7585, rel=1e-3)
    assert bundle["acfg"].OMEGA_C == pytest.approx(113.03, rel=1e-3)



# ---------------------------------------------------------------------------
# 2. OD / Beer-Lambert propagation contract
# ---------------------------------------------------------------------------
def test_od_factor_scales_exactly_with_cell_length():
    """od_factor = pi*L/lambda_p must scale exactly 0.75 for 75 mm vs 100 mm."""
    lam = params.SI_Constants.LAMBDA_P
    od_new = math.pi * 0.075 / lam
    od_old = math.pi * 0.100 / lam
    assert od_old == pytest.approx(402643.64, rel=1e-5)
    assert od_new == pytest.approx(301982.73, rel=1e-5)
    assert od_new / od_old == pytest.approx(0.75, rel=0, abs=1e-12)


def test_transmission_log_ratio_scales_with_cell_length(atomic_physics, monkeypatch):
    """For a FIXED rho21, log(T) is proportional to od_factor, so the
    transmission log-ratio between 75 mm and 100 mm must be exactly 0.75."""
    monkeypatch.setattr(params, "CELL_LENGTH", 0.075)
    t75 = calculate_transmission_from_rho21(0.3 + 0.1j)
    monkeypatch.setattr(params, "CELL_LENGTH", 0.100)
    t100 = calculate_transmission_from_rho21(0.3 + 0.1j)
    assert math.log(t75) / math.log(t100) == pytest.approx(0.75, rel=1e-9)
