"""PHY-SW-001 — physical-effect switch tests.

Contract under test (NO parameters.py edits, NO re-imports — all switches
are explicit per-call arguments):

  A. Default invariance — all switches default (None/True) reproduce the
     pre-PHY-SW-001 numerics (golden values captured at the frozen OP-001
     sensor configuration before the switch implementation landed).
  B. Doppler OFF — exactly one v=0 response; Maxwell integration never runs.
  C. Transit OFF — no DPH-002B reset collapse operators; the switch wins
     over an explicit rate; Gamma_t itself stays computable.
  D/E/F. Dephasing component gates remove exactly their own contribution.
  G. Independence — gates never disable each other.
  H. Analytical parity profile — production-declared switch state.

Golden reference (pre-PHY-SW-001, Python 3.11, frozen OP-001 parameters:
P_probe=0.10 uW, E_LO=1.5 V/m, level-optimization center):
    rho_on_center  = 0.00268438955877501 - 0.15897870788738547j
    rho_off_center = -0.001445101578547911 - 0.004187288433885707j
    T_full_center  = 0.2298311742746899
"""

import pytest

from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import parameters as params
from rydberg_aoa.physics.lindblad import (
    calculate_dephasing_rates,
    calculate_transit_rate_per_us,
    resolve_effective_transit_rate,
)
from rydberg_aoa.physics.susceptibility.calculate_transmission_from_rho21 import (
    calculate_transmission_from_rho21,
)
from rydberg_aoa.simulation import steady_state
from rydberg_aoa.simulation.effect_switches import (
    ANALYTICAL_PARITY_EFFECTS,
    analytical_parity_effects,
    describe_effects,
    full_effects,
)
from tests.atomic_init_support import ensure_resolved_atomic_physics

GOLDEN_RHO_ON_CENTER = complex(0.00268438955877501, -0.15897870788738547)
GOLDEN_RHO_OFF_CENTER = complex(-0.001445101578547911, -0.004187288433885707)
GOLDEN_T_FULL_CENTER = 0.2298311742746899
GOLDEN_TRANSIT = 1.7129319374349739
GOLDEN_DEPH = [
    0.006283504083813117,
    0.006283504083813117,
    3.1877663353034545e-07,
]


@pytest.fixture(scope="module")
def sensor_bundle():
    """Frozen OP-001 sensor parameter bundle (level-optimization center).

    ARCH-001T: opts into the explicit ARCH-001 seam through the shared
    support helper (idempotent; skipped with a ``BLOCKED: QN-001A`` reason
    while the coupling mJ contract is open).
    """
    ensure_resolved_atomic_physics()
    dp0, dc0, drf0 = acfg.get_cached_detunings()
    return {
        "p_omegas": (acfg.OMEGA_P, acfg.OMEGA_C, 0.0),
        "p_deltas": (dp0, dc0, drf0),
        "p_gammas": [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43],
        "p_dephasing": list(calculate_dephasing_rates()),
        "dp0": dp0, "dc0": dc0, "drf0": drf0,
    }


def _one_point_spectrum(bundle, **switches):
    """Run the production 1-point scan at the center with given switches."""
    dp_scan, t_num, _ = steady_state.simulate_spectrum(
        acfg.OMEGA_RF_BASE, [0.0],
        bundle["p_omegas"], bundle["p_deltas"],
        bundle["p_gammas"], bundle["p_dephasing"], None,
        **switches
    )
    return t_num[0]


# --- A. Default invariance ----------------------------------------------------
def test_default_dephasing_budget_unchanged(atomic_physics):
    """Default dephasing budget equals the pre-PHY-SW-001 frozen values."""
    assert calculate_dephasing_rates() == pytest.approx(GOLDEN_DEPH, rel=1e-12)


def test_default_transit_resolution_unchanged(configured_atom):
    """Config-gated resolver with defaults returns the legacy rate."""
    assert resolve_effective_transit_rate() == pytest.approx(
        GOLDEN_TRANSIT, rel=1e-12)
    assert resolve_effective_transit_rate() == calculate_transit_rate_per_us()


def test_default_transit_resolution_equals_explicit_all_true(atomic_physics):
    """None-gates and explicit all-True gates agree exactly."""
    assert resolve_effective_transit_rate(
        None, transit_enabled=True) == resolve_effective_transit_rate()
    assert calculate_dephasing_rates(
        laser_linewidth_enabled=True,
        collision_dephasing_enabled=True,
    ) == calculate_dephasing_rates()


def test_single_point_solve_matches_pre_phy_sw001_golden(sensor_bundle):
    """RF-on/RF-off single-point steady-state solves hit the golden values."""
    b = sensor_bundle
    rho_on = steady_state._solve_probe_coherence(
        b["dp0"], b["dc0"], acfg.OMEGA_P, acfg.OMEGA_C,
        acfg.OMEGA_RF_BASE, b["drf0"],
        b["p_gammas"], b["p_dephasing"],
        transit_rate_per_us=GOLDEN_TRANSIT,
        steady_state_backend="direct",
    )
    rho_off = steady_state._solve_probe_coherence(
        b["dp0"], b["dc0"], acfg.OMEGA_P, acfg.OMEGA_C,
        0.0, b["drf0"],
        b["p_gammas"], b["p_dephasing"],
        transit_rate_per_us=GOLDEN_TRANSIT,
        steady_state_backend="direct",
    )
    assert rho_on == pytest.approx(GOLDEN_RHO_ON_CENTER, rel=1e-9, abs=1e-12)
    assert rho_off == pytest.approx(GOLDEN_RHO_OFF_CENTER, rel=1e-9, abs=1e-12)


def test_full_spectrum_center_matches_pre_phy_sw001_golden(sensor_bundle):
    """Full production path (adaptive Doppler, 1-point scan) hits golden T."""
    t_center = _one_point_spectrum(
        sensor_bundle, transit_rate_per_us=GOLDEN_TRANSIT)
    assert t_center == pytest.approx(GOLDEN_T_FULL_CENTER, rel=1e-9, abs=1e-12)


# --- B. Doppler OFF -----------------------------------------------------------
def test_doppler_off_is_exactly_single_v0_response(sensor_bundle, monkeypatch):
    """doppler_enabled=False: no Maxwell integration, one v=0 solve."""
    b = sensor_bundle

    def _forbidden(*args, **kwargs):
        raise AssertionError("Maxwell integration ran with doppler_enabled=False")

    monkeypatch.setattr(steady_state, "integrate_maxwell_response", _forbidden)
    t_off = _one_point_spectrum(b, doppler_enabled=False,
                                transit_rate_per_us=GOLDEN_TRANSIT)

    rho_v0 = steady_state._solve_probe_coherence(
        b["dp0"], b["dc0"], acfg.OMEGA_P, acfg.OMEGA_C,
        acfg.OMEGA_RF_BASE, b["drf0"],
        b["p_gammas"], b["p_dephasing"],
        transit_rate_per_us=GOLDEN_TRANSIT,
    )
    assert t_off == pytest.approx(
        calculate_transmission_from_rho21(rho_v0), rel=1e-12)


def test_doppler_enabled_true_overrides_global_off(sensor_bundle, monkeypatch):
    """doppler_enabled=True forces averaging even with the global flag off."""
    b = sensor_bundle
    monkeypatch.setattr(params, "DOPPLER_ENABLED", False)
    called = {"n": 0}
    real = steady_state.integrate_maxwell_response

    def _count(*args, **kwargs):
        called["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(steady_state, "integrate_maxwell_response", _count)
    _one_point_spectrum(b, doppler_enabled=True,
                        transit_rate_per_us=GOLDEN_TRANSIT)
    assert called["n"] == 1


# --- C. Transit OFF -----------------------------------------------------------
def test_transit_switch_wins_over_explicit_rate():
    assert resolve_effective_transit_rate(5.0, transit_enabled=False) == 0.0
    assert resolve_effective_transit_rate(None, transit_enabled=False) == 0.0


def test_transit_off_builds_no_reset_ops(sensor_bundle, monkeypatch):
    """transit_enabled=False: reset ops never built; default path builds them."""
    b = sensor_bundle
    import importlib
    csm_mod = importlib.import_module(
        "rydberg_aoa.physics.hamiltonian.create_sensor_model")
    calls = {"n": 0}
    real = csm_mod.create_transit_reset_collapse_ops

    def _spy(gamma_t):
        calls["n"] += 1
        return real(gamma_t)

    monkeypatch.setattr(csm_mod, "create_transit_reset_collapse_ops", _spy)
    _one_point_spectrum(b, transit_enabled=False)
    assert calls["n"] == 0, "reset ops were built with transit_enabled=False"

    # And the default path DOES build them (the spy is not vacuous).
    _one_point_spectrum(b, transit_rate_per_us=GOLDEN_TRANSIT)
    assert calls["n"] > 0


def test_transit_gamma_t_still_computable_when_disabled():
    """Gamma_t calculation remains available independent of the switch."""
    assert calculate_transit_rate_per_us() == pytest.approx(
        GOLDEN_TRANSIT, rel=1e-12)


def test_no_transit_reappearance_in_dephasing_budget(atomic_physics):
    """Disabled transit must not leak into p_dephasing (DPH-002B invariant)."""
    assert calculate_dephasing_rates() == pytest.approx(GOLDEN_DEPH, rel=1e-12)
    assert resolve_effective_transit_rate(None, transit_enabled=False) == 0.0


# --- D/E/F. Dephasing component gates -----------------------------------------
def test_laser_linewidth_off_removes_only_linewidth(atomic_physics):
    lw_off = calculate_dephasing_rates(laser_linewidth_enabled=False)
    lw_zeroed = calculate_dephasing_rates(
        gamma_probe_linewidth=0.0,
        gamma_coupling_linewidth=0.0,
        gamma_rf_linewidth=0.0,
    )
    assert lw_off == pytest.approx(lw_zeroed, rel=0.0, abs=0.0)


def test_collision_off_removes_only_collision(atomic_physics):
    col_off = calculate_dephasing_rates(collision_dephasing_enabled=False)
    n_zeroed = calculate_dephasing_rates(n_density_m3=0.0)
    assert col_off == pytest.approx(n_zeroed, rel=0.0, abs=0.0)


def test_both_dephasing_switches_off_give_zero_budget(atomic_physics):
    assert calculate_dephasing_rates(
        laser_linewidth_enabled=False,
        collision_dephasing_enabled=False,
    ) == [0.0, 0.0, 0.0]


def test_dephasing_components_independent(atomic_physics):
    """Each gate removes exactly its own contribution; sum identity is exact."""
    default = calculate_dephasing_rates()
    lw_only = calculate_dephasing_rates(collision_dephasing_enabled=False)
    col_only = calculate_dephasing_rates(laser_linewidth_enabled=False)
    for i in range(3):
        assert default[i] == lw_only[i] + col_only[i]
    # linewidth-only budget is NOT the collision-only budget (both nonzero)
    assert lw_only[0] > 0.0 and col_only[0] > 0.0


# --- G. Independence at the spectrum level -------------------------------------
def test_spectrum_level_switch_independence(sensor_bundle):
    """Dephasing gates remain active when Doppler+transit are both OFF.

    Oracle note: at the RF-on center WITHOUT transit repopulation the cell is
    optically saturated (T ~ 1e-36), so T is a degenerate oracle there; the
    effect is asserted on the probe coherence, and the switch-derived budget
    is asserted EXACTLY equal to the caller-passed budget (threading proof).
    """
    b = sensor_bundle
    transit_off = resolve_effective_transit_rate(None, transit_enabled=False)
    assert transit_off == 0.0

    deph_on = calculate_dephasing_rates()
    deph_off = calculate_dephasing_rates(
        laser_linewidth_enabled=False, collision_dephasing_enabled=False)

    def rho(pd):
        return steady_state._solve_probe_coherence(
            b["dp0"], b["dc0"], acfg.OMEGA_P, acfg.OMEGA_C,
            acfg.OMEGA_RF_BASE, b["drf0"], b["p_gammas"], pd,
            transit_rate_per_us=transit_off, steady_state_backend="direct")

    # Dephasing still matters with Doppler+transit OFF (independence).
    assert abs(rho(deph_on) - rho(deph_off)) > 1e-8

    # Threading: switch-derived budget == caller-passed budget (exact).
    scan_args = (acfg.OMEGA_RF_BASE, [0.0], b["p_omegas"], b["p_deltas"],
                 b["p_gammas"])
    _, t_switch, _ = steady_state.simulate_spectrum(
        *scan_args, deph_on, None,
        doppler_enabled=False, transit_enabled=False,
        laser_linewidth_dephasing_enabled=False,
        collision_dephasing_enabled=False)
    _, t_passed, _ = steady_state.simulate_spectrum(
        *scan_args, deph_off, None,
        doppler_enabled=False, transit_enabled=False)
    assert t_switch[0] == t_passed[0]


def test_laser_linewidth_switch_leaves_transit_intact(sensor_bundle, monkeypatch):
    """Laser-linewidth OFF does not disable the transit channel."""
    b = sensor_bundle
    import importlib
    csm_mod = importlib.import_module(
        "rydberg_aoa.physics.hamiltonian.create_sensor_model")
    calls = {"n": 0}
    real = csm_mod.create_transit_reset_collapse_ops

    def _spy(gamma_t):
        calls["n"] += 1
        return real(gamma_t)

    monkeypatch.setattr(csm_mod, "create_transit_reset_collapse_ops", _spy)
    _one_point_spectrum(b, laser_linewidth_dephasing_enabled=False)
    assert calls["n"] > 0


# --- H. Analytical parity profile ----------------------------------------------
def test_analytical_parity_profile_exact_state():
    assert analytical_parity_effects() == {
        "doppler_enabled": False,
        "transit_enabled": False,
        "laser_linewidth_dephasing_enabled": True,
        "collision_dephasing_enabled": True,
    }
    assert full_effects() == {k: True for k in ANALYTICAL_PARITY_EFFECTS}


def test_describe_effects_reports_exact_state():
    assert describe_effects() == {
        "doppler_enabled": params.DOPPLER_ENABLED,
        "transit_enabled": params.TRANSIT_ENABLED,
        "laser_linewidth_dephasing_enabled":
            params.LASER_LINEWIDTH_DEPHASING_ENABLED,
        "collision_dephasing_enabled": params.COLLISION_DEPHASING_ENABLED,
    }
    reported = describe_effects(analytical_parity_effects())
    assert reported == {k: bool(v) for k, v in ANALYTICAL_PARITY_EFFECTS.items()}


def test_unknown_effect_switch_rejected():
    with pytest.raises(ValueError):
        describe_effects({"rf_enabled": True})


def test_no_rf_switch_exists():
    """RF-off remains the Omega_RF=0 contract — no redundant flag anywhere."""
    for key in list(vars(params)):
        assert "RF_ENABLED" not in key




