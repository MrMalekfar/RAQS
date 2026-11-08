"""AN-RF-001 - analytical RF detuning follows the RF energy orientation.

Scope (the analytical branch ONLY)
----------------------------------
The Appendix-A closed form in ``rydberg_aoa.analytical.solutions`` is built
from the historical rotating frame ``H44 = Delta_p + Delta_c + Delta_rf``,
so its RF detuning must be evaluated with the same energy-orientation rule
as the numerical Hamiltonian (RF-DIR-001C):

    rf_orientation=None   -> historical ``+Delta_rf`` (bit-identical)
    direction = +1 (up)   -> historical ``+Delta_rf`` (bit-identical)
    direction = -1 (down) -> ``-Delta_rf``  (= Delta_p + Delta_c - Delta_rf)

The flip derives ONLY from the established RF-DIR-001A contract object
(``RFTransitionOrientation.is_downward``); there is no new sign flag.  The
symbolic branch (``analytical/symbolic.py``) is untouched by this ticket:
``calculate_rho21_exact`` already evaluates its cached +Delta_rf derivation
at ``_effective_delta_rf(...)`` (RF-DIR-001C, locked by
tests/test_rfdir001c_hamiltonian_direction.py).  This file never imports or
calls the symbolic solver (no ``sp.linsolve``, no symbolic-parity suite).

Parity convention note
----------------------
The historical Appendix-A closed form equals ``-conj(rho12)`` of the QuTiP
observable ``Tr[sigma21 . rho]`` - a pre-existing, ORIENTATION-INDEPENDENT
convention (verified in both directions).  Production consumes only
``Im(chi)`` (``calculate_transmission_from_rho21``), where the two branches
agree; the parity tests compare ``-conj(ana)`` vs ``num`` AND assert the
imaginary part directly.  The RF sign is the only orientation-dependent
quantity.

Scope fences: no decay / static-Hamiltonian / time-domain / signed-IF / AoA
production code changes (asserted by the confinement test).
Units: detunings and Rabi [rad/us]; decay/dephasing [1/us]; no 2*pi here.
"""

import pathlib

import numpy as np
import pytest

from rydberg_aoa.analytical import solutions
from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.arc.types import AtomicLevel
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config.laser_rf_config import RF_PARAMS
from rydberg_aoa.physics.detuning.calculate_rf_detuning import (
    calculate_rf_detuning,
)
from rydberg_aoa.simulation import steady_state as ss

# --- the established production pair (test-only prescribed ladder) --------
NU_R1_TO_R2_HZ = -1.3264910772876007e11  # ARC signed: R1=26D5/2 -> R2=27P3/2
DOWNWARD = resolve_rf_orientation(NU_R1_TO_R2_HZ)   # -1 (production)
UPWARD = resolve_rf_orientation(-NU_R1_TO_R2_HZ)    # +1 (synthetic)
DELTA_RF_GOLDEN = -0.6768797624  # rad/us, lab definition at 132.649 GHz
R1 = {"n": 26, "l": 2, "j": 2.5}  # 26D5/2, mJ=+1/2 prescribed (test-only)
R2 = {"n": 27, "l": 1, "j": 1.5}  # 27P3/2, mJ=+1/2 prescribed (test-only)


def _arc_atom():
    """Live ARC atom or a clean skip (mirrors test_rfdir001b/001c)."""
    arc = pytest.importorskip("arc")
    try:
        return arc.Rubidium85()
    except Exception as exc:
        pytest.skip(f"ARC atomic database unavailable: {exc}")

# --- deterministic matched-physics parameter set (section 6) --------------
# Doppler off / noise off / transit off by construction: the numerical side
# calls the production single-point seam directly with noise_model=None,
# transit_rate_per_us=0.0 (both branches then see ZERO transit and NO noise).
# gamma_32 = gamma_43 = 0 is applied IDENTICALLY on both sides because the
# Appendix-A polynomial contains only gamma_2 - physical effects absent from
# the analytical model are not compared (section 6 clause 2).
DP0 = 0.0                     # zero-centered scan (rad/us)
DC0 = 0.10                    # Delta_c (rad/us)
DRF0 = -0.6768797624          # Delta_rf (rad/us), current-pair magnitude
OMEGA_P0 = 0.20               # rad/us
OMEGA_C0 = 2.0                # rad/us
OMEGA_RF0 = 0.50              # rad/us
GAMMAS0 = [38.065, 0.0, 0.0]  # [gamma_21, gamma_32, gamma_43] (1/us)
DEPH0 = [0.5, 0.0, 0.0]       # [deph_2, deph_3, deph_4] (1/us)
GAMMA_2_EFF0 = GAMMAS0[0] + DEPH0[0]  # simulate_spectrum's gamma_2_eff
DP_POINTS = (0.35, -0.42, 0.0)

# Observed margins: corrected branches rel <= 1.41e-6 (threshold 1e-4),
# old +Delta_rf rule rel >= 5.9e-2 (floor 1e-3).
PARITY_TOL = 1e-4
OLD_FORMULA_FAILURE_FLOOR = 1e-3


@pytest.fixture
def synthetic_cfg(monkeypatch):
    """Deterministic zero-centered detunings + Rabi for the closed form.

    The closed form reads ``acfg.get_cached_detunings()`` / ``acfg.OMEGA_*``
    directly (historical behavior), so the published module object is patched
    per-test; ``monkeypatch`` restores it automatically (test-order safety).
    """
    monkeypatch.setattr(acfg, "get_cached_detunings", lambda: (DP0, DC0, DRF0))
    monkeypatch.setattr(acfg, "OMEGA_P", OMEGA_P0)
    monkeypatch.setattr(acfg, "OMEGA_C", OMEGA_C0)
    return acfg


def _reference_rho21(delta_p_single, delta_rf_eff, gamma_2_eff,
                     omega_rf=OMEGA_RF0):
    """Historical Appendix-A assembly with an EXPLICIT Delta_rf choice.

    Re-uses ``solutions.calculate_coefficients`` (the unchanged polynomial
    source, NOT the unit under test) but decides the Delta_rf handed to it
    explicitly - that sign decision is exactly what AN-RF-001 changes.  The
    assembly mirrors ``calculate_analytical_rho21`` operation-for-operation
    so comparisons can use bit-exact equality.
    """
    dp_center, dc, _ = acfg.get_cached_detunings()
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = solutions.calculate_coefficients(
        delta_p_single + dp_center, dc, float(delta_rf_eff),
        acfg.OMEGA_C, acfg.OMEGA_P, gamma_2_eff,
    )
    omega_rf_sq = omega_rf ** 2
    omega_rf_pow4 = omega_rf_sq ** 2
    p_real = a1 * omega_rf_pow4 + a2 * omega_rf_sq + a3
    q_imag = b1 * omega_rf_pow4 + b2 * omega_rf_sq + b3
    numerator = p_real - 1j * q_imag
    denominator = c1 * omega_rf_pow4 + c2 * omega_rf_sq + c3
    return acfg.OMEGA_P * (numerator / denominator)


def _numerical_rho12(delta_p, orientation):
    """Production single-point numerical coherence (direct backend)."""
    return complex(ss._solve_probe_coherence(
        delta_p, DC0, OMEGA_P0, OMEGA_C0, OMEGA_RF0, DRF0,
        GAMMAS0, DEPH0,
        noise_model=None, transit_rate_per_us=0.0,
        steady_state_backend="direct", rf_orientation=orientation,
    ))


def _analytical(delta_p, orientation):
    return complex(solutions.calculate_analytical_rho21(
        delta_p, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0,
        rf_orientation=orientation,
    ))


# ---------------------------------------------------------------------------
# 1. Legacy None = historical formula (section 3 backward compatibility)
# ---------------------------------------------------------------------------


def test_legacy_none_matches_the_historical_formula(synthetic_cfg):
    """rf_orientation=None must reproduce the pre-ticket evaluation exactly."""
    for dp in DP_POINTS:
        got = solutions.calculate_analytical_rho21(
            dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0)
        want = _reference_rho21(dp, +DRF0, GAMMA_2_EFF0)
        assert got == want  # bit-identical, not merely close


# ---------------------------------------------------------------------------
# 2. Upward (synthetic pair) = historical formula (section 5 requirement)
# ---------------------------------------------------------------------------


def test_upward_orientation_keeps_the_historical_evaluation(synthetic_cfg):
    """direction=+1 is bit-identical to the historical +Delta_rf branch."""
    assert UPWARD.direction == +1 and UPWARD.is_upward
    for dp in DP_POINTS:
        got = solutions.calculate_analytical_rho21(
            dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0,
            rf_orientation=UPWARD)
        assert got == _reference_rho21(dp, +DRF0, GAMMA_2_EFF0)
        legacy = solutions.calculate_analytical_rho21(
            dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0)
        assert got == legacy  # upward == legacy, bit-identical


# ---------------------------------------------------------------------------
# 3. Downward flips ONLY the RF detuning contribution (section 4)
# ---------------------------------------------------------------------------


def test_downward_evaluates_the_rf_term_with_the_flipped_sign(synthetic_cfg):
    """direction=-1 evaluates the SAME polynomial at -Delta_rf."""
    assert DOWNWARD.direction == -1 and DOWNWARD.is_downward
    for dp in DP_POINTS:
        got = solutions.calculate_analytical_rho21(
            dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0,
            rf_orientation=DOWNWARD)
        # the R2-diagonal / denominator position now carries
        # Delta_p + Delta_c - Delta_rf:
        assert got == _reference_rho21(dp, -DRF0, GAMMA_2_EFF0)
        # ...and the old fixed +Delta_rf evaluation is genuinely different:
        old = _reference_rho21(dp, +DRF0, GAMMA_2_EFF0)
        assert got != old
        assert abs(got - old) > 1e-6


def test_zero_rf_detuning_is_orientation_invariant(monkeypatch):
    """With Delta_rf = 0 the orientation rule has nothing to flip."""
    monkeypatch.setattr(acfg, "get_cached_detunings", lambda: (DP0, DC0, 0.0))
    monkeypatch.setattr(acfg, "OMEGA_P", OMEGA_P0)
    monkeypatch.setattr(acfg, "OMEGA_C", OMEGA_C0)
    dp = 0.35
    legacy = solutions.calculate_analytical_rho21(
        dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0)
    up = solutions.calculate_analytical_rho21(
        dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0, rf_orientation=UPWARD)
    dn = solutions.calculate_analytical_rho21(
        dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0, rf_orientation=DOWNWARD)
    assert legacy == up == dn


# ---------------------------------------------------------------------------
# 4. Current real pair (section 9): real ARC state, downward, lab Delta_rf
# ---------------------------------------------------------------------------


def test_current_real_pair_downward_sign_and_lab_detuning(monkeypatch):
    """Real ARC current pair (section 9) - live-ARC golden, pattern from
    test_rfdir001b/001c: reading ARC directly needs no ARCH-001
    initialization, so this sits OUTSIDE the documented QN-001A skip
    boundary and runs in every environment with the ARC database.
    """
    atom = _arc_atom()
    signed = atom.getTransitionFrequency(26, 2, 2.5, 27, 1, 1.5)
    assert signed < 0.0                      # signed atomic frequency < 0
    orient = resolve_rf_orientation(signed)
    assert orient.direction == -1 and orient.is_downward
    assert orient.carrier_hz > 0.0           # positive carrier

    f_lo = float(RF_PARAMS["frequency_hz"])
    drf_real = calculate_rf_detuning(
        atom, AtomicLevel(**R1), AtomicLevel(**R2), f_lo)
    # laboratory detuning definition unchanged (RF-DIR-001A/B, no 2*pi here):
    assert drf_real == pytest.approx(DELTA_RF_GOLDEN, rel=1e-6)

    # the analytical branch uses Delta_p + Delta_c - Delta_rf on the real pair
    monkeypatch.setattr(acfg, "get_cached_detunings",
                        lambda: (DP0, DC0, drf_real))
    monkeypatch.setattr(acfg, "OMEGA_P", OMEGA_P0)
    monkeypatch.setattr(acfg, "OMEGA_C", OMEGA_C0)
    dp = 0.35
    got = solutions.calculate_analytical_rho21(
        dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0,
        rf_orientation=orient)
    assert got == _reference_rho21(dp, -drf_real, GAMMA_2_EFF0)
    assert got != _reference_rho21(dp, +drf_real, GAMMA_2_EFF0)


# ---------------------------------------------------------------------------
# 5. Analytical vs numerical rho21 under matched supported physics (section 6)
# ---------------------------------------------------------------------------


def test_analytical_matches_the_numerical_coherence_in_both_directions(
        synthetic_cfg):
    """The corrected closed form matches the production numerical solve.

    FAILS on pre-AN-RF-001 code: the orientation keyword does not exist there
    (TypeError), and with the old fixed ``+Delta_rf`` rule the downward branch
    deviates from the numerical solve by >= 5.9e-2.
    """
    for orientation, name in ((UPWARD, "upward"), (DOWNWARD, "downward")):
        for dp in DP_POINTS:
            num = _numerical_rho12(dp, orientation)
            ana = _analytical(dp, orientation)
            # pre-existing orientation-independent Appendix-A convention:
            rel_full = abs(-np.conj(ana) - num) / abs(num)
            assert rel_full < PARITY_TOL, f"{name} dp={dp}: {rel_full:.3e}"
            # production consumes Im(chi) only - assert that directly too:
            rel_im = abs(ana.imag - num.imag) / abs(num.imag)
            assert rel_im < PARITY_TOL, f"{name} Im dp={dp}: {rel_im:.3e}"


def test_old_fixed_plus_delta_rf_fails_for_the_downward_pair(synthetic_cfg):
    """Section 6: parity MUST fail with the old fixed +Delta_rf formula.

    The pre-ticket behavior is reproduced exactly by calling the closed form
    without an orientation (the old code ignored orientation entirely).
    """
    for dp in DP_POINTS:
        num = _numerical_rho12(dp, DOWNWARD)
        ana_old = solutions.calculate_analytical_rho21(
            dp, acfg, OMEGA_RF0, gamma_2_eff=GAMMA_2_EFF0)  # old rule
        rel_old = abs(-np.conj(ana_old) - num) / abs(num)
        assert rel_old > OLD_FORMULA_FAILURE_FLOOR, (
            f"old +Delta_rf unexpectedly matches at dp={dp}: {rel_old:.3e}")
        # ...while the corrected evaluation matches by a wide margin:
        rel_new = abs(-np.conj(_analytical(dp, DOWNWARD)) - num) / abs(num)
        assert rel_new < PARITY_TOL
        assert rel_new * 1e3 < rel_old  # the sign flip is what fixes parity


# ---------------------------------------------------------------------------
# 6. One orientation object reaches BOTH branches (section 6 plumbing)
# ---------------------------------------------------------------------------


def test_simulate_spectrum_passes_one_orientation_to_both_branches(
        monkeypatch):
    """simulate_spectrum must feed the IDENTICAL rf_orientation to the
    numerical solver seam AND to the analytical closed form - the single
    source of truth required by sections 2/3 (no independent sign flag)."""
    numeric_seen, analytical_seen = [], []

    def fake_solve(*args, **kwargs):
        numeric_seen.append(kwargs.get("rf_orientation", "MISSING"))
        return 0.0 + 0.0j

    def fake_analytical(*args, **kwargs):
        analytical_seen.append(kwargs.get("rf_orientation", "MISSING"))
        return 0.0 + 0.0j

    monkeypatch.setattr(ss, "_solve_probe_coherence", fake_solve)
    monkeypatch.setattr(ss.solutions, "calculate_analytical_rho21",
                        fake_analytical)
    monkeypatch.setattr(ss, "calculate_transmission_from_rho21",
                        lambda rho: 1.0)

    common = dict(doppler_enabled=False, transit_enabled=False)
    args = (0.0, [0.0], (0.05, 0.05, 0.0), (0.0, 0.0, 0.0),
            [0.05, 0.05, 0.1], [0.01, 0.01, 0.01], 0.0)

    ss.simulate_spectrum(*args, rf_orientation=DOWNWARD, **common)
    assert numeric_seen and analytical_seen
    assert all(o is DOWNWARD for o in numeric_seen)
    assert all(o is DOWNWARD for o in analytical_seen)

    numeric_seen.clear()
    analytical_seen.clear()
    ss.simulate_spectrum(*args, rf_orientation=None, **common)
    assert numeric_seen and analytical_seen
    assert all(o is None for o in numeric_seen)   # legacy build in BOTH
    assert all(o is None for o in analytical_seen)


# ---------------------------------------------------------------------------
# 7. Scope fences: no time-domain / AoA / decay / static-H change (section 9)
# ---------------------------------------------------------------------------


def test_analytical_orientation_rule_stays_out_of_the_fenced_modules():
    """AN-RF-001 must not leak beyond the analytical branch (+ its caller).

    The tokens below are unique to this ticket's change; their absence in the
    RF-DIR-001B (decay) / RF-DIR-001C (static H) / RF-DIR-001D (signed IF,
    AoA) production modules proves those tickets' behavior is untouched.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    fenced = [
        "rydberg_aoa/simulation/time_domain.py",
        "rydberg_aoa/simulation/doa_pipeline.py",
        "rydberg_aoa/simulation/runner.py",
        "rydberg_aoa/physics/hamiltonian",
        "rydberg_aoa/arc/decay_rates.py",
        "rydberg_aoa/physics/lindblad",
        "rydberg_aoa/physics/detuning",
        "rydberg_aoa/doa",
        "rydberg_aoa/signal_processing",
    ]
    tokens = ("AN-RF-001", "calculate_analytical_rho21",
              "_effective_delta_rf", "Delta_L")
    checked = 0
    for rel in fenced:
        path = root / rel
        files = ([path] if path.is_file()
                 else sorted(p for p in path.rglob("*.py")))
        for f in files:
            text = f.read_text(encoding="utf-8")
            for tok in tokens:
                assert tok not in text, f"{tok!r} leaked into {f}"
            checked += 1
    assert checked >= 10  # the fence really scanned the tree

    # symbolic.py: audited, UNCHANGED by this ticket - it keeps only the
    # RF-DIR-001C evaluation-side rule (``_effective_delta_rf``) and must not
    # carry any AN-RF-001 construct (section 13: no symbolic refactor).
    symbolic = (root / "rydberg_aoa/analytical/symbolic.py").read_text(
        encoding="utf-8")
    assert "AN-RF-001" not in symbolic
    assert "calculate_analytical_rho21" not in symbolic
    assert "Delta_L" not in symbolic

