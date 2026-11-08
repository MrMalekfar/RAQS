"""RF-DIR-001C - energy-aware rotating-frame Hamiltonian (coherent RF only).

Scope (the C ticket ONLY):

* the laboratory detuning Delta_RF = 2*pi*(f_LO - |nu_signed|) stays
  orientation-independent (RF-DIR-001A/B convention, rad/us);
* the rotating-frame state-4 diagonal is energy-oriented:

      upward / legacy : H44 = D_p + D_c + D_rf
      downward        : H44 = D_p + D_c - D_rf

  derived here from an independent two-state laboratory-frame RWA oracle
  (the co-rotating field exponential flips with the energy direction);
* rf_orientation=None and direction=+1 stay bit-identical to the historical
  build; the RF off-diagonal keeps its magnitude and carries only the
  Hermitian complex-Rabi pairing H34 = conj(H43);
* decay direction (RF-DIR-001B), signed IF / time-domain heterodyne phase
  and AoA (RF-DIR-001D) are NOT touched by this ticket.

Per RF-DIR-001C section 13 no expensive symbolic derivation and no
symbolic-parity suite runs here: the symbolic check is matrix-construction
level only.
"""

import inspect
import pathlib

import numpy as np
import pytest

from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.arc.types import AtomicLevel
from rydberg_aoa.config.laser_rf_config import RF_PARAMS
from rydberg_aoa.physics.detuning.calculate_rf_detuning import calculate_rf_detuning
from rydberg_aoa.physics.hamiltonian.create_sensor_model import create_sensor_model

# --- the established production pair (test-only prescribed ladder) --------
NU_R1_TO_R2_HZ = -1.3264910772876007e11  # ARC signed: R1=26D5/2 -> R2=27P3/2
DOWNWARD = resolve_rf_orientation(NU_R1_TO_R2_HZ)  # direction = -1 (production)
UPWARD = resolve_rf_orientation(-NU_R1_TO_R2_HZ)   # direction = +1 (synthetic)

R1 = {"n": 26, "l": 2, "j": 2.5}  # 26D5/2, mJ=+1/2 prescribed (test-only)
R2 = {"n": 27, "l": 1, "j": 1.5}  # 27P3/2, mJ=+1/2 prescribed (test-only)
DELTA_RF_GOLDEN = -0.6768797624  # rad/us at f_LO = 132.649 GHz (lab definition)

# Non-trivial parameter set: every detuning nonzero so a sign error shows.
ARGS = (0.7, 0.9, 0.6, 0.13, -0.21, -0.37, [0.3, 0.3, 0.15], [0.05, 0.04, 0.03])


def _arc_atom():
    """Live ARC atom or a clean skip (mirrors test_rfdir001a/001b)."""
    arc = pytest.importorskip("arc")
    try:
        return arc.Rubidium85()
    except Exception as exc:
        pytest.skip(f"ARC atomic database unavailable: {exc}")


def _two_state_rwa(e3, e4, omega, s, n_samples=64, t0=0.0):
    """Independent two-state laboratory-frame RWA oracle.

    Lab frame (hbar = 1): state |3> at e3, state |4> at e4, field
    E(t) proportional to exp(-i*omega*t) + exp(+i*omega*t):

        H_lab(t) = [[e3, v (e^{-iwt} + e^{+iwt})],
                    [c.c.,            e4        ]]

    Candidate co-rotating frame U(t) = diag(1, exp(i*s*omega*t)):

        H_eff(t) = U^dag H U - i U^dag dU/dt
                 = [[e3, coupling(t) * e^{i s w t}],
                    [c.c.,      e4 + s * omega     ]]

    The exact one-period sample average projects out the e^{+-2i w t}
    terms (RWA), leaving the STATIC part.  Only the frame sign s that makes
    the system degenerate at the PHYSICAL resonance omega = |e4 - e3| is
    admissible - and that sign flips with the energy direction.

    This function never calls create_sensor_model; it is the oracle.
    """
    period = 2.0 * np.pi / omega
    ts = t0 + period * np.arange(n_samples) / n_samples
    v = -0.5 * 1.7  # arbitrary real dipole coupling amplitude
    samples = []
    for t in ts:
        coupling = v * (np.exp(-1j * omega * t) + np.exp(1j * omega * t))
        coupling *= np.exp(1j * s * omega * t)
        samples.append(np.array([[e3, coupling],
                                 [np.conj(coupling), e4 + s * omega]],
                                dtype=complex))
    return np.mean(samples, axis=0)


# ---------------------------------------------------------------------------
# 1. Legacy None and direction=+1 are exactly the historical build
# ---------------------------------------------------------------------------


def test_none_and_upward_are_bit_identical():
    """Section 5/10 hard requirement: +1 (and None) keep the old H."""
    H_none, _, _ = create_sensor_model(*ARGS)
    H_up, _, _ = create_sensor_model(*ARGS, rf_orientation=UPWARD)
    np.testing.assert_array_equal(H_up.full(), H_none.full())


def test_legacy_none_reproduces_the_historical_matrix_formula():
    """Hand-built pre-001C formula - exact array equality, both ways."""
    op, oc, orf, dp, dc, drf = ARGS[:6]
    expected = np.zeros((4, 4), dtype=complex)
    expected[1, 1] = dp
    expected[2, 2] = dp + dc
    expected[3, 3] = dp + dc + drf
    expected[1, 0] = expected[0, 1] = -0.5 * op
    expected[2, 1] = expected[1, 2] = -0.5 * oc
    expected[3, 2] = expected[2, 3] = -0.5 * orf

    H_none, _, _ = create_sensor_model(*ARGS)
    np.testing.assert_array_equal(H_none.full(), expected)
    H_up, _, _ = create_sensor_model(*ARGS, rf_orientation=UPWARD)
    np.testing.assert_array_equal(H_up.full(), expected)


# ---------------------------------------------------------------------------
# 2. direction=-1 flips exactly the RF diagonal - nothing else
# ---------------------------------------------------------------------------


def test_downward_changes_only_the_state4_rf_diagonal():
    H_none, _, _ = create_sensor_model(*ARGS)
    H_down, _, _ = create_sensor_model(*ARGS, rf_orientation=DOWNWARD)
    diff = H_down.full() - H_none.full()

    mask = np.ones(diff.shape, dtype=bool)
    mask[3, 3] = False
    np.testing.assert_array_equal(diff[mask],
                                  np.zeros(diff.shape, dtype=complex)[mask])

    drf = ARGS[5]
    np.testing.assert_allclose(diff[3, 3], -2.0 * drf, rtol=1e-12, atol=1e-15)


def test_downward_h44_is_dp_plus_dc_minus_delta_rf_exactly():
    dp, dc, drf = ARGS[3], ARGS[4], ARGS[5]
    H_down, _, _ = create_sensor_model(*ARGS, rf_orientation=DOWNWARD)
    assert H_down.full()[3, 3] == (dp + dc - drf)

    H_none, _, _ = create_sensor_model(*ARGS)
    hd, hn = H_down.full(), H_none.full()
    for r, c in ((0, 1), (1, 2), (2, 3)):
        assert hd[r, c] == hn[r, c]
        assert hd[c, r] == hn[c, r]
    for k in range(3):          # states 1-3 diagonals untouched
        assert hd[k, k] == hn[k, k]


# ---------------------------------------------------------------------------
# 3. Exact three-drive resonance for every orientation (section 11)
# ---------------------------------------------------------------------------


def test_zero_detunings_are_resonant_for_every_orientation():
    zeros = [0.0, 0.0, 0.0]
    for orientation in (None, UPWARD, DOWNWARD):
        H, _, _ = create_sensor_model(
            0.7, 0.9, 0.6, 0.0, 0.0, 0.0, zeros, zeros,
            rf_orientation=orientation,
        )
        np.testing.assert_array_equal(np.real(np.diag(H.full())), np.zeros(4))

    # The sign choice cannot make one orientation off-resonant at Delta=0.
    H_up, _, _ = create_sensor_model(0.7, 0.9, 0.6, 0.0, 0.0, 0.0, zeros, zeros,
                                     rf_orientation=UPWARD)
    H_dn, _, _ = create_sensor_model(0.7, 0.9, 0.6, 0.0, 0.0, 0.0, zeros, zeros,
                                     rf_orientation=DOWNWARD)
    np.testing.assert_array_equal(H_up.full(), H_dn.full())


# ---------------------------------------------------------------------------
# 4. +1 MHz on the LO: lab Delta_RF gains exactly +2*pi (section 12)
# ---------------------------------------------------------------------------


def test_plus_one_mhz_shifts_the_lab_detuning_by_two_pi():
    atom = _arc_atom()
    f0 = float(RF_PARAMS["frequency_hz"])
    level1, level2 = AtomicLevel(**R1), AtomicLevel(**R2)
    d0 = calculate_rf_detuning(atom, level1, level2, f0)
    d1 = calculate_rf_detuning(atom, level1, level2, f0 + 1e6)
    assert d1 - d0 == pytest.approx(2.0 * np.pi, rel=1e-12)


def test_plus_one_mhz_enters_h44_with_the_orientation_sign():
    """The LAB quantity shifts by +2*pi for both orientations; only the way
    it enters the state-4 energy may depend on orientation."""
    step = 2.0 * np.pi
    op, oc, orf, dp, dc, drf = ARGS[:6]
    gammas, deph = ARGS[6], ARGS[7]
    for orientation, sign in ((None, 1.0), (UPWARD, 1.0), (DOWNWARD, -1.0)):
        H0, _, _ = create_sensor_model(op, oc, orf, dp, dc, drf, gammas, deph,
                                       rf_orientation=orientation)
        H1, _, _ = create_sensor_model(op, oc, orf, dp, dc, drf + step,
                                       gammas, deph, rf_orientation=orientation)
        assert (H1.full()[3, 3] - H0.full()[3, 3]
                == pytest.approx(sign * step, abs=1e-12))


# ---------------------------------------------------------------------------
# 5. Complex RF Rabi: H = H dagger exactly, all three orientations (section 7)
# ---------------------------------------------------------------------------


def test_hamiltonian_is_hermitian_for_complex_rf_rabi():
    omega_rf_complex = 0.6 + 0.35j
    base = (0.7, 0.9, omega_rf_complex, 0.13, -0.21, -0.37,
            [0.3, 0.3, 0.15], [0.05, 0.04, 0.03])
    for orientation in (None, UPWARD, DOWNWARD):
        H, _, _ = create_sensor_model(*base, rf_orientation=orientation)
        hm = H.full()
        np.testing.assert_allclose(hm, hm.conj().T, rtol=0.0, atol=0.0)
        # not vacuous: the imaginary part of the RF slot is really there
        assert hm[3, 2].imag != 0.0
        # section 6 pairing: H43 = conj(H34) exactly
        assert hm[3, 2] == np.conj(hm[2, 3])


# ---------------------------------------------------------------------------
# 6. Independent two-state oracle vs production (sections 4 and 8)
# ---------------------------------------------------------------------------


def test_oracle_frame_sign_is_selected_by_energy_order():
    """At exact resonance only ONE rotating frame is degenerate, and which
    one flips with the energy ordering."""
    w, e3 = 0.74, 5.0

    # upward edge: e4 > e3, resonance at w = e4 - e3
    e4_up = e3 + w
    up_static = _two_state_rwa(e3, e4_up, w, -1.0)    # |4> rotates at -w
    up_other = _two_state_rwa(e3, e4_up, w, +1.0)
    assert up_static[1, 1].real == pytest.approx(up_static[0, 0].real)
    assert up_other[1, 1].real != pytest.approx(up_other[0, 0].real)

    # downward edge: e4 < e3, resonance at w = e3 - e4
    e4_dn = e3 - w
    dn_static = _two_state_rwa(e3, e4_dn, w, +1.0)    # |4> rotates at +w
    dn_other = _two_state_rwa(e3, e4_dn, w, -1.0)
    assert dn_static[1, 1].real == pytest.approx(dn_static[0, 0].real)
    assert dn_other[1, 1].real != pytest.approx(dn_other[0, 0].real)


def test_oracle_static_part_is_time_independent_with_fixed_coupling():
    w, e3 = 0.74, 5.0
    for s in (-1.0, +1.0):
        a = _two_state_rwa(e3, e3 + 0.2, w, s, t0=0.0)
        b = _two_state_rwa(e3, e3 + 0.2, w, s, t0=0.31)
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)
        assert abs(a[0, 1]) == pytest.approx(0.5 * 1.7)  # |v| both frames


def test_production_matches_the_oracle_in_both_energy_directions():
    """Expectations come ONLY from the oracle; create_sensor_model is read
    for comparison.  In the natural (S-picture) frame the RF diagonal is
    -Delta_rf for an upward edge and +Delta_rf for a downward edge; the
    production convention is the established legacy anchor prod = -natural,
    applied consistently to BOTH directions."""
    w, e3 = 0.74, 5.0
    delta_rf = 0.23

    # upward: Delta_rf = w - (e4 - e3)   ->   e4 = e3 + (w - Delta_rf)
    e4_up = e3 + (w - delta_rf)
    nat_up = _two_state_rwa(e3, e4_up, w, -1.0)
    nat_diff_up = (nat_up[1, 1] - nat_up[0, 0]).real
    assert nat_diff_up == pytest.approx(-delta_rf)

    # downward: Delta_rf = w - (e3 - e4) ->   e4 = e3 - (w - Delta_rf)
    e4_dn = e3 - (w - delta_rf)
    nat_dn = _two_state_rwa(e3, e4_dn, w, +1.0)
    nat_diff_dn = (nat_dn[1, 1] - nat_dn[0, 0]).real
    assert nat_diff_dn == pytest.approx(+delta_rf)

    zeros_g = [0.0, 0.0, 0.0]
    base = (0.0, 0.0, 0.5, 0.0, 0.0, delta_rf, zeros_g, zeros_g)
    H_up, _, _ = create_sensor_model(*base, rf_orientation=UPWARD)
    H_dn, _, _ = create_sensor_model(*base, rf_orientation=DOWNWARD)
    prod_diff_up = H_up.full()[3, 3] - H_up.full()[2, 2]
    prod_diff_dn = H_dn.full()[3, 3] - H_dn.full()[2, 2]

    assert prod_diff_up == pytest.approx(-nat_diff_up)   # legacy anchor
    assert prod_diff_dn == pytest.approx(-nat_diff_dn)   # 001C correction
    assert prod_diff_up == pytest.approx(+delta_rf)
    assert prod_diff_dn == pytest.approx(-delta_rf)
    assert prod_diff_dn == pytest.approx(-prod_diff_up)   # pure sign flip


# ---------------------------------------------------------------------------
# 7. Current pair, real ARC: golden facts (section 9)
# ---------------------------------------------------------------------------


def test_current_pair_golden_orientation_and_lab_detuning():
    atom = _arc_atom()
    signed = atom.getTransitionFrequency(26, 2, 2.5, 27, 1, 1.5)
    assert signed < 0.0                     # R1 -> R2 is downward in energy
    carrier = abs(signed)
    assert carrier > 0.0                    # positive carrier
    assert DOWNWARD.direction == -1 and DOWNWARD.is_downward
    assert UPWARD.direction == +1 and UPWARD.is_upward
    assert (DOWNWARD.upper_ladder_index, DOWNWARD.lower_ladder_index) == (3, 4)

    f_lo = float(RF_PARAMS["frequency_hz"])
    # laboratory definition recomputed INDEPENDENTLY of the helper:
    delta_lab = 2.0 * np.pi * (abs(f_lo) - carrier) * 1e-6
    assert delta_lab == pytest.approx(DELTA_RF_GOLDEN, rel=1e-6)
    # ...and it is exactly what the production helper returns (unchanged)
    d_prod = calculate_rf_detuning(atom, AtomicLevel(**R1), AtomicLevel(**R2), f_lo)
    assert d_prod == pytest.approx(delta_lab, rel=1e-12)

    # the corrected state-4 diagonal on the real pair
    op, oc, orf, dp, dc = 0.7, 0.9, 0.6, 0.13, -0.21
    gammas, deph = [0.3, 0.3, 0.15], [0.05, 0.04, 0.03]
    H_up, _, _ = create_sensor_model(op, oc, orf, dp, dc, d_prod, gammas, deph,
                                     rf_orientation=UPWARD)
    H_dn, _, _ = create_sensor_model(op, oc, orf, dp, dc, d_prod, gammas, deph,
                                     rf_orientation=DOWNWARD)
    assert H_up.full()[3, 3] == (dp + dc + d_prod)
    assert H_dn.full()[3, 3] == (dp + dc - d_prod)


# ---------------------------------------------------------------------------
# 8. Scope guards: decay contract intact; time-domain / IF / AoA untouched
# ---------------------------------------------------------------------------


def test_rf_decay_contract_is_unchanged_by_the_hamiltonian_ticket():
    gammas, deph = [0.3, 0.3, 0.15], [0.05, 0.04, 0.03]
    # collapse operators must not depend on the detunings: no leak of the
    # 001C rf_sign logic into the RF-DIR-001B decay section
    _, c_a, _ = create_sensor_model(0.7, 0.9, 0.6, 0.13, -0.21, -0.37,
                                    gammas, deph, rf_orientation=DOWNWARD)
    _, c_b, _ = create_sensor_model(0.7, 0.9, 0.6, 1.30, 0.40, 0.90,
                                    gammas, deph, rf_orientation=DOWNWARD)
    assert len(c_a) == len(c_b)
    for ca, cb in zip(c_a, c_b):
        np.testing.assert_array_equal(ca.full(), cb.full())

    # RF-DIR-001B energy-ordered jump |4><3| still in place
    jump = c_a[2].full()
    assert jump[3, 2] == pytest.approx(np.sqrt(gammas[2]))
    assert jump[2, 3] == 0.0
    # legacy jump |3><4| for rf_orientation=None
    _, c_legacy, _ = create_sensor_model(0.7, 0.9, 0.6, 0.13, -0.21, -0.37,
                                         gammas, deph)
    assert c_legacy[2].full()[2, 3] == pytest.approx(np.sqrt(gammas[2]))
    assert c_legacy[2].full()[3, 2] == 0.0


def test_no_001c_symbol_reaches_time_domain_if_or_aoa_sources():
    """RF-DIR-001C section 15: static LO rotating frame ONLY."""
    import rydberg_aoa
    import rydberg_aoa.simulation.time_domain as time_domain

    pkg_root = pathlib.Path(rydberg_aoa.__file__).parent
    tokens = ("rf_sign", "is_downward", "is_upward", "RF-DIR-001C")
    roots = [pathlib.Path(time_domain.__file__),
             pkg_root / "signal_processing",
             pkg_root / "doa"]
    checked = 0
    for root in roots:
        assert root.exists(), f"expected scope-guard root missing: {root}"
        files = [root] if root.is_file() else list(root.rglob("*.py"))
        for py in files:
            text = py.read_text(encoding="utf-8")
            for tok in tokens:
                assert tok not in text, f"{tok!r} leaked into {py}"
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 9. Symbolic scope: matrix-construction level only (section 13)
# ---------------------------------------------------------------------------


def test_symbolic_generator_keeps_delta_rf_out_of_population_rows():
    """Premise of the exact call-time negation: the Delta_rf symbol enters
    the Lindblad generator ONLY through H44 inside state-4 coherence rows,
    never in population equations (the collapse structure carries only
    gamma/dephasing symbols).  Evaluating the cached derivation at
    -Delta_rf is therefore EXACTLY the downward-edge derivation.

    Matrix-construction level: _build_lindblad assembles sympy matrices;
    no linsolve, no lambdify, no parity suite (section 13).
    """
    from rydberg_aoa.analytical import symbolic as sym_mod

    # NOTE: the function docstring claims 3 returns; reality is 5
    # (L, rho, symbols, coherence-map, population-map) - pre-existing
    # stale docstring, recorded, not fixed by this ticket.
    L, _rho, symbols, _coh, _pop = sym_mod._build_lindblad()
    drf = symbols["drf"]

    for i in range(4):                       # populations: no Delta_rf
        assert not L[i, i].has(drf), f"population row {i} depends on Delta_rf"
    coherence_has_drf = any(L[i, j].has(drf)
                            for i in range(4) for j in range(4) if i != j)
    assert coherence_has_drf
    # explicitly in the |3><4| coherence row (H33 - H44 = -Delta_rf)
    assert L[2, 3].has(drf)


def test_effective_delta_rf_helper_contract():
    from rydberg_aoa.analytical.symbolic import _effective_delta_rf

    x = -0.6768797624
    assert _effective_delta_rf(x, None) == x          # legacy passthrough
    assert _effective_delta_rf(x, UPWARD) == x        # direction=+1
    assert _effective_delta_rf(x, DOWNWARD) == -x     # direction=-1
    arr = np.linspace(-1.0, 1.0, 5)
    np.testing.assert_array_equal(_effective_delta_rf(arr, DOWNWARD), -arr)
    np.testing.assert_array_equal(_effective_delta_rf(arr, None), arr)


def test_calculate_rho21_exact_exposes_a_trailing_orientation_kwarg():
    from rydberg_aoa.analytical.symbolic import calculate_rho21_exact

    sig = inspect.signature(calculate_rho21_exact)
    names = list(sig.parameters)
    assert names[-1] == "rf_orientation"
    assert sig.parameters["rf_orientation"].default is None
    # the historical positional prefix is intact for every old caller
    assert names[:9] == ["delta_p", "delta_c", "delta_rf",
                         "omega_p", "omega_c", "omega_rf",
                         "gammas", "dephasings", "mode"]
