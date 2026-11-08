"""AUTO-001A-R1 — RF-first manifold resolver and FS-only scan contracts.

The selection stage of the EXPERIMENTAL pipeline (ARCOptimizer) under
AUTO-001A-R1:

* ``scan_fs_rf_candidates`` enumerates ``n1`` and ``n2`` INDEPENDENTLY in
  the configured RF n-range (no adjacent-only restriction) and admits a
  pair by the FINE-STRUCTURE electric-dipole rules only — RF side
  ``|Δl| = 1, |Δj| ≤ 1`` plus the coupling-side rules against the excited
  state.  No mJ/q/polarisation scan and no dipole matrix element ever
  participates in level selection (QN-001B/C keeps dipoles in the declared,
  channel-dependent path; they attach to the winner only AFTER selection).
* ``resolve_rf_first_manifold`` ranks the scanned candidates against the
  APPLIED RF carrier with the single deterministic key
  ``(mismatch_hz, n1, n2, l1, l2, j1, j2)`` and resolves the winner's
  energy orientation from ARC's SIGNED ladder-order frequency through the
  RF-DIR-001 primitive ``arc/rf_orientation.resolve_rf_orientation``.

No optimizer math, no PRESCRIBED-mode behaviour and no physics formula is
changed: PRESCRIBED mode never reaches ARCOptimizer, and the F05 fail-fast
contracts (no fabricated manifold, no silent scan failure) stay intact.
"""

import pytest

from rydberg_aoa.arc.rf_orientation import RFTransitionOrientation
from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)
from rydberg_aoa.level_finder.arc_level_finder import (
    RF_MAX_L_TO_CHECK,
    AtomicLevel,
    resolve_rf_first_manifold,
    rf_first_ranking_key,
    scan_fs_rf_candidates,
)

#: Applied RF carrier used by the resolver tests (10 GHz).
_APPLIED_HZ = 10.0e9
#: Matching tolerance used by the scan tests (1 MHz).
_TOL_HZ = 1.0e6
#: Excited state for the coupling-side rules: a P state (|Δl| = 1 arm).
_EXCITED_J = 1.5


def _level(n, l, j):
    return AtomicLevel(n=n, l=l, j=j, label=f"{n}{'SPDF'[l]}{j:.1f}")


def _candidate(mismatch_hz=0.0, n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5,
               carrier_hz=None):
    """The exact FS-only dict shape ``scan_fs_rf_candidates`` appends."""
    if carrier_hz is None:
        carrier_hz = _APPLIED_HZ + abs(float(mismatch_hz))
    mismatch = abs(carrier_hz - _APPLIED_HZ)
    return {
        "from_level": _level(n1, l1, j1),
        "to_level": _level(n2, l2, j2),
        "transition_freq_hz": carrier_hz,
        "signed_transition_freq_hz": -carrier_hz,   # default: downward edge
        "mismatch_hz": mismatch,
        # legacy alias — the same physical quantity (| |nu| - f_applied |)
        "rf_frequency_error_hz": mismatch,
    }


class _SignedFreqAtom:
    """Minimal ARC stand-in: one signed frequency, recorded call."""

    def __init__(self, signed_hz):
        self.signed_hz = signed_hz
        self.calls = []

    def getTransitionFrequency(self, n1, l1, j1, n2, l2, j2):
        self.calls.append((n1, l1, j1, n2, l2, j2))
        return self.signed_hz


class _ExplodingAtom:
    def getTransitionFrequency(self, *quantum_numbers):
        raise RuntimeError("ARC transition database unavailable")


class _ConstantScanAtom:
    """Scan stand-in: the same signed carrier for every queried pair.

    Every ARC query is RECORDED so a test can prove which pairs (for example
    same-n pairs) actually reached frequency evaluation.
    """

    def __init__(self, signed_hz):
        self.signed_hz = signed_hz
        self.calls = 0
        self.queries = []

    def getTransitionFrequency(self, n1, l1, j1, n2, l2, j2):
        self.calls += 1
        self.queries.append((n1, l1, j1, n2, l2, j2))
        return self.signed_hz


class _SelectivePairAtom:
    """Scan stand-in: a near-carrier frequency for ONE target pair only.

    Every other structurally eligible pair is returned far outside the
    matching tolerance, so the target pair is the only possible candidate.
    """

    def __init__(self, target_qns, signed_hz, far_signed_hz):
        self.target_qns = tuple(target_qns)
        self.signed_hz = signed_hz
        self.far_signed_hz = far_signed_hz

    def getTransitionFrequency(self, n1, l1, j1, n2, l2, j2):
        if (n1, l1, j1, n2, l2, j2) == self.target_qns:
            return self.signed_hz
        return self.far_signed_hz


def _scan(signed_hz, applied_hz=_APPLIED_HZ, n_range=(20, 24),
          excited=None, tol_hz=_TOL_HZ):
    """Run ``scan_fs_rf_candidates`` with a constant-carrier ARC stand-in."""
    atom = _ConstantScanAtom(signed_hz)
    if excited is None:
        excited = _level(5, 1, _EXCITED_J)   # e.g. 5P3/2-like excited state
    result = scan_fs_rf_candidates(
        atom, n_range, excited, applied_hz, tol_hz)
    return result, atom


# ---------------------------------------------------------------------------
# Failure contracts (F05 / AUTO-001A-R1 never fabricate a manifold)
# ---------------------------------------------------------------------------
def test_empty_candidate_list_raises():
    with pytest.raises(PhysicsCalculationError) as excinfo:
        resolve_rf_first_manifold([], _SignedFreqAtom(-1.326e11), _APPLIED_HZ)
    message = str(excinfo.value)
    assert "AUTO-001A" in message and "F05" in message


def test_missing_atom_raises_configuration_error():
    with pytest.raises(PhysicsConfigurationError):
        resolve_rf_first_manifold([_candidate(1e6)], None, _APPLIED_HZ)


@pytest.mark.parametrize("carrier", [None, 0.0, -1.0,
                                     float("nan"), float("inf")])
def test_invalid_applied_carrier_raises(carrier):
    """The ranking reference is never defaulted or guessed (F05)."""
    with pytest.raises(PhysicsConfigurationError) as excinfo:
        resolve_rf_first_manifold(
            [_candidate(1e6)], _SignedFreqAtom(-(_APPLIED_HZ + 1e6)), carrier)
    assert "F05" in str(excinfo.value)


def test_zero_signed_frequency_raises_instead_of_inferring_direction():
    with pytest.raises(PhysicsCalculationError) as excinfo:
        resolve_rf_first_manifold([_candidate(1e6)],
                                  _SignedFreqAtom(0.0), _APPLIED_HZ)
    assert "F05" in str(excinfo.value)


def test_non_finite_signed_frequency_raises():
    with pytest.raises(PhysicsCalculationError):
        resolve_rf_first_manifold([_candidate(1e6)],
                                  _SignedFreqAtom(float("nan")), _APPLIED_HZ)


def test_arc_exception_is_wrapped_not_swallowed():
    with pytest.raises(PhysicsCalculationError) as excinfo:
        resolve_rf_first_manifold([_candidate(1e6)], _ExplodingAtom(),
                                  _APPLIED_HZ)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_stale_scan_carrier_is_rejected():
    """A scan-time carrier that disagrees with a fresh ARC query must fail."""
    stale = _candidate(carrier_hz=_APPLIED_HZ + 1e6)
    atom = _SignedFreqAtom(-(_APPLIED_HZ + 50e6))   # fresh ARC disagrees
    with pytest.raises(PhysicsCalculationError) as excinfo:
        resolve_rf_first_manifold([stale], atom, _APPLIED_HZ)
    assert "inconsistent" in str(excinfo.value)
    assert "F05" in str(excinfo.value)


# ---------------------------------------------------------------------------
# RF-first ranking: (mismatch_hz, n1, n2, l1, l2, j1, j2) against the
# APPLIED carrier — deterministic and dipole-free (AUTO-001A-R1)
# ---------------------------------------------------------------------------
def test_smallest_mismatch_wins_regardless_of_input_order():
    close = _candidate(1e3)      # true mismatch | |nu| - f_applied | = 1 kHz
    far = _candidate(5e7)        # 50 MHz
    winner_signed = -close["transition_freq_hz"]     # winner's own carrier
    for pool in ([far, close], [close, far]):
        selected = resolve_rf_first_manifold(
            list(pool), _SignedFreqAtom(winner_signed), _APPLIED_HZ)
        assert selected["mismatch_hz"] == pytest.approx(1e3)


def test_equal_mismatch_breaks_tie_on_quantum_numbers_not_dipole():
    """The candidates carry NO dipole: a dipole-participating key would
    raise KeyError here, so this locks the FS-only (dipole-free) ranking."""
    older = _candidate(1e6, n1=40, l1=2, j1=2.5, n2=41, l2=1, j2=1.5)
    newer = _candidate(1e6, n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5)
    selected = resolve_rf_first_manifold(
        [older, newer],
        _SignedFreqAtom(-newer["transition_freq_hz"]), _APPLIED_HZ)
    assert selected["from_level"].n == 26
    assert "dipole_rf" not in selected and "rf_transition" not in selected


def test_resolver_reranks_against_the_applied_carrier_not_stale_mismatch():
    """A stale stored mismatch_hz must not decide the selection."""
    honest = _candidate(1e3, n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5)
    stale = _candidate(1e3, n1=20, l1=0, j1=0.5, n2=21, l2=1, j2=1.5)
    # The stale entry LIES: its true carrier is 50 MHz off the applied one …
    stale["transition_freq_hz"] = _APPLIED_HZ + 5.0e7
    stale["mismatch_hz"] = 1e3
    stale["rf_frequency_error_hz"] = 1e3
    # … but with n1 = 20 it would win any key trusting that stale value.
    atom = _SignedFreqAtom(-honest["transition_freq_hz"])
    selected = resolve_rf_first_manifold([stale, honest], atom, _APPLIED_HZ)
    assert selected["from_level"].n == 26
    assert selected["mismatch_hz"] == pytest.approx(1e3)


def test_selection_equals_the_documented_ranking_key():
    """Regression lock: the resolver's pick must equal
    ``min(candidates, key=rf_first_ranking_key)`` exactly."""
    pool = [
        _candidate(7e7, n1=30, l1=0, j1=0.5, n2=31, l2=1, j2=1.5),
        _candidate(2e7, n1=40, l1=2, j1=2.5, n2=41, l2=1, j2=1.5),
        _candidate(2e7, n1=50, l1=3, j1=2.5, n2=51, l2=2, j2=1.5),
        _candidate(9e6, n1=60, l1=1, j1=1.5, n2=61, l2=0, j2=0.5),
    ]
    expected = min(list(pool), key=rf_first_ranking_key)
    atom = _SignedFreqAtom(-expected["transition_freq_hz"])
    selected = resolve_rf_first_manifold(list(pool), atom, _APPLIED_HZ)
    assert selected["from_level"].label == expected["from_level"].label
    assert selected["to_level"].label == expected["to_level"].label
    assert selected["mismatch_hz"] == pytest.approx(expected["mismatch_hz"])


def test_ranking_key_is_seven_element_and_orders_quantum_numbers():
    low = _candidate(1e6, n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5)
    high = _candidate(1e6, n1=30, l1=2, j1=2.5, n2=31, l2=1, j2=1.5)
    key_low = rf_first_ranking_key(low)
    key_high = rf_first_ranking_key(high)
    assert len(key_low) == 7
    assert key_low == (1e6, 26, 27, 2, 1, 2.5, 1.5)
    assert key_low < key_high                    # mismatch tie → n1 order


def test_resolver_does_not_mutate_the_scan_list_entries():
    candidate = _candidate(1e6)
    selected = resolve_rf_first_manifold(
        [candidate],
        _SignedFreqAtom(-candidate["transition_freq_hz"]), _APPLIED_HZ)
    assert "rf_orientation" not in candidate          # input untouched
    assert selected is not candidate                  # copy, not the same dict
    assert isinstance(selected["rf_orientation"], RFTransitionOrientation)
    # The selection output stays dipole-free: dipoles attach post-selection.
    assert "dipole_rf" not in selected and "rf_transition" not in selected


# ---------------------------------------------------------------------------
# Energy orientation resolved from ARC's SIGNED frequency (RF-DIR-001)
# ---------------------------------------------------------------------------
def test_downward_pair_resolves_downward_orientation_in_ladder_order():
    signed = -1.3264910772876007e11   # production pair: R1(26D5/2) -> R2(27P3/2)
    atom = _SignedFreqAtom(signed)
    selected = resolve_rf_first_manifold(
        [_candidate(carrier_hz=abs(signed))], atom, _APPLIED_HZ)

    orientation = selected["rf_orientation"]
    assert orientation.direction == -1
    assert orientation.is_downward
    assert orientation.signed_frequency_hz == signed
    assert orientation.carrier_hz == abs(signed)
    assert selected["signed_transition_freq_hz"] == signed
    # Ladder order: the SIGNED frequency is queried R1 -> R2 (from -> to).
    assert atom.calls == [(26, 2, 2.5, 27, 1, 1.5)]
    # The selection evidence is recomputed against the APPLIED carrier.
    assert selected["mismatch_hz"] == pytest.approx(
        abs(abs(signed) - _APPLIED_HZ))
    assert selected["rf_frequency_error_hz"] == pytest.approx(
        selected["mismatch_hz"])


def test_upward_pair_resolves_upward_orientation():
    carrier = 5.0e9
    atom = _SignedFreqAtom(+carrier)
    selected = resolve_rf_first_manifold(
        [_candidate(carrier_hz=carrier)], atom, _APPLIED_HZ)
    orientation = selected["rf_orientation"]
    assert orientation.direction == +1
    assert orientation.is_upward
    assert orientation.upper_role == "R2"
    assert orientation.lower_role == "R1"


# ---------------------------------------------------------------------------
# FS-only, domain-complete RF candidate scan (scan_fs_rf_candidates)
# ---------------------------------------------------------------------------
def test_scan_admits_only_fine_structure_eligible_pairs():
    (candidates, checked), _ = _scan(-_APPLIED_HZ)
    assert candidates and checked > 0
    for c in candidates:
        r1, r2 = c["from_level"], c["to_level"]
        assert abs(r1.l - r2.l) == 1          # RF-side |Δl| = 1
        assert abs(r1.j - r2.j) <= 1          # RF-side |Δj| <= 1
        assert abs(r1.l - 1) == 1             # coupling-side |Δl| = 1
        assert abs(r1.j - _EXCITED_J) <= 1    # coupling-side |Δj| <= 1


def test_scan_candidates_carry_the_fs_only_schema():
    (candidates, _), _ = _scan(-_APPLIED_HZ)
    expected_keys = {"from_level", "to_level", "transition_freq_hz",
                     "signed_transition_freq_hz", "mismatch_hz",
                     "rf_frequency_error_hz"}
    for c in candidates:
        assert set(c) == expected_keys
        assert "dipole_rf" not in c and "rf_transition" not in c
        assert c["from_level"].mj is None and c["to_level"].mj is None


def test_scan_enumerates_independent_n1_n2():
    """Domain-complete: n2 is NOT restricted to n1 + 1 (no adjacent-only)."""
    (candidates, _), _ = _scan(-_APPLIED_HZ, n_range=(20, 24))
    assert any(abs(c["from_level"].n - c["to_level"].n) >= 3
               for c in candidates)


def test_scan_carrier_outside_tolerance_is_domain_exclusion():
    """| |nu| - f_applied | > tol excludes the pair — it is not a failure."""
    (candidates, checked), _ = _scan(-2.0e9, applied_hz=1.0e9, tol_hz=1.0e6)
    assert candidates == []
    assert checked > 0


def test_scan_arc_failure_raises_and_is_never_dropped():
    with pytest.raises(PhysicsCalculationError) as excinfo:
        scan_fs_rf_candidates(_ExplodingAtom(), (20, 24), _level(5, 1, 1.5),
                              _APPLIED_HZ, _TOL_HZ)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "F05" in str(excinfo.value)


def test_scan_non_finite_frequency_raises():
    with pytest.raises(PhysicsCalculationError):
        _scan(float("nan"))


def test_scan_validates_its_inputs():
    excited = _level(5, 1, _EXCITED_J)
    with pytest.raises(PhysicsConfigurationError):
        scan_fs_rf_candidates(None, (20, 24), excited, _APPLIED_HZ, _TOL_HZ)
    with pytest.raises(PhysicsConfigurationError):
        scan_fs_rf_candidates(_ConstantScanAtom(-_APPLIED_HZ), (20, 24),
                              None, _APPLIED_HZ, _TOL_HZ)
    with pytest.raises(PhysicsConfigurationError):
        _scan(-_APPLIED_HZ, applied_hz=0.0)
    with pytest.raises(PhysicsConfigurationError):
        _scan(-_APPLIED_HZ, applied_hz=float("nan"))
    with pytest.raises(PhysicsConfigurationError):
        _scan(-_APPLIED_HZ, tol_hz=-1.0)
    with pytest.raises(PhysicsConfigurationError):
        _scan(-_APPLIED_HZ, n_range=(24, 24))
    with pytest.raises(PhysicsConfigurationError):
        _scan(-_APPLIED_HZ, n_range=(30, 20))


def test_scan_coupling_side_j_rule_excludes_high_j1():
    """Excited j = 1/2: an l1 = 2 arm with j1 = 5/2 is coupling-side |Δj| = 2."""
    excited = _level(5, 1, 0.5)
    (candidates, _), _ = _scan(-_APPLIED_HZ, excited=excited)
    assert candidates
    assert not any(c["from_level"].l == 2 and c["from_level"].j == 2.5
                   for c in candidates)


def test_scan_records_carrier_mismatch_and_aliases():
    carrier = _APPLIED_HZ + 0.3e6          # 0.3 MHz <= the 1 MHz tolerance
    (candidates, _), _ = _scan(-carrier)
    assert candidates
    for c in candidates:
        assert c["transition_freq_hz"] == pytest.approx(carrier)
        assert c["signed_transition_freq_hz"] == pytest.approx(-carrier)
        assert c["mismatch_hz"] == pytest.approx(0.3e6)
        assert c["rf_frequency_error_hz"] == pytest.approx(c["mismatch_hz"])


def test_scan_respects_the_l_cap():
    (candidates, _), _ = _scan(-_APPLIED_HZ)
    assert all(c["from_level"].l <= RF_MAX_L_TO_CHECK
               and c["to_level"].l <= RF_MAX_L_TO_CHECK
               for c in candidates)


# ---------------------------------------------------------------------------
# AUTO-001A-R2 — same-n eligibility (FS identity, not n alone)
# ---------------------------------------------------------------------------
def test_scan_evaluates_same_n_pairs_with_fs_identity():
    """Equal n does not exclude a pair: an eligible same-n pair must REACH
    ARC frequency evaluation and become a candidate, and its two ends must
    be fine-structure-distinct (never the same state)."""
    (candidates, _), atom = _scan(-_APPLIED_HZ, n_range=(20, 24))
    assert atom.calls > 0
    same_n_queries = [q for q in atom.queries if q[0] == q[3]]
    assert same_n_queries, (
        "same-n pairs never reached ARC frequency evaluation — the n2==n1 "
        "exclusion must be gone")

    same_n_candidates = [c for c in candidates
                         if c["from_level"].n == c["to_level"].n]
    assert same_n_candidates, (
        "an FS-eligible same-n pair (|dl| = 1 on both arms) must survive "
        "the tolerance filter")
    for c in same_n_candidates:
        r1, r2 = c["from_level"], c["to_level"]
        # FS identity — n alone never decides whether the states differ.
        assert (r1.n, r1.l, r1.j) != (r2.n, r2.l, r2.j)
        assert abs(r1.l - r2.l) == 1
        assert abs(r1.j - r2.j) <= 1


def test_same_n_pair_can_win_a_synthetic_rf_first_search():
    """A same-n pair is not merely scanned — it can WIN the RF-first search
    when its carrier is the best match at the applied frequency."""
    excited = _level(5, 1, _EXCITED_J)
    target = (20, 0, 0.5, 20, 1, 0.5)        # same-n: 20S1/2 -> 20P1/2
    atom = _SelectivePairAtom(
        target_qns=target,
        signed_hz=-(_APPLIED_HZ + 1e3),       # 1 kHz off  -> inside 1 MHz tol
        far_signed_hz=-(_APPLIED_HZ + 5e8),   # 500 MHz off -> domain exclusion
    )
    candidates, checked = scan_fs_rf_candidates(
        atom, (20, 24), excited, _APPLIED_HZ, _TOL_HZ)
    assert checked > 0
    assert candidates, "the eligible same-n pair must reach and pass the scan"

    winner = resolve_rf_first_manifold(
        candidates, _SignedFreqAtom(-(_APPLIED_HZ + 1e3)), _APPLIED_HZ)
    assert (winner["from_level"].n, winner["from_level"].l,
            winner["from_level"].j) == target[:3]
    assert (winner["to_level"].n, winner["to_level"].l,
            winner["to_level"].j) == target[3:]
    assert winner["mismatch_hz"] == pytest.approx(1e3)


def test_scan_keeps_non_adjacent_and_downward_pairs():
    """R2 regression lock: independent n1/n2 enumeration keeps pairs with
    |n2 - n1| >= 3, and the SIGNED ARC frequency (downward edges, nu < 0)
    survives on every candidate."""
    (candidates, _), _ = _scan(-_APPLIED_HZ, n_range=(20, 24))
    assert any(abs(c["from_level"].n - c["to_level"].n) >= 3
               for c in candidates), "non-adjacent pairs disappeared"
    assert any(c["signed_transition_freq_hz"] < 0.0 for c in candidates), (
        "downward (nu < 0) pairs disappeared")
    for c in candidates:
        assert c["signed_transition_freq_hz"] < 0.0
        assert c["transition_freq_hz"] == pytest.approx(
            abs(c["signed_transition_freq_hz"]))
        assert c["mismatch_hz"] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# AUTO-001A-R2 — ranking behaviour: exact ties and a changed applied carrier
# ---------------------------------------------------------------------------
def test_exact_tie_ranking_is_input_order_independent():
    """Every candidate carries the SAME mismatch: the winner is decided by
    the quantum-number tail of the ranking key alone, so reversing or
    shuffling the input pool must not change it (exact-tie contract)."""
    tied = [
        _candidate(1e6, n1=30, l1=0, j1=0.5, n2=31, l2=1, j2=1.5),
        _candidate(1e6, n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5),
        _candidate(1e6, n1=40, l1=3, j1=2.5, n2=39, l2=2, j2=1.5),
    ]
    assert len({c["mismatch_hz"] for c in tied}) == 1   # a real exact tie

    # One fixed stub carrier: if the resolver ever picked a DIFFERENT
    # candidate than the deterministic key, the fresh-ARC consistency check
    # would raise and fail this test.
    top = min(tied, key=rf_first_ranking_key)
    atom_signed = -top["transition_freq_hz"]

    def _winner(pool):
        return resolve_rf_first_manifold(
            list(pool), _SignedFreqAtom(atom_signed), _APPLIED_HZ)

    baseline = _winner(tied)
    reversed_pool = _winner(list(reversed(tied)))
    shuffled_pool = _winner([tied[1], tied[2], tied[0]])
    signatures = {
        (w["from_level"].n, w["from_level"].l, w["from_level"].j,
         w["to_level"].n, w["to_level"].l, w["to_level"].j)
        for w in (baseline, reversed_pool, shuffled_pool)
    }
    assert len(signatures) == 1, (
        f"exact-tie winner depends on input order: {signatures}")
    # Deterministic expectation: the lowest n1 wins an exact mismatch tie.
    assert baseline["from_level"].n == 26


def test_changing_applied_carrier_can_change_the_winner():
    """The ranking reference is the APPLIED carrier: the same two eligible
    pairs yield different winners at different carriers — no pair is
    privileged by the scan itself."""
    low = _candidate(carrier_hz=10.001e9,
                     n1=26, l1=2, j1=2.5, n2=27, l2=1, j2=1.5)
    high = _candidate(carrier_hz=20.001e9,
                      n1=40, l1=2, j1=2.5, n2=41, l2=1, j2=1.5)
    pool = [low, high]

    winner_at_10 = resolve_rf_first_manifold(
        list(pool), _SignedFreqAtom(-low["transition_freq_hz"]), 10.0e9)
    assert winner_at_10["from_level"].n == 26
    assert winner_at_10["mismatch_hz"] == pytest.approx(1e6)

    winner_at_20 = resolve_rf_first_manifold(
        list(pool), _SignedFreqAtom(-high["transition_freq_hz"]), 20.0e9)
    assert winner_at_20["from_level"].n == 40
    assert winner_at_20["mismatch_hz"] == pytest.approx(1e6)