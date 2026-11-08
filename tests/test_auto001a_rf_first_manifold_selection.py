"""AUTO-001A-R2 — LIVE ARC evidence for the RF-first manifold selection.

At the ACTUAL configured RF carrier this file proves, with real ARC data:

1. WHICH configured RF field supplies the selection carrier —
   ``RF_PARAMS`` is the documented alias of ``LO_RF_PARAMS`` (the LO field);
   the heterodyne signal field lives in its own ``SIGNAL_RF_PARAMS``
   dictionary and is NOT the ranking reference.
2. An INDEPENDENT enumeration of the eligible fine-structure pairs over the
   ACTIVE bounds — captured from the production call itself, never
   hard-coded — including same-n pairs, with signed frequencies obtained
   directly from ARC.  This oracle does NOT use the production scanner
   (``scan_fs_rf_candidates``) or the production resolver
   (``resolve_rf_first_manifold``) for its own ordering.
3. The production-selected winner equals the oracle's rank-1 candidate and
   the candidate SETS agree pair-for-pair with identical mismatches; the
   top five candidates are reported (see the printed evidence).
4. The FS-only result boundary at the configured carrier (both Rydberg mJ
   fields None, no legacy dipole matrix) and full preservation of the
   LO/signal configuration dictionaries and frequencies.

No expected pair is hard-coded anywhere in this file: the winner comes
from the oracle alone.  The expensive production scan and the oracle
enumeration each run ONCE (module-scoped fixtures) and are reused by every
test below.
"""

import copy
import math

import pytest

from rydberg_aoa.arc import LaserParams, RFParams
from rydberg_aoa.arc.ground_states import find_excited_state
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import atomic_resolution as ar
from rydberg_aoa.config import laser_rf_config as lrf
from rydberg_aoa.config import parameters as params
from rydberg_aoa.config.parameters import LEVEL_SELECTION_TOLERANCES
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.level_finder import SingleStageLevelOptimizer
from rydberg_aoa.level_finder.arc_level_finder import (
    ARCOptimizer,
    RF_MAX_L_TO_CHECK,
)
from tests.atomic_init_support import (
    is_documented_qn001a_coupling_block,
    restore_published_atomic_state,
    snapshot_published_atomic_state,
)

pytestmark = pytest.mark.slow


def _config_snapshot():
    """Every RF/laser dictionary and frequency the selection must preserve."""
    return {
        "RF_PARAMS": copy.deepcopy(lrf.RF_PARAMS),
        "LO_RF_PARAMS": copy.deepcopy(lrf.LO_RF_PARAMS),
        "SIGNAL_RF_PARAMS": copy.deepcopy(lrf.SIGNAL_RF_PARAMS),
        "LASER_PARAMS": copy.deepcopy(lrf.LASER_PARAMS),
        "RF_LO_FREQUENCY": params.RF_LO_FREQUENCY,
        "RF_SIGNAL_FREQUENCY": params.RF_SIGNAL_FREQUENCY,
    }


@pytest.fixture(scope="module")
def production_run():
    """Run the production entry path ONCE, capturing the ACTIVE bounds.

    ``ARCOptimizer.optimize_levels`` is wrapped only to RECORD the
    ``narrowed_ranges`` the real call carries — the oracle must scan exactly
    the domain production used, with no duplicated hard-coded bounds — and
    the original method then executes untouched.
    """
    captured = {}
    original = ARCOptimizer.optimize_levels

    def _spy(self, narrowed_ranges, *args, **kwargs):
        captured["rf_n_range"] = (int(narrowed_ranges.rf_n_range[0]),
                                  int(narrowed_ranges.rf_n_range[1]))
        return original(self, narrowed_ranges, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(ARCOptimizer, "optimize_levels", _spy)
    try:
        before = _config_snapshot()
        result = SingleStageLevelOptimizer(arc_atom=acfg.ATOM).optimize(
            LaserParams(**lrf.LASER_PARAMS["probe"]),
            LaserParams(**lrf.LASER_PARAMS["coupling"]),
            RFParams(**lrf.RF_PARAMS),
        )
        after = _config_snapshot()
    finally:
        patch.undo()

    assert captured.get("rf_n_range"), (
        "the production entry path never reached the FS scan")
    assert result and "levels" in result, (
        "the configured RF carrier must yield a selected manifold")
    return {
        "result": result,
        "rf_n_range": captured["rf_n_range"],
        "before": before,
        "after": after,
    }


def _independent_fs_oracle(atom, n_range, excited, carrier_hz, tolerance_hz,
                           max_l):
    """Independent eligible-pair enumeration + ranking (the ORACLE).

    Implements the documented eligibility rules from the specification —
    fine-structure electric-dipole rules on both arms — with its OWN loops
    and its OWN ARC queries; it never imports or calls the production
    scanner or resolver.  Results are ranked by
    (|mismatch|, n1, n2, l1, l2, j1, j2).
    """
    n_lo, n_hi = n_range
    exc_l, exc_j = int(excited.l), float(excited.j)
    found = []
    for n1 in range(n_lo, n_hi):                 # start-inclusive/stop-exclusive
        for l1 in range(min(n1, max_l + 1)):
            if abs(l1 - exc_l) != 1:             # coupling-side |Δl| = 1
                continue
            for j1 in ([l1 - 0.5, l1 + 0.5] if l1 > 0 else [0.5]):
                if abs(j1 - exc_j) > 1:          # coupling-side |Δj| <= 1
                    continue
                for n2 in range(n_lo, n_hi):     # independent — same-n allowed
                    for l2 in range(min(n2, max_l + 1)):
                        if abs(l1 - l2) != 1:    # RF-side |Δl| = 1
                            continue
                        for j2 in ([l2 - 0.5, l2 + 0.5] if l2 > 0 else [0.5]):
                            if abs(j1 - j2) > 1:  # RF-side |Δj| <= 1
                                continue
                            signed = float(atom.getTransitionFrequency(
                                n1, l1, j1, n2, l2, j2))
                            if not math.isfinite(signed):
                                raise AssertionError(
                                    "ARC returned a non-finite frequency for "
                                    f"({n1},{l1},{j1})->({n2},{l2},{j2}): "
                                    f"{signed!r}")
                            carrier = abs(signed)
                            mismatch = abs(carrier - carrier_hz)
                            if mismatch <= tolerance_hz:
                                found.append({
                                    "n1": n1, "l1": l1, "j1": j1,
                                    "n2": n2, "l2": l2, "j2": j2,
                                    "signed_hz": signed,
                                    "carrier_hz": carrier,
                                    "mismatch_hz": mismatch,
                                })
    found.sort(key=lambda c: (c["mismatch_hz"], c["n1"], c["n2"],
                              c["l1"], c["l2"], c["j1"], c["j2"]))
    return found


@pytest.fixture(scope="module")
def oracle(production_run):
    """One independent ARC enumeration over the ACTIVE production bounds."""
    excited = find_excited_state(
        acfg.ATOM, lrf.LASER_PARAMS["probe"]["wavelength_nm"])
    assert excited is not None, "the configured probe has no excited state"
    carrier_hz = float(lrf.RF_PARAMS["frequency_hz"])
    tolerance_hz = float(LEVEL_SELECTION_TOLERANCES["frequency_tolerance_hz"])
    candidates = _independent_fs_oracle(
        acfg.ATOM, production_run["rf_n_range"], excited,
        carrier_hz, tolerance_hz, RF_MAX_L_TO_CHECK)
    return {
        "excited": excited,
        "carrier_hz": carrier_hz,
        "tolerance_hz": tolerance_hz,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# 1. The selection carrier: which configured RF field supplies it
# ---------------------------------------------------------------------------
def test_selection_carrier_is_the_lo_rf_field_not_the_signal(
        production_run, oracle):
    """``RF_PARAMS`` is the LO alias (documented in laser_rf_config); the
    signal RF dictionary and both source frequencies stay independent, and
    the production metrics name exactly the LO carrier as applied."""
    assert lrf.RF_PARAMS == dict(lrf.LO_RF_PARAMS)
    assert lrf.RF_PARAMS is not lrf.LO_RF_PARAMS     # a copy, not the object
    assert lrf.RF_PARAMS is not lrf.SIGNAL_RF_PARAMS
    assert lrf.RF_PARAMS["frequency_hz"] == params.RF_LO_FREQUENCY
    assert lrf.SIGNAL_RF_PARAMS["frequency_hz"] == params.RF_SIGNAL_FREQUENCY
    assert lrf.SIGNAL_RF_PARAMS["frequency_hz"] != lrf.RF_PARAMS["frequency_hz"]

    metrics = production_run["result"]["optimization_metrics"]
    assert metrics["rf_applied_carrier_hz"] == lrf.RF_PARAMS["frequency_hz"]
    assert oracle["carrier_hz"] == lrf.RF_PARAMS["frequency_hz"]

    print("\nAUTO-001A-R2 — selection-carrier report")
    print(f"  selection carrier (RF_PARAMS = LO alias): "
          f"{lrf.RF_PARAMS['frequency_hz']:.3f} Hz")
    print(f"  signal RF field (SIGNAL_RF_PARAMS):       "
          f"{lrf.SIGNAL_RF_PARAMS['frequency_hz']:.3f} Hz")
    print("  IF (signal - LO):                         "
          f"{(lrf.SIGNAL_RF_PARAMS['frequency_hz'] - lrf.RF_PARAMS['frequency_hz']) / 1e6:.4f} MHz")
    print(f"  applied carrier echoed by production:     "
          f"{metrics['rf_applied_carrier_hz']:.3f} Hz")


# ---------------------------------------------------------------------------
# 2. Active bounds, endpoint semantics, l cap and frequency tolerance
# ---------------------------------------------------------------------------
def test_oracle_domain_and_endpoint_report(production_run, oracle):
    n_lo, n_hi = production_run["rf_n_range"]
    assert n_lo < n_hi
    candidates = oracle["candidates"]
    assert candidates, "the independent ARC enumeration found no pairs"
    excited = oracle["excited"]
    for c in candidates:
        # half-open [n_lo, n_hi) endpoints on BOTH indices; same-n allowed
        assert n_lo <= c["n1"] < n_hi and n_lo <= c["n2"] < n_hi
        assert c["mismatch_hz"] <= oracle["tolerance_hz"]
        assert abs(c["l1"] - c["l2"]) == 1
        assert abs(c["j1"] - c["j2"]) <= 1
        assert abs(c["l1"] - int(excited.l)) == 1
        assert abs(c["j1"] - float(excited.j)) <= 1
        # FS identity — the two ends are never the same state
        assert (c["n1"], c["l1"], c["j1"]) != (c["n2"], c["l2"], c["j2"])

    print("\nAUTO-001A-R2 — search-domain report")
    print(f"  active RF n-range:                [{n_lo}, {n_hi}) "
          "(start-inclusive, stop-exclusive)")
    print(f"  l cap (RF_MAX_L_TO_CHECK):        {RF_MAX_L_TO_CHECK} (S, P, D, F)")
    print(f"  frequency tolerance:              "
          f"+/-{oracle['tolerance_hz'] / 1e6:.0f} MHz")
    print(f"  excited state (configured probe): {excited.label}")
    print(f"  eligible pairs (real ARC):        {len(candidates)}")


# ---------------------------------------------------------------------------
# 3. Top five eligible candidates (reported evidence)
# ---------------------------------------------------------------------------
def test_top_five_candidates_report(oracle):
    top5 = oracle["candidates"][:5]
    assert len(top5) == 5
    mismatches = [c["mismatch_hz"] for c in top5]
    assert mismatches == sorted(mismatches)
    print("\nAUTO-001A-R2 — top five eligible FS candidates "
          "at the configured carrier")
    print("  rank  R1 (n,l,j)              R2 (n,l,j)              "
          "signed [Hz]           carrier [Hz]          "
          "|mismatch| [Hz]     direction")
    for rank, c in enumerate(top5, start=1):
        direction = "+1 (up)" if c["signed_hz"] > 0 else "-1 (down)"
        print(f"  {rank:>4}  ({c['n1']},{c['l1']},{c['j1']})".ljust(34)
              + f"({c['n2']},{c['l2']},{c['j2']})".ljust(26)
              + f"{c['signed_hz']:>19.3f}  {c['carrier_hz']:>19.3f}  "
                f"{c['mismatch_hz']:>17.3f}   {direction}")


# ---------------------------------------------------------------------------
# 4. Production-selected winner vs the independent oracle (rank-1 + sets)
# ---------------------------------------------------------------------------
def test_production_winner_equals_the_oracle_rank1(production_run, oracle):
    """The production-selected winner IS the oracle's rank-1 candidate.

    Two independent computations over the SAME active bounds and the SAME
    applied carrier must agree:

    * the winner's quantum numbers, carrier and mismatch match the oracle's
      rank-1 entry, and
    * the candidate SETS agree pair-for-pair with identical mismatches —
      no pair is missing from either side (the production scanner and the
      oracle enumerate the domain independently; neither imports the other).

    No pair is hard-coded: the expected winner comes from the oracle alone.
    """
    result = production_run["result"]
    selected = result["selected_rf_pair"]
    r1, r2 = selected["from_level"], selected["to_level"]
    best = oracle["candidates"][0]

    # -- winner equality ---------------------------------------------------
    assert (r1.n, r1.l, r1.j) == (best["n1"], best["l1"], best["j1"])
    assert (r2.n, r2.l, r2.j) == (best["n2"], best["l2"], best["j2"])
    assert selected["mismatch_hz"] == pytest.approx(
        best["mismatch_hz"], abs=1e-6)
    assert selected["transition_freq_hz"] == pytest.approx(
        best["carrier_hz"], abs=1e-6)
    assert selected["signed_transition_freq_hz"] == pytest.approx(
        best["signed_hz"], abs=1e-6)
    # the production list itself is ranked so its head is the same winner
    head = result["rf_transitions_found"][0]
    assert (head["from_level"].n, head["from_level"].l, head["from_level"].j,
            head["to_level"].n, head["to_level"].l, head["to_level"].j) == (
        best["n1"], best["l1"], best["j1"],
        best["n2"], best["l2"], best["j2"])

    # -- candidate SETS agree pair-for-pair ---------------------------------
    production_pairs = {
        (p["from_level"].n, p["from_level"].l, p["from_level"].j,
         p["to_level"].n, p["to_level"].l, p["to_level"].j): p["mismatch_hz"]
        for p in result["rf_transitions_found"]
    }
    oracle_pairs = {
        (c["n1"], c["l1"], c["j1"], c["n2"], c["l2"], c["j2"]):
            c["mismatch_hz"]
        for c in oracle["candidates"]
    }
    production_only = sorted(set(production_pairs) - set(oracle_pairs))
    oracle_only = sorted(set(oracle_pairs) - set(production_pairs))
    assert not production_only and not oracle_only, (
        "production and independent-oracle candidate sets differ; "
        f"production-only={production_only[:5]}, oracle-only={oracle_only[:5]}")
    for key, mismatch in production_pairs.items():
        assert mismatch == pytest.approx(oracle_pairs[key], abs=1e-6), (
            f"mismatch differs for pair {key}: production={mismatch!r} "
            f"oracle={oracle_pairs[key]!r}")

    print("\nAUTO-001A-R2 — winner vs oracle report")
    print(f"  selected winner:  ({best['n1']},{best['l1']},{best['j1']}) -> "
          f"({best['n2']},{best['l2']},{best['j2']}) "
          f"(mismatch {best['mismatch_hz'] / 1e6:.6f} MHz)")
    print(f"  candidate sets:   production={len(production_pairs)}, "
          f"oracle={len(oracle_pairs)} — pair-for-pair identical")


# ---------------------------------------------------------------------------
# 5. FS-only result boundary + full configuration preservation
# ---------------------------------------------------------------------------
def test_fs_only_result_boundary_and_configuration_preservation(production_run):
    """The configured-carrier result is FS-only and mutates NO configuration.

    * both selected Rydberg ``mj`` fields stay ``None`` (and every scan
      candidate's levels as well);
    * no legacy ``dipole_matrix``/``transitions`` result key, no metrics
      ``dipole_rf`` claim — nothing that could only be produced by the
      removed maximum-over-mJ scan;
    * the LO/signal RF dictionaries, laser dictionaries and both configured
      RF frequencies are bit-identical to their pre-run snapshot.
    """
    # -- full preservation of LO/signal/laser configuration ----------------
    assert production_run["before"] == production_run["after"], (
        "production FS selection mutated the configured RF/laser parameters")

    result = production_run["result"]

    # -- FS-only result boundary at the configured carrier -----------------
    assert "dipole_matrix" not in result, (
        "the FS-only selection must not fabricate a legacy dipole matrix")
    assert "transitions" not in result, (
        "the FS-only selection must not fabricate transition objects")
    metrics = result["optimization_metrics"]
    assert "dipole_rf" not in metrics, (
        "optimization metrics must carry no dipole claim (QN-001B/C)")
    selected = result["selected_rf_pair"]
    assert "dipole_rf" not in selected

    r1 = result["levels"]["rydberg1"]
    r2 = result["levels"]["rydberg2"]
    assert r1.mj is None and r2.mj is None, (
        "the EXPERIMENTAL ladder must declare no Rydberg mJ")
    assert selected["from_level"].mj is None
    assert selected["to_level"].mj is None
    assert all("dipole_rf" not in p for p in result["rf_transitions_found"])
    assert all(p["from_level"].mj is None and p["to_level"].mj is None
               for p in result["rf_transitions_found"])

    print("\nAUTO-001A-R2 — FS-only boundary + preservation report")
    print(f"  selected R1 mJ={r1.mj!r}, R2 mJ={r2.mj!r} "
          f"(levels {r1.label} / {r2.label})")
    print("  configuration snapshot: LO/signal/laser dictionaries and "
          "frequencies unchanged")


# ---------------------------------------------------------------------------
# 6. Production wiring: banned legacy dipole helpers + explicit missing-mJ stop
# ---------------------------------------------------------------------------
def test_production_wiring_bans_legacy_dipole_helpers_and_stops_at_missing_mj(
        monkeypatch):
    """AUTO-001A-R2 production wiring (fail-loudly evidence).

    With the legacy ``arc/dipole_matrix`` maximum-over-mJ scanner and the
    generic ``arc/dipole`` element sweep patched to RAISE if ever called:

    * the production FS selection inside ``initialize_atomic_physics()``
      must still return the selected manifolds — proven by the level-cache
      entry written before the boundary (four states, both Rydberg mJ
      fields ``None``), with no dipole cache entry and no ``dipole_rf``
      optimization-results claim;
    * full initialization must still stop at the EXPLICIT missing-mJ
      coupling boundary — the strict QN-001A predicate
      (``PhysicsConfigurationError`` naming HFS->FS, "no mJ is inferred"
      and "does not scan mJ") — and publish nothing (ARCH-001
      transactional: the initialized flag stays False).

    ARCH-001TX: the production seam now ROLLS BACK its pre-boundary rewrite
    (coupling wavelength, ``SI_Constants.LAMBDA_P``, the pipeline caches) on
    failure, so the FS-selection evidence is captured with a spy at the
    FS-return boundary instead of from post-failure cache state, and the
    post-failure assertions pin the rollback contract itself.  The
    ``finally`` block is kept as belt-and-suspenders isolation for this
    test regardless of collection order.
    """
    import rydberg_aoa.arc.dipole as dipole_mod
    import rydberg_aoa.arc.dipole_matrix as dipole_matrix_mod

    hits = {"legacy_matrix": 0, "generic_dme": 0}

    def _ban(name):
        def _boom(*args, **kwargs):
            hits[name] += 1
            raise AssertionError(
                "AUTO-001A-R2: the production FS-only selection path called "
                f"the banned legacy helper {name!r} — no magnetic-channel "
                "or maximum-over-mJ scan may run during FS selection "
                "(QN-001B/C)")
        return _boom

    monkeypatch.setattr(dipole_matrix_mod, "calculate_dipole_matrix",
                        _ban("legacy_matrix"))
    monkeypatch.setattr(dipole_mod, "calculate_dipole_matrix_element",
                        _ban("generic_dme"))

    # ARCH-001TX: capture the FS-selection result AT THE RETURN BOUNDARY of
    # the production level step.  The transactional rollback afterwards
    # removes the cache evidence by design, so the selected manifolds are
    # asserted from this spy instead of from post-failure cache state.
    real_initialize_levels = acfg._initialize_atomic_levels
    fs_return = {}

    def _spy_initialize_levels(*args, **kwargs):
        result = real_initialize_levels(*args, **kwargs)
        fs_return["levels"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(acfg, "_initialize_atomic_levels",
                        _spy_initialize_levels)

    assert ar.ATOMIC_RESOLUTION_MODE is ar.AtomicResolutionMode.EXPERIMENTAL, (
        "this wiring test pins the EXPERIMENTAL production path; the "
        "resolution switch is contaminated by an earlier test")

    published_before = snapshot_published_atomic_state()
    laser_before = copy.deepcopy(lrf.LASER_PARAMS)
    lambda_p_before = params.SI_Constants.LAMBDA_P
    caches_before = {
        name: dict(getattr(acfg, name))
        for name in ("_ATOMIC_LEVELS_CACHE", "_DETUNING_CACHE",
                     "_DIPOLE_MATRIX_CACHE", "_OPTIMIZATION_RESULTS_CACHE")
    }
    try:
        # deterministic entry: the seam must RUN (no idempotent short-circuit)
        # and its evidence must be attributable to THIS call alone.
        for name in caches_before:
            getattr(acfg, name).clear()
        acfg._ATOMIC_PHYSICS_INITIALIZED = False

        with pytest.raises(PhysicsConfigurationError) as excinfo:
            acfg.initialize_atomic_physics()

        # (b) the EXPLICIT missing-mJ boundary — strict QN-001A predicate
        assert is_documented_qn001a_coupling_block(excinfo.value), (
            "initialization did not stop at the documented HFS->FS "
            f"missing-mJ boundary: {excinfo.value}")
        assert "F05" in str(excinfo.value)

        # the banned legacy helpers were NEVER reached
        assert hits == {"legacy_matrix": 0, "generic_dme": 0}, (
            f"legacy dipole helpers were called: {hits}")

        # (a) FS selection inside the seam DID return the selected manifolds —
        # captured at the FS-return boundary (ARCH-001TX: the rollback below
        # removes the cache evidence by design).
        assert "levels" in fs_return, (
            "the production FS selection never returned to the spy")
        entry = fs_return["levels"]
        for state in ("ground", "excited", "rydberg1", "rydberg2"):
            assert state in entry, f"selected ladder is missing {state!r}"
        r1, r2 = entry["rydberg1"], entry["rydberg2"]
        assert r1["mj"] is None and r2["mj"] is None, (
            "FS selection must leave both Rydberg mJ fields unset")
        assert (r1["n"], r1["l"], r1["j"]) != (r2["n"], r2["l"], r2["j"]), (
            "the selected RF manifold must contain two distinct states")

        # ARCH-001TX: the failed attempt rolled back EVERY cache it wrote —
        # no half-populated entry may survive the transaction.
        assert len(acfg._ATOMIC_LEVELS_CACHE) == 0, (
            "the level cache survived a failed initialization (rollback)")
        assert len(acfg._DETUNING_CACHE) == 0, (
            "the detuning cache survived a failed initialization (rollback)")
        assert len(acfg._OPTIMIZATION_RESULTS_CACHE) == 0, (
            "the optimization-results cache survived a failed "
            "initialization (rollback)")
        assert len(acfg._DIPOLE_MATRIX_CACHE) == 0, (
            "EXPERIMENTAL initialization must not leave a legacy dipole "
            "cache entry")
        # ...and the pre-boundary configuration rewrite was rolled back too.
        assert lrf.LASER_PARAMS == laser_before, (
            "LASER_PARAMS survived a failed initialization (rollback)")
        assert params.SI_Constants.LAMBDA_P == lambda_p_before, (
            "SI_Constants.LAMBDA_P survived a failed initialization "
            "(rollback)")

        # ARCH-001 transactional: nothing was published before the boundary
        assert acfg.is_atomic_physics_initialized() is False
    finally:
        lrf.LASER_PARAMS.clear()
        lrf.LASER_PARAMS.update(laser_before)
        params.SI_Constants.LAMBDA_P = lambda_p_before
        for name, snapshot in caches_before.items():
            getattr(acfg, name).clear()
            getattr(acfg, name).update(snapshot)
        restore_published_atomic_state(published_before)

    print("\nAUTO-001A-R2 — production wiring report")
    print("  legacy dipole helpers reached: 0 (banned and never called)")
    print("  selected ladder from cache:    "
          f"R1(n={r1['n']},l={r1['l']},j={r1['j']},mj={r1['mj']!r}) -> "
          f"R2(n={r2['n']},l={r2['l']},j={r2['j']},mj={r2['mj']!r})")
    print("  initialization stopped at:     documented QN-001A HFS->FS "
          "missing-mJ boundary (F05) — nothing published")