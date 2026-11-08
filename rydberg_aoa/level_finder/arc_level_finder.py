import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from ..arc.interface import ARCInterface
from ..config.parameters import LEVEL_SELECTION_TOLERANCES
from ..exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)
from ..arc.rf_orientation import resolve_rf_orientation
from ..arc._scan_validation import real_scalar, integer_scalar, arc_real_scalar
from .types import NarrowedRanges, OptimizationCriteria, AtomicLevel
from rydberg_aoa.config import constants as const
from ..config.laser_rf_config import LASER_PARAMS
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUTO-001A-R1 — RF-first manifold selection (EXPERIMENTAL path only).
#
# The selection stage is FINE-STRUCTURE only: the electric-dipole rules
# |Δl| = 1, |Δj| ≤ 1 on the RF pair plus the coupling-side feasibility rules
# against the excited state.  It never touches mJ/q/polarisation and never
# computes a dipole matrix element — the magnetic channel belongs to the
# channel-dependent path (QN-001B/C), never to level selection.
# ---------------------------------------------------------------------------

#: Highest l enumerated by the RF scan (l = 0..3 → S, P, D, F) — the
#: historical performance cap, unchanged.
RF_MAX_L_TO_CHECK = 3

_L_LETTERS = 'SPDFGHIJKLMNOP'


def _arc_level_energy(atom, level):
    """Read finite ARC FS energy [eV] from the supplied atom only."""
    try:
        return arc_real_scalar(atom.getEnergy(level.n, level.l, level.j),
                               f"ARC energy {level.label} [eV]")
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f"LAB-001: ARC energy failed for {level.label}. F05.") from exc


def rf_first_ranking_key(candidate: Dict[str, Any]) -> Tuple[float, int, int, int, int, float, float]:
    """The single deterministic RF-first ranking key (AUTO-001A-R1).

    ``(mismatch_hz, n1, n2, l1, l2, j1, j2)`` — the ACTUAL carrier closest
    to the applied RF field wins first; quantum numbers only order exact
    mismatches.  No dipole or any other physical preference participates.
    """
    r1, r2 = candidate['from_level'], candidate['to_level']
    return (
        float(candidate['mismatch_hz']),
        int(r1.n), int(r2.n),
        int(r1.l), int(r2.l),
        float(r1.j), float(r2.j),
    )


def resolve_rf_first_manifold(candidates, atom, applied_carrier_hz) -> Dict[str, Any]:
    """AUTO-001A-R1 — RF-first manifold resolver (EXPERIMENTAL path only).

    Selects the ``(R1, R2)`` manifold from the scanned FS candidates with the
    deterministic RF-first ranking

        (mismatch_hz, n1, n2, l1, l2, j1, j2),  mismatch = | |nu| - f_applied |

    where every candidate's ACTUAL ARC carrier ``nu`` is re-ranked against
    ``applied_carrier_hz``: the physically closest carrier to the applied RF
    wins first, and the quantum numbers only break exact mismatches.  The
    winner's ENERGY orientation is then resolved from ARC's **signed**
    ladder-order frequency through
    :func:`rydberg_aoa.arc.rf_orientation.resolve_rf_orientation` (RF-DIR-001),
    so the orientation is fixed at selection time instead of being
    re-inferred downstream.

    Scope: EXPERIMENTAL mode only (PRESCRIBED mode never reaches
    :class:`ARCOptimizer`).  This stage never computes dipoles and never
    scans magnetic channels; candidate INPUTS are not mutated (ranked
    copies are returned).

    Parameters
    ----------
    candidates : list[dict]
        Scanned FS pairs carrying ``from_level``/``to_level``/
        ``transition_freq_hz`` (|nu| [Hz]); input order is irrelevant.
    atom : ARC atom
        Provides the signed ladder-order frequency ``nu = (E_R2 - E_R1)/h``
        of the SELECTED pair.
    applied_carrier_hz : float
        The RF carrier the ranking references (``rf_params.frequency_hz``).

    Raises
    ------
    PhysicsConfigurationError
        No ARC atom, or an invalid applied carrier (F05).
    PhysicsCalculationError
        Empty candidate list, ARC failure, a zero/non-finite signed
        frequency, or a stale scan-time carrier that disagrees with a fresh
        ARC query (F05: direction never inferred, no inconsistent pick).
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "AUTO-001A: RF-first manifold resolution requires an ARC atom. "
            "F05: no manifold is selected without a physics source."
        )
    if not candidates:
        raise PhysicsCalculationError(
            "AUTO-001A: no RF manifold candidate reached the resolver "
            "(empty scan result). F05: no fabricated or substituted manifold."
        )
    applied_carrier_hz = real_scalar(applied_carrier_hz, "applied RF carrier [Hz]")
    if applied_carrier_hz <= 0.0:
        raise PhysicsConfigurationError(
            "AUTO-001A-R1: the applied RF carrier must be finite and positive "
            f"(got {applied_carrier_hz!r}). F05: RF-first ranking never "
            "defaults or guesses the carrier."
        )

    # RF-first ranking against the applied carrier — each normalized copy
    # carries the canonical mismatch, then the single module key orders them.
    ranked = sorted(
        ({**c, 'mismatch_hz': abs(arc_real_scalar(
            c['transition_freq_hz'], "scanned RF carrier [Hz]", positive=True) - applied_carrier_hz)}
         for c in candidates),
        key=rf_first_ranking_key)
    best = ranked[0]
    r1, r2 = best['from_level'], best['to_level']

    try:
        signed_hz = arc_real_scalar(atom.getTransitionFrequency(
            r1.n, r1.l, r1.j, r2.n, r2.l, r2.j),
            f"ARC RF frequency {r1.label} -> {r2.label} [Hz]", nonzero=True)
        orientation = resolve_rf_orientation(signed_hz)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            "AUTO-001A: energy orientation of the selected RF manifold "
            f"{r1.label} <-> {r2.label} could not be resolved from ARC's "
            f"signed frequency: {type(exc).__name__}: {exc}. "
            "F05: no inferred RF direction."
        ) from exc

    # Consistency: a fresh ARC query for the winner must reproduce the
    # carrier recorded at scan time — otherwise the candidate list is stale
    # and RF-first selection from it would be untrustworthy (F05).
    stored_carrier_hz = float(best['transition_freq_hz'])
    if not math.isclose(abs(signed_hz), stored_carrier_hz,
                        rel_tol=1e-9, abs_tol=1e-6):
        raise PhysicsCalculationError(
            "AUTO-001A-R1: inconsistent candidate carrier for "
            f"{r1.label} <-> {r2.label}: scanned |nu| = {stored_carrier_hz!r} Hz "
            f"vs fresh ARC |nu| = {abs(signed_hz)!r} Hz. "
            "F05: no selection from inconsistent data."
        )

    best['rf_orientation'] = orientation
    best['signed_transition_freq_hz'] = orientation.signed_frequency_hz
    best['mismatch_hz'] = abs(orientation.carrier_hz - applied_carrier_hz)
    return best


def scan_fs_rf_candidates(atom, n_range, excited_level, applied_carrier_hz,
                          frequency_tolerance_hz, max_l=RF_MAX_L_TO_CHECK):
    """AUTO-001A-R1 — domain-complete FS-only RF candidate scan (EXPERIMENTAL).

    ``n1`` and ``n2`` are enumerated INDEPENDENTLY inside ``n_range``
    (stop-exclusive; no adjacent-only restriction and NO same-n exclusion —
    equal principal quantum numbers do NOT imply equal fine-structure
    identity, so a same-n pair stays eligible when it obeys the rules below;
    the |Δl| = 1 rule alone guarantees the two ends are different states).
    A pair is structurally eligible iff it obeys the fine-structure
    electric-dipole rules of the RF-first pipeline:

        RF pair       : |l1 - l2| = 1  and  |j1 - j2| <= 1
        coupling side : |l1 - l_exc| = 1  and  |j1 - j_exc| <= 1

    No mJ/q/polarisation scan, no dipole matrix element and no dipole
    threshold participates here (QN-001B/C: the magnetic channel belongs to
    the DECLARED channel-dependent path, never to level selection).

    A structurally eligible pair becomes a candidate iff
    ``| |nu| - f_applied | <= frequency_tolerance_hz``; everything outside
    the tolerance is a DOMAIN EXCLUSION, while an ARC failure is raised —
    F05: a scan failure is never silently dropped.

    Returns
    -------
    (candidates, combinations_checked)
        Each candidate carries ``from_level``/``to_level`` (mj = None),
        ``transition_freq_hz`` = |nu| [Hz], ``signed_transition_freq_hz``
        = nu [Hz] and ``mismatch_hz`` = ``rf_frequency_error_hz``
        = | |nu| - f_applied | [Hz].

    Raises
    ------
    PhysicsConfigurationError
        Missing atom/excited level, invalid n-range, carrier or tolerance.
    PhysicsCalculationError
        ARC raised, or returned a non-finite frequency (F05).
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "AUTO-001A-R1: FS-only RF scan requires an ARC atom. "
            "F05: no candidates without a physics source.")
    if excited_level is None:
        raise PhysicsConfigurationError(
            "AUTO-001A-R1: FS-only RF scan requires the excited state for "
            "the coupling-side selection rules. F05: no assumed state.")
    applied_carrier_hz = real_scalar(applied_carrier_hz, "applied RF carrier [Hz]")
    if applied_carrier_hz <= 0.0:
        raise PhysicsConfigurationError(
            "AUTO-001A-R1: applied RF carrier must be finite and positive "
            f"(got {applied_carrier_hz!r}). F05: no defaulted carrier.")
    frequency_tolerance_hz = real_scalar(frequency_tolerance_hz, "frequency tolerance [Hz]")
    if frequency_tolerance_hz < 0.0:
        raise PhysicsConfigurationError(
            "AUTO-001A-R1: frequency tolerance must be finite and >= 0 "
            f"(got {frequency_tolerance_hz!r}). F05: no defaulted tolerance.")
    if not isinstance(n_range, (tuple, list)) or len(n_range) != 2:
        raise PhysicsConfigurationError(
            f"LAB-001: n_range must be two integer endpoints; got {n_range!r}. F05.")
    n_lo = integer_scalar(n_range[0], "n_range lower bound", minimum=1)
    n_hi = integer_scalar(n_range[1], "n_range exclusive upper bound", minimum=1)
    max_l = integer_scalar(max_l, "max_l")
    if max_l >= len(_L_LETTERS):
        raise PhysicsConfigurationError("LAB-001: max_l exceeds supported orbital labels. F05.")
    if n_hi <= n_lo:
        raise PhysicsConfigurationError(
            f"AUTO-001A-R1: invalid RF n-range {n_range!r} (need lo < hi). "
            "F05: no implicit range substitution.")

    candidates: List[Dict[str, Any]] = []
    combinations_checked = 0
    excited_l = integer_scalar(excited_level.l, "excited l")
    excited_j = real_scalar(excited_level.j, "excited j")
    if excited_j not in ([excited_l - 0.5, excited_l + 0.5] if excited_l else [0.5]):
        raise PhysicsConfigurationError("LAB-001: excited j is incompatible with l for an alkali FS state. F05.")

    for n1 in range(n_lo, n_hi):
        for l1 in range(min(n1, max_l + 1)):
            if abs(l1 - excited_l) != 1:            # coupling-side |Δl| = 1
                continue
            for j1 in ([l1 - 0.5, l1 + 0.5] if l1 > 0 else [0.5]):
                if abs(j1 - excited_j) > 1:          # coupling-side |Δj| <= 1
                    continue
                for n2 in range(n_lo, n_hi):
                    for l2 in range(min(n2, max_l + 1)):
                        if abs(l1 - l2) != 1:        # RF-side |Δl| = 1
                            continue
                        for j2 in ([l2 - 0.5, l2 + 0.5] if l2 > 0 else [0.5]):
                            if abs(j1 - j2) > 1:      # RF-side |Δj| <= 1
                                continue
                            combinations_checked += 1
                            try:
                                signed_hz = arc_real_scalar(atom.getTransitionFrequency(
                                    n1, l1, j1, n2, l2, j2),
                                    f"ARC RF frequency ({n1},{l1},{j1})->({n2},{l2},{j2}) [Hz]",
                                    nonzero=True)
                            except PhysicsError:
                                raise
                            except Exception as exc:
                                raise PhysicsCalculationError(
                                    "AUTO-001A-R1: ARC transition frequency "
                                    f"failed for ({n1},{l1},{j1})->"
                                    f"({n2},{l2},{j2}): {type(exc).__name__}: "
                                    f"{exc}. F05: a scan failure is never "
                                    "dropped as a domain exclusion."
                                ) from exc
                            if not math.isfinite(signed_hz):
                                raise PhysicsCalculationError(
                                    "AUTO-001A-R1: ARC returned a non-finite "
                                    f"frequency for ({n1},{l1},{j1})->"
                                    f"({n2},{l2},{j2}): {signed_hz!r}. "
                                    "F05: no fabricated replacement.")
                            carrier_hz = abs(signed_hz)
                            mismatch_hz = abs(carrier_hz - applied_carrier_hz)
                            if mismatch_hz > frequency_tolerance_hz:
                                continue            # domain exclusion, not failure
                            from_level = AtomicLevel(
                                n=n1, l=l1, j=j1,
                                label=f"{n1}{_L_LETTERS[l1]}{j1:.1f}")
                            to_level = AtomicLevel(
                                n=n2, l=l2, j=j2,
                                label=f"{n2}{_L_LETTERS[l2]}{j2:.1f}")
                            candidates.append({
                                'from_level': from_level,
                                'to_level': to_level,
                                'transition_freq_hz': carrier_hz,
                                'signed_transition_freq_hz': signed_hz,
                                'mismatch_hz': mismatch_hz,
                                # legacy alias — the same physical quantity
                                'rf_frequency_error_hz': mismatch_hz,
                            })
                            logger.debug(
                                f"FS RF candidate: {from_level.label} -> "
                                f"{to_level.label} (|nu|={carrier_hz/1e6:.3f} "
                                f"MHz, mismatch={mismatch_hz/1e6:.3f} MHz)")
    return candidates, combinations_checked


class ARCOptimizer:
    """
    Phase 1: RF-first ARC-based optimization.

    Scans the configured RF n-range with restricted l (AUTO-001A-R1:
    fine-structure selection rules only) to find RF transition pairs
    (n1→n2) whose ACTUAL ARC carrier matches the applied RF frequency.
    """

    def __init__(self, arc_atom=None):
        """
        Initialize ARC optimizer.

        Args:
            arc_atom: ARC atom object (e.g., Rubidium85())
        """
        self.arc_interface = ARCInterface(arc_atom)

    def optimize_levels(self, narrowed_ranges: NarrowedRanges,
                        probe_params: Any,
                        coupling_params: Any,
                        rf_params: Any,
                        criteria: OptimizationCriteria = None) -> Dict[str, Any]:
        """
        Perform RF-first ARC-based level optimization.

        Phase 1: Find all valid RF transitions (n1→n2 pairs) matching RF frequency
        using ARC across restricted n and l ranges.

        Args:
            narrowed_ranges: Search ranges (used for compatibility)
            probe_params: Probe laser parameters
            coupling_params: Coupling laser parameters
            rf_params: RF field parameters
            criteria: Optimization criteria

        Returns:
            Dictionary with RF transition pairs and level selection
            (AUTO-001A-R2, FS-only: no dipole matrix and no fabricated
            transition object are produced by selection)
        """

        logger.info("Phase 1: RF-first level finding - scanning RF frequency to determine Rydberg states")
        atom = self.arc_interface.atom
        if atom is None:
            raise PhysicsConfigurationError("LAB-001: RF-first optimization requires an ARC atom. F05.")
        applied_carrier_hz = real_scalar(rf_params.frequency_hz, "rf_params.frequency_hz")
        if applied_carrier_hz <= 0:
            raise PhysicsConfigurationError("LAB-001: applied RF carrier must be positive. F05.")
        logger.info(f"Target RF frequency: {applied_carrier_hz/1e9:.2f} GHz")

        if criteria is None:
            criteria = OptimizationCriteria()

        # Step 1: Find ground state using ARC (atom type only)
        ground_level = self.arc_interface.find_ground_state()
        ground_level.energy = _arc_level_energy(atom, ground_level)
        logger.info(f"Ground state (ARC): {ground_level.label}")

        # Step 2: Find excited state from ground state + probe wavelength using ARC
        from ..arc.ground_states import find_excited_state
        excited_level = find_excited_state(self.arc_interface.atom, probe_params.wavelength_nm)
        if not excited_level:
            raise PhysicsCalculationError(
                f"LAB-001: No excited state found for probe wavelength {probe_params.wavelength_nm} nm. F05: no alternate state.")
        logger.info(f"Excited state (ARC from probe wavelength): {excited_level.label}")
        excited_level.energy = _arc_level_energy(atom, excited_level)
        # Step 3: FS-only RF candidate scan (AUTO-001A-R1) — independent n1/n2
        # inside the configured RF range; fine-structure selection rules only
        # (no mJ/q scan and no dipole calculation during selection).
        rf_n_range = (narrowed_ranges.rf_n_range[0], narrowed_ranges.rf_n_range[1])

        # The applied carrier is the single RF-first ranking reference.
        logger.info(f"Scanning RF transitions in stop-exclusive n={rf_n_range} for {applied_carrier_hz/1e9:.2f} GHz")
        try:
            freq_tolerance = LEVEL_SELECTION_TOLERANCES['frequency_tolerance_hz']
        except KeyError as exc:
            raise PhysicsConfigurationError("LAB-001: required frequency_tolerance_hz is missing. F05.") from exc

        rf_transitions_found, combinations_checked = scan_fs_rf_candidates(
            atom, rf_n_range, excited_level, applied_carrier_hz, freq_tolerance)

        if not rf_transitions_found:
            raise PhysicsCalculationError(
                f"LAB-001: No RF manifold candidate inside n={rf_n_range} "
                f"(stop-exclusive), l=0..{RF_MAX_L_TO_CHECK}, "
                f"carrier={applied_carrier_hz:g} Hz, tolerance={freq_tolerance:g} Hz; "
                f"checked {combinations_checked} FS-eligible pairs. F05: no widened search.")

        logger.info(f"RF transitions found: {len(rf_transitions_found)}. Total combinations checked: {combinations_checked}")

        # RF-first reported ordering — the deterministic AUTO-001A-R1 key.
        rf_transitions_found.sort(key=rf_first_ranking_key)

        # AUTO-001A — RF-first manifold resolver: the same ordering selects
        # the pair, and ARC's signed frequency fixes its energy orientation
        # now (RF-DIR-001) instead of re-inferring it downstream.
        best_rf_pair = resolve_rf_first_manifold(
            rf_transitions_found, atom, applied_carrier_hz)

        # --- CONSISTENT COUPLING WAVELENGTH CALCULATION ---
        # Calculate coupling wavelength from the *actual* excited state found (Step 2)
        # to the selected Rydberg1 state.
        rydberg1 = best_rf_pair['from_level']
        rydberg2 = best_rf_pair['to_level']
        rydberg1.energy = _arc_level_energy(atom, rydberg1)  # eV
        rydberg2.energy = _arc_level_energy(atom, rydberg2)  # eV
        # we can calculate frequency of each levels via E = h*f => f = E/h
        #hartree_to_joules = 4.3597447222085515e-18
        #rydberg1.frequency_THz = rydberg1.energy * hartree_to_joules / const.H * 1e-12 # in Hz
        #rydberg2.frequency_THz = rydberg2.energy * hartree_to_joules / const.H * 1e-12# in Hz
        #rydberg1.wavelength_nm = (const.C_LIGHT / (rydberg1.frequency_THz * 1e12)) * 1e9
        #rydberg2.wavelength_nm = (const.C_LIGHT / (rydberg2.frequency_THz * 1e12)) * 1e9
        excited = excited_level # Use the dynamically found excited state

        try:
            # ARC getTransitionFrequency returns Hz.
            coupling_freq_hz = arc_real_scalar(atom.getTransitionFrequency(
                excited.n, excited.l, excited.j, rydberg1.n, rydberg1.l, rydberg1.j
            ), f"ARC coupling frequency {excited.label} -> {rydberg1.label} [Hz]", positive=True)
            coupling_freq_mhz = coupling_freq_hz * 1e-6
            coupling_wavelength_nm = arc_real_scalar(
                (const.C_LIGHT / coupling_freq_hz) * 1e9, "derived coupling wavelength [nm]", positive=True)
            logger.info(f"Calculated coupling wavelength: {coupling_wavelength_nm:.2f} nm ({excited.label} -> {rydberg1.label})")
        except PhysicsError:
            raise
        except Exception as exc:
            # F05: the coupling wavelength is a REQUIRED physics input. The
            # historical fallback (a hard-coded/config default, which in this
            # module even failed with AttributeError because LASER_PARAMS is a
            # dict) is removed: fail loudly instead.
            raise PhysicsCalculationError(
                "Coupling wavelength could not be calculated from ARC for the "
                f"transition {excited.label} -> {rydberg1.label}: "
                f"{type(exc).__name__}: {exc}. "
                "F05: no default-wavelength substitution."
            ) from exc

        # AUTO-001A-R2: FS-only result boundary. The selection stage returns
        # the selected manifolds WITHOUT any magnetic-channel dipole matrix:
        # the legacy ``arc/dipole_matrix`` maximum-over-mJ scan is not the
        # declared QN-001B/C channel path and must never run here. A channel
        # dipole is computed only where a channel is EXPLICITLY declared (the
        # QN boundary then fails fast in the caller's Rabi step, unchanged).
        result = {
            'levels': {
                'ground': ground_level,
                'excited': excited_level,
                'rydberg1': best_rf_pair['from_level'],
                'rydberg2': best_rf_pair['to_level']
            },
            'rf_transitions_found': rf_transitions_found,
            'selected_rf_pair': best_rf_pair,
            'optimization_metrics': {
                'rf_frequency_error_hz': best_rf_pair['rf_frequency_error_hz'],
                'mismatch_hz': best_rf_pair['mismatch_hz'],
                'rf_applied_carrier_hz': applied_carrier_hz,
                'stage': 'RF_pair_found',
                'rf_pairs_count': len(rf_transitions_found),
                'target_rf_freq_hz': rf_params.frequency_hz,
                'actual_rf_freq_hz': best_rf_pair['transition_freq_hz'],
                'rf_energy_direction': best_rf_pair['rf_orientation'].direction,
                'rf_carrier_hz': best_rf_pair['rf_orientation'].carrier_hz,
                'signed_rf_frequency_hz': best_rf_pair['rf_orientation'].signed_frequency_hz,
                'coupling_wavelength_nm': coupling_wavelength_nm
            }
        }

        return result
