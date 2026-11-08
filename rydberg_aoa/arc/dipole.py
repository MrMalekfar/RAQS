"""
Dipole moment calculations for ARC Interface.

Provides dipole matrix element calculations for laser and RF transitions in atomic systems,
ensuring electric dipole selection rules and appropriate mj summation for Rydberg states.
"""

import logging
from typing import Optional
import numpy as np
from .types import AtomicLevel, validate_magnetic_quantum_number
from ..config.parameters import LEVEL_SELECTION_TOLERANCES as LSTol
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARC dipole unit conversion (single source of truth).
#
# Every ARC dipole matrix element is returned in atomic units (1 e·a0). The
# whole project converts with this ONE value, so the Rabi-frequency chain
# (arc/rabi_frequency.py) and the cached dipole matrix used by the
# susceptibility chain can never disagree about |d|.
#     exact CODATA value: 8.4783536255e-30 C·m (relative difference 4.2e-5)
# ---------------------------------------------------------------------------
EA0_TO_CM = 8.478e-30  # 1 e·a0 -> C·m


def calculate_dipole_matrix_element(atom, level1: AtomicLevel, n2: int, l2: int, j2: float):
    """Calculate dipole matrix element between two atomic levels with validation.

    Performs electric dipole selection rule checks (Δl = ±1, Δj = 0,±1),
    handles hyperfine vs fine structure appropriately, and validates dipole magnitude.

    Args:
        atom: ARC atom object.
        level1 (AtomicLevel): First atomic level.
        n2 (int): Principal quantum number for second level.
        l2 (int): Orbital angular momentum for second level.
        j2 (float): Total angular momentum for second level.

    Returns:
        dict or float: Dictionary with 'dipole_si' (C*m) and optional mj/q info, or float (dipole in C*m or 0.0).
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the dipole matrix element "
            f"{getattr(level1, 'label', level1)} -> {n2}{l2}{j2} cannot be "
            "calculated. F05: no fallback is permitted."
        )
    count = 0
    dipole_max_ea0 = 0
    # Enforce electric dipole selection rules: Δl = ±1, Δj = 0, ±1
    delta_l = abs(l2 - level1.l)
    delta_j = abs(j2 - level1.j)

    if delta_l != 1:
        logger.debug(f"Invalid transition {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: Δl = {delta_l} ≠ 1 (electric dipole forbidden)")
        return 0.0

    if delta_j > 1:
        logger.debug(f"Invalid transition {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: Δj = {delta_j} > 1 (forbidden)")
        return 0.0

    try:
        dipole_max_ea0 = 0
        mj1_opt = 0  # Initialize
        mj2_opt = 0  # Initialize  
        q_opt = 0    # Initialize
        count = 0
        # Sum mj states for total dipole strength in Rydberg-Rydberg transitions
        if level1.n > 10 and n2 > 10:

            # Limit MJ range to cover high-j states (D5/2: j=2.5 needs mj=±2.5, P3/2: mj=±1.5, etc.)
            max_mj = max_mj = max(int(level1.j * 2) + 1, int(j2 * 2) + 1) // 2  # Increased for half-integer j states up to j=3.5
            for mj1_val in np.arange(-max_mj, max_mj + 0.5, 0.5):  # Include half-integers
                for mj2_val in np.arange(-max_mj, max_mj + 0.5, 0.5):
                    # Try different polarizations q = mj2 - mj1
                    expected_q = int(mj2_val - mj1_val)
                    if expected_q in [-1, 0, 1]:  # Valid polarization
                        try:
                            dipole_test = atom.getDipoleMatrixElement(
                                level1.n, level1.l, level1.j, mj1_val,
                                n2, l2, j2, mj2_val,
                                expected_q
                            )
                            # Update the maximum dipole found, and store the associated quantum numbers
                            if abs(dipole_test) > dipole_max_ea0:
                                dipole_max_ea0 = abs(dipole_test)
                                mj1_opt = mj1_val
                                mj2_opt = mj2_val
                                q_opt = expected_q
                                count += 1
                            #
                        except:
                            continue

            if count == 0:
                # F05: no mj contribution was found for a Rydberg-Rydberg
                # transition whose element the model is about to use. The
                # historical fabricated "physics minimum" value is removed;
                # fail loudly instead.
                raise PhysicsCalculationError(
                    f"No mj contribution found for the Rydberg-Rydberg "
                    f"transition {level1.n}{level1.l}{level1.j} -> "
                    f"{n2}{l2}{j2} (mj1={level1.mj}). The model explicitly "
                    "requires this dipole matrix element. "
                    "F05: no placeholder dipole."
                )

            dipole = dipole_max_ea0
            logger.debug(f"Rydberg-Rydberg dipole {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: {dipole:.2e} ea0 (summed over {count} mj combinations)")
        elif level1.f is not None and n2 <= 10:  # Use HFS when level1 has hyperfine structure and n2 is low-lying
            # Use hyperfine-resolved method
            f2 = j2 + atom.I  # Total angular momentum for final state
            mf2 = 0  # Assume stretched state
            dipole = 0.0
            for q in [-1, 0, 1]:  # Try different polarizations
                try:
                    dipole_test = atom.getDipoleMatrixElementHFS(
                        level1.n, level1.l, level1.j, level1.f, level1.mf,
                        n2, l2, j2, f2, mf2,
                        q
                    )
                    dipole = max(dipole, abs(dipole_test))
                except:
                    continue
            logger.debug(f"Hyperfine dipole {level1.n}{level1.l}{level1.j} F={level1.f} -> {n2}{l2}{j2}: {dipole:.2e} ea0")
        else:
            # Fine structure for transitions involving Rydberg states
            # Include mj summation for transitions to/from Rydberg levels to capture all coupling strengths
            if n2 > 10 or level1.n > 10:
                # For transitions involving Rydberg states, sum over mj states for complete coupling
                dipole_max_ea0 = 0
                count = 0
                max_mj = min(4, int(level1.j * 2) + 1)  # Limit based on j values

                for mj1_val in np.arange(-level1.j, level1.j + 1e-10, 0.5):
                    for mj2_val in np.arange(-j2, j2 + 1e-10, 0.5):
                        expected_q = int(mj2_val - mj1_val)
                        if expected_q in [-1, 0, 1]:
                            try:
                                dipole_test = atom.getDipoleMatrixElement(
                                    level1.n, level1.l, level1.j, mj1_val,
                                    n2, l2, j2, mj2_val,
                                    expected_q
                                )
                                logger.debug(f"MJ combination mj1={mj1_val}, mj2={mj2_val}, q={expected_q}: dipole={dipole_test:.2e}")
                                if abs(dipole_test) > dipole_max_ea0:
                                    dipole_max_ea0 = abs(dipole_test)
                                    mj1_opt = mj1_val
                                    mj2_opt = mj2_val
                                    q_opt = expected_q
                                    count += 1
                            except Exception as e:
                                logger.debug(f"MJ combination mj1={mj1_val}, mj2={mj2_val}, q={expected_q} failed: {e}")
                                continue

                if count == 0:
                    # F05: the historical fabricated "physics minimum" value
                    # for Rydberg-involved transitions is removed; fail loudly.
                    raise PhysicsCalculationError(
                        f"No mj contribution found for the Rydberg-involved "
                        f"transition {level1.n}{level1.l}{level1.j} -> "
                        f"{n2}{l2}{j2} (mj1={level1.mj}). The model "
                        "explicitly requires this dipole matrix element. "
                        "F05: no placeholder dipole."
                    )

                dipole = dipole_max_ea0  # Use the maximum found
                logger.debug(f"Rydberg-involved dipole {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: {dipole:.2e} ea0 (mj summed, {count} combinations)")
            else:
                # For low-lying to low-lying transitions, use simple max over polarizations
                dipole = 0.0
                for q in [-1, 0, 1]:  # Try different polarizations
                    try:
                        dipole_test = atom.getDipoleMatrixElement(
                            level1.n, level1.l, level1.j, 0,
                            n2, l2, j2, 0,
                            q
                        )
                        dipole = max(dipole, abs(dipole_test))
                    except:
                        continue
                logger.debug(f"Regular dipole {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: {dipole:.2e} ea0")

        dipole_si = abs(dipole) * EA0_TO_CM  # Convert to SI units

        # F05 validation: a NaN/inf dipole must never reach the model.
        if not np.isfinite(dipole_si):
            raise PhysicsCalculationError(
                f"ARC dipole matrix element for "
                f"{level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2} converted to "
                f"a non-finite value ({dipole_si!r} C*m) from raw element "
                f"{dipole!r} ea0. F05: no fallback."
            )

        # Validate dipole magnitude - use different thresholds for different transition types
        dipole_threshold = LSTol['min_dipole_threshold'] if (level1.n > 10 or n2 > 10) else LSTol['min_dipole_threshold']*10  # More lenient for Rydberg transitions
        if dipole_si < dipole_threshold:
            # Documented, deterministic tolerance-based rejection of a
            # negligibly small element (LEVEL_SELECTION_TOLERANCES). This is
            # NOT a failure fallback: the element is real and simply below the
            # model's numerical significance threshold.
            logger.warning(f"Negligible dipole for {level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2}: {dipole_si:.2e} C*m is below the significance threshold {dipole_threshold:.2e} C*m; returning 0.0 (tolerance-based rejection, not a fallback)")
            return 0.0

        return {
        'dipole_si': dipole_si,
        'mj1': mj1_opt,
        'mj2': mj2_opt,
        'q': q_opt
         }

    except PhysicsError:
        raise
    except Exception as exc:
        # F05: the designated ARC dipole calculation failed. The historical
        # placeholder values (1e-30 C*m for Rydberg-Rydberg transitions, 0.0
        # otherwise) are REMOVED: they injected a fabricated dipole moment
        # into the model. Fail loudly with the full context instead.
        raise PhysicsCalculationError(
            f"ARC dipole calculation failed for "
            f"{level1.n}{level1.l}{level1.j} -> {n2}{l2}{j2} "
            f"(mj1={level1.mj}): {type(exc).__name__}: {exc}. "
            "F05: no placeholder dipole and no zero substitution."
        ) from exc


# ---------------------------------------------------------------------------
# QN-001B.1 — exact STRUCTURAL validation of the declared HFS -> FS magnetic
# channel (no dipole thresholds, no numerical cutoffs).
#
# For an HFS -> FS E1 transition driven by polarisation q:
#
#     |n1 l1 j1 F1 mF1>  -q->  |n2 l2 j2 mJ2>
#
# projection conservation fixes the uncoupled components the field connects:
#
#     mJ2 = mJ1 + q   =>   mJ1 = mJ2 - q       (electronic projection)
#     mF1 = mJ1 + mI  =>   mI  = mF1 - mJ1     (nuclear projection)
#
# The channel is STRUCTURALLY forbidden — identically zero angular amplitude,
# NOT merely weak — when either implied projection cannot exist:
#
#     |mJ1| > j1        or        |mI| > I      (I = atom.I, the isotope spin)
#
# Such a channel is rejected BEFORE ARC's dipole evaluation, so the decision
# never depends on floating-point residue (ARC 3.9.0 returns ~7.5e-19 e*a0
# for some exactly-forbidden channels instead of exactly 0.0). This is
# deliberately NOT a dipole-magnitude threshold: an allowed but weak
# transition is never rejected for being small, and the post-ARC exact-zero
# guard of QN-001B is retained for channels that pass these projection bounds
# but whose Clebsch/Wigner coefficient still vanishes.
# ---------------------------------------------------------------------------
def _validate_hfs_to_fs_magnetic_channel(atom, from_level: AtomicLevel,
                                         to_level: AtomicLevel, q: int) -> None:
    """Reject a structurally impossible declared HFS -> FS channel (QN-001B.1).

    A hyperfine source state ``|F1 mF1>`` is a superposition of uncoupled
    ``|j1 mJ1> |I mI>`` components with ``mF1 = mJ1 + mI``. The E1 field of
    polarisation ``q`` connects only the component satisfying
    ``mJ2 = mJ1 + q``. The declared channel ``|F1 mF1> -> |mJ2>`` can be
    non-zero only if BOTH implied projections exist:

        mJ1 = mJ2 - q    with   |mJ1| <= j1  (and j1 - mJ1 an integer)
        mI  = mF1 - mJ1  with   |mI|  <= I   (and I  - mI  an integer)

    The projection arithmetic is delegated to the single QN-001A validator
    ``types.validate_magnetic_quantum_number`` (half-integer-safe bounds and
    integer-step rule); this helper adds only the HFS -> FS channel
    interpretation and the isotope nuclear spin from the ACTUAL atom (``I``).

    Args:
        atom: ARC atom object; the nuclear spin comes from ``atom.I``.
        from_level: source level; the caller has already required explicit
            ``F`` and ``mF`` (``j`` is validated by ``AtomicLevel``).
        to_level: destination level; the caller has already required an
            explicit ``mJ``.
        q: already-validated polarisation in {-1, 0, +1}.

    Returns:
        None — the channel is structurally admissible; the caller proceeds to
        exactly ONE ARC dipole evaluation.

    Raises:
        PhysicsConfigurationError: the isotope nuclear spin is unavailable or
            non-finite, or the implied ``mJ1`` / ``mI`` is not a valid
            projection. Every message states that no alternate mJ or q is
            selected automatically (QN-001B: compute a channel, never select
            one).
    """
    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")
    q_int = int(q)
    channel = (f"(F1={from_level.f}, mF1={from_level.mf}, "
               f"mJ2={to_level.mj}, q={q_int})")

    # isotope nuclear spin — from the ACTUAL atom, never a hard-coded guess
    try:
        nuclear_spin = float(atom.I)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"cannot structurally validate the HFS->FS channel {transition} "
            f"{channel}: the ARC atom exposes no usable isotope nuclear spin I "
            f"(atom.I = {getattr(atom, 'I', '<absent>')!r}). QN-001B.1: the "
            "structural projection check requires the actual isotope spin; "
            "no fallback is permitted."
        ) from exc
    if not np.isfinite(nuclear_spin):
        raise PhysicsConfigurationError(
            f"cannot structurally validate the HFS->FS channel {transition} "
            f"{channel}: the isotope nuclear spin is not finite "
            f"(atom.I = {nuclear_spin!r}). QN-001B.1: no fallback is "
            "permitted."
        )

    # mJ1 = mJ2 - q — the field raises the electronic projection by exactly q
    mj1 = float(to_level.mj) - q_int
    try:
        validate_magnetic_quantum_number(
            mj1, from_level.j, "mJ1 (implied by mJ2 - q)", "j1")
    except PhysicsConfigurationError as exc:
        if abs(mj1) > float(from_level.j):
            reason = (f"mJ1 = mJ2 - q = {mj1} lies outside [-j1,+j1] "
                      f"(j1 = {float(from_level.j)})")
        else:
            reason = (f"mJ1 = mJ2 - q = {mj1} is not a valid projection of "
                      f"j1 = {float(from_level.j)}")
        raise PhysicsConfigurationError(
            "The explicitly configured HFS->FS channel is structurally "
            f"forbidden: {reason} for {transition} {channel}. Structural "
            f"detail: {exc} No alternate mJ or q is selected automatically. "
            "QN-001B.1: this exact projection check runs before ARC evaluates "
            "the dipole matrix element."
        ) from exc

    # mI = mF1 - mJ1 — the HFS basis requires mF1 = mJ1 + mI
    m_i = float(from_level.mf) - mj1
    try:
        validate_magnetic_quantum_number(
            m_i, nuclear_spin, "mI (implied by mF1 - mJ1)", "I")
    except PhysicsConfigurationError as exc:
        if abs(m_i) > nuclear_spin:
            reason = (f"the implied nuclear projection mI = mF1 - mJ1 = {m_i} "
                      f"lies outside [-I,+I] (I = {nuclear_spin})")
        else:
            reason = (f"the implied nuclear projection mI = mF1 - mJ1 = {m_i} "
                      f"is not a valid projection of I = {nuclear_spin}")
        raise PhysicsConfigurationError(
            "The explicitly configured HFS->FS channel is structurally "
            f"forbidden: {reason} for {transition} {channel}. Structural "
            f"detail: {exc} No alternate mJ or q is selected automatically. "
            "QN-001B.1: this exact projection check runs before ARC evaluates "
            "the dipole matrix element."
        ) from exc


# ---------------------------------------------------------------------------
# QN-001B — explicit HFS -> FS magnetic-channel dipole matrix element.
#
# This primitive computes EXACTLY ONE DECLARED magnetic channel
#
#     |n1 l1 j1 F1 mF1>  ->  |n2 l2 j2 mJ2>   with laser polarisation q
#
# through ARC's dedicated HFS -> FS element. It never SELECTS a channel: there
# is no mJ inference, no mJ scan, no q scan and no maximum over magnetic
# sublevels. The configuration declares the channel; the caller decides which
# channel that is.
# ---------------------------------------------------------------------------
def calculate_hfs_to_fs_dipole_matrix_element(atom, from_level: AtomicLevel,
                                              to_level: AtomicLevel,
                                              q: int) -> float:
    """SIGNED ARC HFS -> FS dipole matrix element [e·a0] for ONE declared channel.

    Physics (QN-001B)
    -----------------
    A laser with polarisation q couples the hyperfine source state
    ``|n1 l1 j1 F1 mF1>`` to the fine-structure destination state
    ``|n2 l2 j2 mJ2>``; ARC evaluates the matrix element

        d = atom.getDipoleMatrixElementHFStoFS(
                n1, l1, j1, F1, mF1, n2, l2, j2, mJ2, q)           [e·a0]

    The destination ``mJ`` must be EXPLICIT. ``mJ`` is the projection of the
    electronic total angular momentum ``j`` while ``mF`` is the projection of
    the hyperfine ``F``: they are different quantum numbers, neither is
    inferred from the other (QN-001A). The SIGNED value is preserved so that a
    future complex/multichannel coupling can use its phase; the magnitude
    ``|d|`` is taken only by the Rabi helper.

    Args:
        atom: ARC atom object.
        from_level: source level; must carry an explicit ``F`` and ``mF``.
        to_level: destination level; must carry an explicit ``mJ`` and must
            NOT simultaneously declare hyperfine quantum numbers (an
            HFS-resolved destination is a different, ambiguous declaration).
        q: laser polarisation in {-1, 0, +1} (+1 = sigma+, 0 = pi,
            -1 = sigma-). Required, never defaulted, never scanned.

    Returns:
        float: SIGNED dipole matrix element in ``e·a0`` (never SI units,
        never ``|d|``).

    Raises:
        PhysicsConfigurationError: missing source ``F``/``mF``, missing
            destination ``mJ``, an ambiguous destination declaration, an
            E1-forbidden declared transition, an invalid ``q``, a
            STRUCTURALLY forbidden magnetic channel — the implied
            ``mJ1 = mJ2 - q`` or ``mI = mF1 - mJ1`` is not a valid projection
            of ``j1`` / ``I`` (checked exactly, BEFORE ARC is called,
            QN-001B.1) — or a genuinely dark (exactly zero) declared channel.
        PhysicsCalculationError: ARC raised, or returned ``None`` /
            non-numeric / NaN / Inf. F05: no fallback.
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the HFS-to-FS dipole matrix element "
            f"{getattr(from_level, 'label', from_level)} -> "
            f"{getattr(to_level, 'label', to_level)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")

    # ---- source: an EXPLICIT hyperfine sublevel ----------------------------
    if from_level.f is None or from_level.mf is None:
        raise PhysicsConfigurationError(
            f"the HFS-to-FS dipole matrix element for {transition} requires "
            "an explicit hyperfine SOURCE sublevel (F, mF); got "
            f"F={from_level.f}, mF={from_level.mf}. QN-001B: the source HFS "
            "state is never inferred."
        )

    # ---- destination: an EXPLICIT fine-structure sublevel ------------------
    if to_level.mj is None:
        raise PhysicsConfigurationError(
            f"the HFS-to-FS dipole matrix element for {transition} requires "
            "an explicit destination FS magnetic sublevel (mJ); got mJ=None. "
            "QN-001B: an explicit mJ is required, no mJ is inferred and no "
            "mJ scan is performed."
        )
    if to_level.f is not None or to_level.mf is not None:
        raise PhysicsConfigurationError(
            f"ambiguous destination declaration for {transition}: the "
            "destination carries a fine-structure magnetic sublevel "
            f"(mJ={to_level.mj}) AND hyperfine quantum numbers "
            f"(F={to_level.f}, mF={to_level.mf}). An HFS-to-FS destination is "
            "the unresolved-HFS fine-structure state and must not declare "
            "both. QN-001B: refusing the ambiguous declaration."
        )

    # ---- structural electric-dipole (E1) selection rules -------------------
    delta_l = abs(to_level.l - from_level.l)
    delta_j = abs(to_level.j - from_level.j)
    if delta_l != 1:
        raise PhysicsConfigurationError(
            f"the declared HFS-to-FS transition {transition} violates the "
            f"electric-dipole selection rule (delta_l = {delta_l} != 1). "
            "QN-001B: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )
    if delta_j > 1:
        raise PhysicsConfigurationError(
            f"the declared HFS-to-FS transition {transition} violates the "
            f"electric-dipole selection rule (delta_j = {delta_j} > 1). "
            "QN-001B: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )
    if from_level.j == 0 and to_level.j == 0:
        raise PhysicsConfigurationError(
            f"the declared HFS-to-FS transition {transition} violates the "
            "electric-dipole selection rule (j = 0 -> j = 0 is forbidden). "
            "QN-001B: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )

    # ---- configured polarisation (validated, never scanned) ----------------
    try:
        q_int = int(q)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} is missing or is not an "
            f"integer ({q!r}); q must be one of -1, 0, +1. QN-001B: the "
            "polarisation is a required input, never defaulted or scanned."
        ) from exc
    if float(q) != float(q_int) or q_int not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} must be one of -1, 0, +1 "
            f"(got {q!r}). QN-001B: no hard-coded q, no q scan."
        )

    # ---- exact structural magnetic-channel validation (QN-001B.1) ----------
    # Runs BEFORE ARC: a structurally impossible channel (implied mJ1 or mI
    # outside its allowed projections) has an IDENTICALLY zero angular
    # amplitude, so the decision must never rest on ARC floating-point
    # residue. A structurally admissible channel proceeds to exactly ONE ARC
    # call (the QN-001B one-call contract).
    _validate_hfs_to_fs_magnetic_channel(atom, from_level, to_level, q_int)

    # ---- designated ARC calculation (the ONE declared channel) -------------
    try:
        dipole_ea0 = atom.getDipoleMatrixElementHFStoFS(
            from_level.n, from_level.l, from_level.j,
            from_level.f, from_level.mf,
            to_level.n, to_level.l, to_level.j,
            to_level.mj,
            q_int,
        )
    except PhysicsError:
        # F05: a Physics* failure is already the correct contextual
        # fail-fast error; never replace it with a fallback value.
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC HFS-to-FS dipole matrix element failed for {transition} "
            f"(F1={from_level.f}, mF1={from_level.mf} -> "
            f"mJ2={to_level.mj}, q={q_int}): "
            f"{type(exc).__name__}: {exc}. F05: no fallback and no zero "
            "substitution."
        ) from exc

    if dipole_ea0 is None:
        raise PhysicsCalculationError(
            f"ARC returned no HFS-to-FS dipole matrix element for "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"mJ2={to_level.mj}, q={q_int}). F05: no fallback is permitted."
        )
    try:
        dipole_val = float(dipole_ea0)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"ARC returned a non-numeric HFS-to-FS dipole matrix element for "
            f"{transition} ({dipole_ea0!r}). F05: no fallback."
        ) from exc
    if not np.isfinite(dipole_val):
        raise PhysicsCalculationError(
            f"ARC returned a non-finite HFS-to-FS dipole matrix element for "
            f"{transition} ({dipole_val!r} e·a0). F05: no fallback."
        )

    # An explicitly declared channel may legitimately be dark: the combination
    # (F1, mF1, mJ2, q) can be dipole-forbidden and ARC then returns exactly
    # zero. That is a CONFIGURATION error here — never a reason to try another
    # mJ or another q, and never a value to substitute.
    if dipole_val == 0.0:
        raise PhysicsConfigurationError(
            "The explicitly configured HFS->FS magnetic channel "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"mJ2={to_level.mj}, q={q_int}) has zero electric-dipole coupling "
            "for the configured polarization. No alternate mJ or q is "
            "selected automatically. QN-001B: configure a different explicit "
            "channel instead."
        )

    logger.debug(
        "QN-001B signed HFS->FS element for %s (F=%.1f, mF=%g -> mJ=%g, "
        "q=%d): %.9e e·a0",
        transition, from_level.f, from_level.mf, to_level.mj, q_int,
        dipole_val)
    return dipole_val  # SIGNED, e·a0 — never |d| and never SI


# ---------------------------------------------------------------------------
# QN-001C.1 — exact STRUCTURAL validation of the declared FS -> FS magnetic
# channel (no dipole thresholds, no numerical cutoffs).
#
# For an E1 transition driven with polarisation q, ARC factorises its element
# as
#
#     d = (-1)^(j1-mj1) * wigner3j(j1, 1, j2; -mj1, -q, mj2) * <j1||er||j2>
#
# and the Wigner 3j symbol vanishes IDENTICALLY unless the spherical
# projection rule
#
#     mj2 = mj1 + q          (equivalently -mj1 - q + mj2 = 0)
#
# holds. Both mJ values are already validated projections of their own j by
# ``AtomicLevel.__post_init__`` (QN-001A), so this helper checks ONLY the
# channel relation. It runs BEFORE ARC is ever called: a structurally
# impossible channel has an IDENTICALLY zero angular amplitude, so the
# decision must never rest on ARC floating-point residue (the QN-001B.1
# lesson). It never selects an alternate mJ or q (QN-001C: compute a
# channel, never select one).
# ---------------------------------------------------------------------------
_FS_FS_CHANNEL_TOL = 1e-9  # exact projection arithmetic (integers / half-integers)


def _validate_fs_to_fs_magnetic_channel(from_level: AtomicLevel,
                                        to_level: AtomicLevel, q_int: int) -> None:
    """Reject a structurally impossible declared FS -> FS magnetic channel.

    Physics (QN-001C.1)
    -------------------
    For the declared channel ``|n1 l1 j1 mJ1> -> |n2 l2 j2 mJ2>`` with
    polarisation ``q`` the exact selection rule is ``mj2 = mj1 + q``.

    Args:
        from_level: source level; must already carry an explicit ``mj``.
        to_level: destination level; must already carry an explicit ``mj``.
        q_int: validated polarisation in {-1, 0, +1}.

    Raises:
        PhysicsConfigurationError: if either projection is non-numeric /
            non-finite, or if ``mj2 - mj1`` differs from ``q_int``. The
            message always states that no alternate mJ or q is selected
            automatically.
    """
    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")
    channel = f"mJ1={from_level.mj}, mJ2={to_level.mj}, q={q_int}"
    try:
        mj1 = float(from_level.mj)
        mj2 = float(to_level.mj)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"cannot structurally validate the FS->FS channel {transition} "
            f"({channel}): the declared projections are not numeric "
            f"(mJ1={from_level.mj!r}, mJ2={to_level.mj!r}). QN-001C.1: no "
            "fallback is permitted."
        ) from exc
    if not (np.isfinite(mj1) and np.isfinite(mj2)):
        raise PhysicsConfigurationError(
            f"cannot structurally validate the FS->FS channel {transition} "
            f"({channel}): the declared projections are not finite "
            f"(mJ1={mj1!r}, mJ2={mj2!r}). QN-001C.1: no fallback is permitted."
        )
    if abs((mj2 - mj1) - q_int) > _FS_FS_CHANNEL_TOL:
        raise PhysicsConfigurationError(
            "The explicitly configured FS->FS magnetic channel is structurally "
            f"forbidden: mj2 - mj1 = {mj2 - mj1} does not equal the "
            f"polarisation q = {q_int} (the E1 projection rule mj2 = mj1 + q) "
            f"for {transition} ({channel}). Structural detail: the angular "
            "factor (-1)^(j1-mj1) wigner3j(j1,1,j2;-mj1,-q,mj2) vanishes "
            "identically for this channel. No alternate mJ or q is selected "
            "automatically. QN-001C.1: this exact projection check runs before "
            "ARC evaluates the dipole matrix element."
        )


# ---------------------------------------------------------------------------
# QN-001C — explicit FS -> FS magnetic-channel dipole matrix element.
#
# This primitive computes EXACTLY ONE DECLARED magnetic channel
#
#     |n1 l1 j1 mJ1>  ->  |n2 l2 j2 mJ2>   with field polarisation q
#
# through ARC's fine-structure element ``getDipoleMatrixElement``. It never
# SELECTS a channel: there is no mJ scan, no q scan and no maximum over
# magnetic sublevels — unlike the generic ``calculate_dipole_matrix_element``
# helper, which silently chose the strongest channel and ignored the
# configured polarisation entirely. The configuration declares the channel;
# the caller decides which channel that is.
# ---------------------------------------------------------------------------
def calculate_fs_to_fs_dipole_matrix_element(atom, from_level: AtomicLevel,
                                             to_level: AtomicLevel,
                                             q: int) -> float:
    """SIGNED ARC FS -> FS dipole matrix element [e·a0] for ONE declared channel.

    Physics (QN-001C)
    -----------------
    A field with polarisation q couples the fine-structure source state
    ``|n1 l1 j1 mJ1>`` to the fine-structure destination state
    ``|n2 l2 j2 mJ2>``; ARC evaluates the matrix element

        d = atom.getDipoleMatrixElement(
                n1, l1, j1, mJ1, n2, l2, j2, mJ2, q)            [e·a0]

    The magnetic projections ``mJ1`` and ``mJ2`` must BOTH be EXPLICIT:
    ``mJ1`` is the projection of ``j1`` and ``mJ2`` the projection of ``j2``.
    Neither is inferred from a hyperfine quantum number and neither is
    scanned (QN-001A/QN-001C). The SIGNED value is preserved so that its
    phase convention stays visible; the magnitude ``|d|`` is taken only by
    the Rabi helper.

    Args:
        atom: ARC atom object.
        from_level: source level; must carry an explicit ``mj`` and must NOT
            declare hyperfine quantum numbers (an HFS-resolved source is a
            different, ambiguous declaration routed elsewhere).
        to_level: destination level; must carry an explicit ``mj`` and must
            NOT declare hyperfine quantum numbers.
        q: field polarisation in {-1, 0, +1} (+1 = sigma+, 0 = pi,
            -1 = sigma-). Required, never defaulted, never scanned.

    Returns:
        float: SIGNED dipole matrix element in ``e·a0`` (never SI units,
        never ``|d|``).

    Raises:
        PhysicsConfigurationError: missing source/destination ``mJ``, an
            ambiguous (HFS + FS) declaration on either level, an
            E1-forbidden declared transition, an invalid ``q``, a
            STRUCTURALLY forbidden magnetic channel — ``mj2 - mj1 != q``
            (checked exactly, BEFORE ARC is called, QN-001C.1) — or a
            genuinely dark (exactly zero) declared channel.
        PhysicsCalculationError: ARC raised, or returned ``None`` /
            non-numeric / NaN / Inf. F05: no fallback.
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the FS-to-FS dipole matrix element "
            f"{getattr(from_level, 'label', from_level)} -> "
            f"{getattr(to_level, 'label', to_level)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")

    # ---- both levels: an EXPLICIT fine-structure sublevel -----------------
    for role, level in (("source", from_level), ("destination", to_level)):
        if level.mj is None:
            raise PhysicsConfigurationError(
                f"the FS-to-FS dipole matrix element for {transition} requires "
                f"an explicit fine-structure magnetic sublevel (mJ) on the "
                f"{role} level; got mJ=None ({getattr(level, 'label', level)}: "
                f"n={level.n}, l={level.l}, j={level.j}, F={level.f}, "
                f"mF={level.mf}). QN-001C: an explicit mJ is required on BOTH "
                "levels; no mJ is inferred and no mJ scan is performed."
            )
        if level.f is not None or level.mf is not None:
            raise PhysicsConfigurationError(
                f"ambiguous {role} declaration for {transition}: the level "
                f"carries a fine-structure magnetic sublevel (mJ={level.mj}) "
                f"AND hyperfine quantum numbers (F={level.f}, mF={level.mf}). "
                "An FS-to-FS channel couples unresolved-HFS fine-structure "
                "states and must not declare both. QN-001C: refusing the "
                "ambiguous declaration."
            )
    # ---- structural electric-dipole (E1) selection rules -------------------
    delta_l = abs(to_level.l - from_level.l)
    delta_j = abs(to_level.j - from_level.j)
    if delta_l != 1:
        raise PhysicsConfigurationError(
            f"the declared FS-to-FS transition {transition} violates the "
            f"electric-dipole selection rule (delta_l = {delta_l} != 1). "
            "QN-001C: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )
    if delta_j > 1:
        raise PhysicsConfigurationError(
            f"the declared FS-to-FS transition {transition} violates the "
            f"electric-dipole selection rule (delta_j = {delta_j} > 1). "
            "QN-001C: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )
    if from_level.j == 0 and to_level.j == 0:
        raise PhysicsConfigurationError(
            f"the declared FS-to-FS transition {transition} violates the "
            f"electric-dipole selection rule (j = 0 -> j = 0 is forbidden). "
            "QN-001C: an E1-forbidden declared channel is a configuration "
            "error, never a zero."
        )

    # ---- configured polarisation (validated, never scanned) ----------------
    try:
        q_int = int(q)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"polarization q for {transition} is missing or is not an integer "
            f"({q!r}); q must be one of -1, 0, +1. QN-001C: the polarisation "
            "is a required input, never defaulted or scanned."
        ) from exc
    if float(q) != float(q_int) or q_int not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"polarization q for {transition} must be one of -1, 0, +1 "
            f"(got {q!r}). QN-001C: no hard-coded q, no q scan."
        )

    # ---- exact structural magnetic-channel validation (QN-001C.1) ----------
    # Runs BEFORE ARC: a structurally impossible channel (mj2 - mj1 != q)
    # has an IDENTICALLY zero angular amplitude, so the decision must never
    # rest on ARC floating-point residue. A structurally admissible channel
    # proceeds to exactly ONE ARC call (the QN-001C one-call contract).
    _validate_fs_to_fs_magnetic_channel(from_level, to_level, q_int)

    # ---- designated ARC calculation (the ONE declared channel) -------------
    try:
        dipole_ea0 = atom.getDipoleMatrixElement(
            from_level.n, from_level.l, from_level.j, from_level.mj,
            to_level.n, to_level.l, to_level.j, to_level.mj,
            q_int,
        )
    except PhysicsError:
        # F05: a Physics* failure is already the correct contextual
        # fail-fast error; never replace it with a fallback value.
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC FS-to-FS dipole matrix element failed for {transition} "
            f"(mJ1={from_level.mj} -> mJ2={to_level.mj}, q={q_int}): "
            f"{type(exc).__name__}: {exc}. F05: no fallback and no zero "
            "substitution."
        ) from exc

    if dipole_ea0 is None:
        raise PhysicsCalculationError(
            f"ARC returned no FS-to-FS dipole matrix element for {transition} "
            f"(mJ1={from_level.mj} -> mJ2={to_level.mj}, q={q_int}). "
            "F05: no fallback is permitted."
        )
    try:
        dipole_val = float(dipole_ea0)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"ARC returned a non-numeric FS-to-FS dipole matrix element for "
            f"{transition} ({dipole_ea0!r}). F05: no fallback."
        ) from exc
    if not np.isfinite(dipole_val):
        raise PhysicsCalculationError(
            f"ARC returned a non-finite FS-to-FS dipole matrix element for "
            f"{transition} ({dipole_val!r} e·a0). F05: no fallback."
        )

    # A structurally admissible channel can still be dark (a vanishing
    # reduced element / angular coefficient). That is a CONFIGURATION error
    # here — never a reason to try another mJ or another q, and never a
    # value to substitute.
    if dipole_val == 0.0:
        raise PhysicsConfigurationError(
            "The explicitly configured FS->FS magnetic channel "
            f"{transition} (mJ1={from_level.mj} -> mJ2={to_level.mj}, "
            f"q={q_int}) has zero electric-dipole coupling. No alternate mJ "
            "or q is selected automatically. QN-001C: configure a different "
            "explicit channel instead."
        )

    logger.debug(
        "QN-001C signed FS->FS element for %s (mJ1=%g -> mJ2=%g, q=%d): "
        "%.9e e·a0",
        transition, from_level.mj, to_level.mj, q_int, dipole_val)
    return dipole_val  # SIGNED, e·a0 — never |d| and never SI
# ---------------------------------------------------------------------------
# QN-001A.1 / MODE-001 — explicit HFS -> HFS (hyperfine-resolved) magnetic
# channel dipole matrix element.
#
# QN-001A established the DECLARED-channel contract for the probe laser, but
# implemented it inside ``calculate_hfs_rabi_frequency`` (which necessarily
# also computes the Gaussian peak field).  The dipole-matrix CACHE, however,
# is populated through ``calculate_dipole_matrix`` ->
# ``calculate_dipole_matrix_element``, whose hyperfine branch sweeps
# ``q in {-1, 0, +1}`` and assumes the destination ``mF2 = 0``
# ("stretched state") instead of honouring the declared destination sublevel.
#
# MODE-001 (PRESCRIBED atomic resolution) needs the SAME declared-channel
# element available as a standalone DIPOLE, exactly as QN-001B and QN-001C
# exposed their legs: SIGNED, in ``e·a0``, from exactly ONE ARC call, with no
# q scan, no mF scan and no maximum taken over magnetic sublevels.
# ---------------------------------------------------------------------------
_HFS_HFS_CHANNEL_TOL = 1e-9  # exact projection arithmetic (integers/half-integers)


def _validate_hfs_to_hfs_magnetic_channel(from_level: AtomicLevel,
                                          to_level: AtomicLevel,
                                          q_int: int) -> None:
    """Reject a structurally impossible declared HFS -> HFS magnetic channel.

    Physics (QN-001A.1)
    -------------------
    For the declared channel ``|n1 l1 j1 F1 mF1> -> |n2 l2 j2 F2 mF2>`` driven
    with polarisation ``q`` the photon carries one unit of angular-momentum
    projection, so the exact projection rule is

        mF2 = mF1 + q          (equivalently  mF2 - mF1 - q = 0)

    Both ``mF`` values are already validated projections of their own ``F`` by
    ``AtomicLevel.__post_init__`` (QN-001A), so this helper checks ONLY the
    channel relation, and it runs BEFORE ARC is called: the angular factor
    vanishes IDENTICALLY for a mismatched channel, and that decision must never
    rest on ARC floating-point residue (the QN-001B.1 lesson).  It never
    selects an alternate ``mF`` or ``q``.

    Args:
        from_level: source level; must already carry an explicit ``F``/``mF``.
        to_level: destination level; must already carry an explicit ``F``/``mF``.
        q_int: validated polarisation in {-1, 0, +1}.

    Raises:
        PhysicsConfigurationError: if either projection is non-numeric /
            non-finite, or if ``mF2 - mF1`` differs from ``q_int``.
    """
    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")
    channel = f"mF1={from_level.mf}, mF2={to_level.mf}, q={q_int}"
    try:
        mf1 = float(from_level.mf)
        mf2 = float(to_level.mf)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"cannot structurally validate the HFS->HFS channel {transition} "
            f"({channel}): the declared projections are not numeric "
            f"(mF1={from_level.mf!r}, mF2={to_level.mf!r}). QN-001A.1: no "
            "fallback is permitted."
        ) from exc
    if not (np.isfinite(mf1) and np.isfinite(mf2)):
        raise PhysicsConfigurationError(
            f"cannot structurally validate the HFS->HFS channel {transition} "
            f"({channel}): the declared projections are not finite "
            f"(mF1={mf1!r}, mF2={mf2!r}). QN-001A.1: no fallback is permitted."
        )
    if abs((mf2 - mf1) - q_int) > _HFS_HFS_CHANNEL_TOL:
        raise PhysicsConfigurationError(
            "The explicitly configured HFS->HFS magnetic channel is "
            f"structurally forbidden: mF2 - mF1 = {mf2 - mf1} does not equal "
            f"the polarisation q = {q_int} (the E1 projection rule "
            f"mF2 = mF1 + q) for {transition} ({channel}). Structural detail: "
            "the angular factor vanishes identically for this channel. No "
            "alternate mF or q is selected automatically. QN-001A.1: this "
            "exact projection check runs before ARC evaluates the dipole "
            "matrix element."
        )


def calculate_hfs_to_hfs_dipole_matrix_element(atom, from_level: AtomicLevel,
                                               to_level: AtomicLevel,
                                               q: int) -> float:
    """SIGNED ARC HFS -> HFS dipole matrix element [e·a0] for ONE declared channel.

    Physics (QN-001A.1)
    -------------------
    A laser with polarisation q couples the hyperfine source state
    ``|n1 l1 j1 F1 mF1>`` to the hyperfine destination state
    ``|n2 l2 j2 F2 mF2>``; ARC evaluates the matrix element

        d = atom.getDipoleMatrixElementHFS(
                n1, l1, j1, F1, mF1, n2, l2, j2, F2, mF2, q)      [e·a0]

    Four quantum numbers are therefore DECLARED and never inferred: ``F`` and
    ``mF`` on both sides.  ``mF`` is the projection of the hyperfine ``F``
    while ``mJ`` is the projection of the electronic ``j`` — different quantum
    numbers, neither inferred from the other (QN-001A).

    The SIGNED value is preserved so that a phase-sensitive consumer can use
    it; the magnitude ``|d|`` is taken only by the Rabi helper
    (``calculate_hfs_rabi_frequency``), which adds the Gaussian peak field.

    Args:
        atom: ARC atom object.
        from_level: source level; must carry an explicit ``F`` and ``mF``.
        to_level: destination level; must carry an explicit ``F`` and ``mF``.
        q: laser polarisation in {-1, 0, +1} (+1 = sigma+, 0 = pi,
            -1 = sigma-). Required, never defaulted, never scanned.

    Returns:
        float: SIGNED dipole matrix element in ``e·a0`` (never SI units,
        never ``|d|``).

    Raises:
        PhysicsConfigurationError: a missing ``F``/``mF`` on either side, an
            E1-forbidden declared transition, an invalid ``q``, a
            STRUCTURALLY forbidden magnetic channel (``mF2 - mF1`` differs
            from ``q``, checked exactly BEFORE ARC is called), or a genuinely
            dark (exactly zero) declared channel.
        PhysicsCalculationError: ARC raised, or returned ``None`` /
            non-numeric / NaN / Inf. F05: no fallback.
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the HFS-to-HFS dipole matrix element "
            f"{getattr(from_level, 'label', from_level)} -> "
            f"{getattr(to_level, 'label', to_level)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")

    # ---- BOTH sides must be EXPLICIT hyperfine sublevels -------------------
    for role, level in (('source', from_level), ('destination', to_level)):
        if level.f is None or level.mf is None:
            raise PhysicsConfigurationError(
                f"the HFS-to-HFS dipole matrix element for {transition} "
                f"requires an explicit hyperfine {role} sublevel (F, mF); got "
                f"F={level.f}, mF={level.mf}. QN-001A.1: a hyperfine sublevel "
                "is never inferred (and never replaced by mJ)."
            )
    # ---- E1 structural rules on the declared transition --------------------
    delta_l = abs(int(round(float(to_level.l))) - int(round(float(from_level.l))))
    delta_j = abs(float(to_level.j) - float(from_level.j))
    if delta_l != 1:
        raise PhysicsConfigurationError(
            f"the HFS-to-HFS dipole matrix element for {transition} violates "
            f"the electric-dipole selection rule (delta_l = {delta_l} != 1). "
            "QN-001A.1: no fallback is permitted."
        )
    if delta_j > 1.0 + _HFS_HFS_CHANNEL_TOL:
        raise PhysicsConfigurationError(
            f"the HFS-to-HFS dipole matrix element for {transition} violates "
            f"the electric-dipole selection rule (delta_j = {delta_j} > 1). "
            "QN-001A.1: no fallback is permitted."
        )
    if abs(float(from_level.j)) <= _HFS_HFS_CHANNEL_TOL \
            and abs(float(to_level.j)) <= _HFS_HFS_CHANNEL_TOL:
        raise PhysicsConfigurationError(
            f"the HFS-to-HFS dipole matrix element for {transition} violates "
            "the electric-dipole selection rule (j = 0 -> j = 0 is "
            "forbidden). QN-001A.1: no fallback is permitted."
        )

    # ---- polarisation q must be declared and valid -------------------------
    try:
        q_int = int(q)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} is missing or is not an "
            f"integer ({q!r}); q must be one of -1, 0, +1. QN-001A.1: the "
            "polarisation is configured, never hard-coded and never scanned."
        ) from exc
    if float(q) != float(q_int) or q_int not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} must be one of -1, 0, +1 "
            f"(got {q!r}). QN-001A.1: no hard-coded q, no q scan."
        )

    # ---- exact structural magnetic-channel validation (BEFORE ARC) ---------
    _validate_hfs_to_hfs_magnetic_channel(from_level, to_level, q_int)
# ---- designated ARC calculation (the ONE declared channel) -------------
    try:
        dipole_ea0 = atom.getDipoleMatrixElementHFS(
            from_level.n, from_level.l, from_level.j,
            from_level.f, from_level.mf,
            to_level.n, to_level.l, to_level.j,
            to_level.f, to_level.mf,
            q_int,
        )
    except PhysicsError:
        # F05: a Physics* failure is already the correct contextual fail-fast
        # error; never replace it with a fallback value.
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC HFS-to-HFS dipole matrix element failed for {transition} "
            f"(F1={from_level.f}, mF1={from_level.mf} -> "
            f"F2={to_level.f}, mF2={to_level.mf}, q={q_int}): "
            f"{type(exc).__name__}: {exc}. F05: no fallback and no zero "
            "substitution."
        ) from exc

    if dipole_ea0 is None:
        raise PhysicsCalculationError(
            f"ARC returned no HFS-to-HFS dipole matrix element for "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"F2={to_level.f}, mF2={to_level.mf}, q={q_int}). "
            "F05: no fallback is permitted."
        )
    try:
        dipole_val = float(dipole_ea0)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"ARC returned a non-numeric HFS-to-HFS dipole matrix element for "
            f"{transition} ({dipole_ea0!r}). F05: no fallback."
        ) from exc
    if not np.isfinite(dipole_val):
        raise PhysicsCalculationError(
            f"ARC returned a non-finite HFS-to-HFS dipole matrix element for "
            f"{transition} ({dipole_val!r} e·a0). F05: no fallback."
        )

    # An explicitly declared channel may legitimately be dark. That is a
    # CONFIGURATION error here — never a reason to try another mF or another
    # q, and never a value to substitute.
    if dipole_val == 0.0:
        raise PhysicsConfigurationError(
            "The explicitly configured HFS->HFS magnetic channel "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"F2={to_level.f}, mF2={to_level.mf}, q={q_int}) has zero "
            "electric-dipole coupling for the configured polarization. No "
            "alternate mF or q is selected automatically. QN-001A.1: "
            "configure a different explicit channel instead."
        )

    logger.debug(
        "QN-001A.1 signed HFS->HFS element for %s (F=%.1f, mF=%g -> "
        "F=%.1f, mF=%g, q=%d): %.9e e·a0",
        transition, from_level.f, from_level.mf, to_level.f, to_level.mf,
        q_int, dipole_val)
    return dipole_val  # SIGNED, e·a0 — never |d| and never SI