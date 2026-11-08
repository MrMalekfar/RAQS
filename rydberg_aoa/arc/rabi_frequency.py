"""
Rabi frequency calculations for ARC Interface.

Contains Rabi frequency calculations for laser and RF transitions using ARC's direct getRabiFrequency method.
Now includes mj summation for Rydberg transitions to ensure non-zero Rabi frequencies.
"""

import logging
import numpy as np
from .types import AtomicLevel, LaserParams, RFParams
from .dipole import (
    calculate_dipole_matrix_element,
    calculate_fs_to_fs_dipole_matrix_element,
    calculate_hfs_to_fs_dipole_matrix_element,
    EA0_TO_CM,
)
from ..exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)

logger = logging.getLogger(__name__)

from rydberg_aoa.config import constants as const


# ---------------------------------------------------------------------------
# QN-001A — hyperfine-resolved primary-path helpers
# ---------------------------------------------------------------------------
def _is_hfs_resolved(level: AtomicLevel) -> bool:
    """True when a level names an EXPLICIT hyperfine sublevel (F and mF)."""
    return level.f is not None and level.mf is not None


def _configured_polarization(laser_params, transition: str) -> int:
    """Return the configured laser polarisation q in {-1, 0, +1} (QN-001A).

    The historical implementation hard-coded ``q = 0`` and ignored
    ``laser_params.polarization`` entirely. The polarisation now comes from the
    configured laser and is validated: q must be an integer in {-1, 0, +1}.

    Scanning q (or mj) and keeping the largest element is explicitly forbidden:
    that would silently change the driven physical transition instead of
    reporting the configured one.
    """
    q_raw = getattr(laser_params, "polarization", None)
    try:
        q = int(q_raw)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} is missing or is not an "
            f"integer ({q_raw!r}); q must be one of -1, 0, +1. QN-001A: the "
            "polarisation comes from the configured laser, never from a "
            "hard-coded default."
        ) from exc
    if float(q_raw) != float(q) or q not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"laser polarization for {transition} must be one of -1, 0, +1 "
            f"(got {q_raw!r}). QN-001A: no hard-coded q fallback."
        )
    return q


def _configured_rf_polarization(rf_params, transition: str) -> int:
    """Return the configured RF polarisation q in {-1, 0, +1} (QN-001C).

    The historical RF path never used a polarisation at all: it delegated to
    the generic max-over-mJ dipole scan, which silently CHOSE the strongest
    magnetic channel and ignored ``rf_params.polarization`` entirely. The
    QN-001C FS -> FS contract computes exactly the DECLARED channel, so q is
    a required input of the configured RF field (dict or ``RFParams``) and is
    never hard-coded, defaulted or scanned.

    Accepts both the dict form (``SIGNAL_RF_PARAMS`` / ``LO_RF_PARAMS``) and
    the ``RFParams`` dataclass.
    """
    q_raw = (rf_params.get('polarization') if isinstance(rf_params, dict)
             else getattr(rf_params, 'polarization', None))
    try:
        q = int(q_raw)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"RF polarization for {transition} is missing or is not an "
            f"integer ({q_raw!r}); q must be one of -1, 0, +1. QN-001C: the "
            "polarisation comes from the configured RF field, never from a "
            "hard-coded default."
        ) from exc
    if float(q_raw) != float(q) or q not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"RF polarization for {transition} must be one of -1, 0, +1 "
            f"(got {q_raw!r}). QN-001C: no hard-coded q fallback."
        )
    return q


def _gaussian_peak_field(power_w: float, waist_m: float) -> float:
    """Peak electric field of the configured TEM00 Gaussian beam [V/m].

    BEAM-002 single-source-of-truth convention: ``waist_m`` is the 1/e²
    intensity RADIUS w0, the on-axis intensity is

        I_peak = 2 P / (pi w0²)                                [W/m²]

    and the peak field follows from the plane-wave intensity relation

        E_peak = sqrt(2 I_peak / (eps0 c))                     [V/m].

    Both optical Rabi paths (HFS -> HFS, QN-001A, and HFS -> FS, QN-001B)
    use exactly this expression — no 2pi factor, no RMS/peak ambiguity.
    """
    i_peak = 2.0 * power_w / (np.pi * waist_m ** 2)
    return np.sqrt(2.0 * i_peak / (const.EPSILON_0 * const.C_LIGHT))


def calculate_rabi_frequency(atom, to_level: AtomicLevel, laser_params: LaserParams, from_level: AtomicLevel = None,
                            ground_state: AtomicLevel = None) -> float:
    """
    Calculate Rabi frequency for laser transition using ARC getRabiFrequency method.

    Designated-primary-path contract (QN-001A + QN-001B):

      * HFS -> HFS transition (both levels carry F *and* mF) — the PRIMARY
        calculation is ARC's hyperfine element
        (``getDipoleMatrixElementHFS``) plus ``Omega = |d| E_peak / hbar``
        (see :func:`calculate_hfs_rabi_frequency`). If it fails it raises; no
        other path is attempted afterwards.
      * HFS -> FS transition (the source carries F and mF, the destination an
        explicit mJ) — the PRIMARY calculation is ARC's dedicated HFS->FS
        element (``getDipoleMatrixElementHFStoFS``) plus the same
        ``Omega = |d| E_peak / hbar`` (see
        :func:`calculate_hfs_to_fs_rabi_frequency`). It computes exactly the
        DECLARED magnetic channel: a destination without an explicit mJ fails
        fast and no mJ is inferred or scanned (QN-001B).
      * otherwise — fine-structure transition — ARC ``getRabiFrequency`` is
        used, and it REQUIRES an explicit ``mj`` on the driving level: no
        mF -> mj inference, no mj scan, no dipole-based substitute.

    The polarisation ``q`` is read from ``laser_params.polarization`` and is
    never hard-coded.

    Args:
        atom: ARC atom object
        to_level: Target level
        laser_params: Laser parameters
        from_level: Starting level (if None, determined automatically)
        ground_state: Ground state level

    Returns:
        Angular Rabi frequency in rad/s (C-001 unit contract: the single
        s->us conversion to rad/us happens in atomic_config; Omega [rad/us]
        = 2pi * f_Rabi [MHz] is a display relation only).
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the laser Rabi frequency for "
            f"{getattr(to_level, 'label', to_level)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    # Determine the starting level
    if from_level is None:
        if ground_state is None:
            raise PhysicsConfigurationError(
                "from_level was not supplied and ground_state is None, so the "
                "laser Rabi frequency for "
                f"{getattr(to_level, 'label', to_level)} cannot be calculated. "
                "Pass from_level explicitly. F05: no fallback is permitted."
            )
        if to_level.n == ground_state.n + 1 and to_level.l == 1:  # Excited state
            from_level = ground_state
        else:
            raise PhysicsConfigurationError(
                "from_level could not be derived for the laser Rabi "
                f"calculation to {getattr(to_level, 'label', to_level)} "
                f"(ground_state={getattr(ground_state, 'label', ground_state)}). "
                "Pass from_level explicitly. F05: no fallback is permitted."
            )

    transition = f"{getattr(from_level, 'label', from_level)} -> {getattr(to_level, 'label', to_level)}"
    power_w = laser_params.power_mw * 1e-3  # Convert mW to W
    # BEAM-002: `beam_waist_mm` is the 1/e2 intensity RADIUS [mm] (derived centrally in laser_rf_config from the Simons FWHM diameter; ARC ``laserWaist`` requires exactly this radius).
    waist_m = laser_params.beam_waist_mm * 1e-3  # 1/e2 radius [m]
    q = _configured_polarization(laser_params, transition)  # QN-001A: from config
    if waist_m <= 0 or power_w <= 0:
        raise PhysicsConfigurationError(
            f"Invalid laser parameters for {transition}: "
            f"power_mw={laser_params.power_mw}, "
            f"beam_waist_mm={laser_params.beam_waist_mm} (both must be > 0). "
            "F05: no fallback is permitted."
        )

    # -------------------------------------------------------------------------
    # QN-001A — designated primary path dispatch
    # -------------------------------------------------------------------------
    if _is_hfs_resolved(from_level) and _is_hfs_resolved(to_level):
        # HFS -> HFS optical transition (the configured probe:
        # 5S1/2 F=3, mF=0 -> 5P3/2 F=4, mF=0). The installed ARC (3.9.0)
        # exposes NO getRabiFrequencyHFS, so the hyperfine Rabi frequency is
        # built from ARC's HFS-resolved dipole matrix element plus the
        # canonical Omega = |d| E_peak / hbar. This is a PRIMARY calculation
        # path, not a fallback: any failure raises and no other path is
        # attempted afterwards.
        return calculate_hfs_rabi_frequency(atom, from_level, to_level,
                                            laser_params)

    if _is_hfs_resolved(from_level):
        # QN-001B: HFS -> FS optical transition. An HFS-resolved source
        # correctly carries mF (never mJ) and must NOT be forced through the
        # generic fine-structure branch below, which requires an explicit
        # source mj. The dedicated helper computes exactly the DECLARED
        # destination mJ channel through ARC's getDipoleMatrixElementHFStoFS;
        # a destination without an explicit mJ (the expected current blocker:
        # 5P3/2 F=4, mF=0 -> 26D5/2) fails fast with the explicit missing-mJ
        # contract instead of being sent to the generic branch.
        return calculate_hfs_to_fs_rabi_frequency(atom, from_level, to_level,
                                                  laser_params)

    # Fine-structure path: the driving level must declare an explicit mJ.
    if from_level.mj is None:
        raise PhysicsConfigurationError(
            f"the driven laser transition {transition} carries no explicit mJ "
            f"(from_level: n={from_level.n}, l={from_level.l}, j={from_level.j}, "
            f"f={from_level.f}, mf={from_level.mf}; to_level: n={to_level.n}, "
            f"l={to_level.l}, j={to_level.j}, f={to_level.f}, mf={to_level.mf}). "
            "A HFS-to-FS (or FS-to-FS) transition requires an explicit Rydberg "
            "mJ / Zeeman-channel contract; QN-001A does not choose one "
            "implicitly and does not scan mJ for a maximum. "
            "F05: no implicit magnetic sublevel."
        )
    if (to_level.mj is not None
            and abs(float(to_level.mj) - (float(from_level.mj) + q)) > 1e-9):
        raise PhysicsConfigurationError(
            f"inconsistent Zeeman channel for {transition}: ARC's "
            f"getRabiFrequency drives mj2 = mj1 + q = "
            f"{float(from_level.mj) + q}, but to_level.mj = {to_level.mj}. "
            "QN-001A: the declared levels must match the driven transition; "
            "refusing to drive a different one. F05: no implicit magnetic "
            "sublevel."
        )

    try:
        # Fine-structure generic call. Installed ARC contract (verified
        # at runtime, C-001):
        #     getRabiFrequency(n1, l1, j1, mj1, n2, l2, j2, q,
        #                      laserPower, laserWaist, s=0.5)
        # There is NO mj2 positional parameter: ARC drives the state
        # mj2 = mj1 + q. Keyword arguments are used so positional drift
        # cannot reintroduce the C-001 argument-order bug.
        omega_rad_s = atom.getRabiFrequency(
            n1=from_level.n, l1=from_level.l, j1=from_level.j,
            mj1=from_level.mj,
            n2=to_level.n, l2=to_level.l, j2=to_level.j,
            q=q,                    # polarization (q); mj2 = mj1 + q
            laserPower=power_w,     # laser power [W]
            laserWaist=waist_m      # laser 1/e^2 waist [m]
        )
        # Installed ARC returns the ANGULAR Rabi frequency in rad/s (NOT Hz
        # and NOT MHz — C-001 unit contract). The value is validated below,
        # after the try/except, so an unusable ARC result cannot escape.

    except PhysicsError:
        # F05: a Physics* failure is already the correct contextual
        # fail-fast error; never replace it with a fallback value.
        raise
    except Exception as exc:
        # F05: the designated laser-Rabi calculation failed. The historical
        # dipole-based substitute is REMOVED: substituting an approximate
        # physics value here silently replaced the model's Rabi frequency.
        # Fail loudly with the full input context instead.
        raise PhysicsCalculationError(
            f"ARC laser Rabi-frequency calculation failed for {transition} "
            f"(n1={from_level.n}, l1={from_level.l}, j1={from_level.j}, "
            f"mj1={from_level.mj}; n2={to_level.n}, l2={to_level.l}, "
            f"j2={to_level.j}; q={q}; laserPower={power_w} W; "
            f"laserWaist={waist_m} m): "
            f"{type(exc).__name__}: {exc}. "
            "F05: no dipole-based fallback and no zero substitution."
        ) from exc

    # F05 validation — the ARC result must itself be usable.
    if omega_rad_s is None:
        raise PhysicsCalculationError(
            f"ARC returned None as the laser Rabi frequency for "
            f"{transition}. F05: no fallback is permitted."
        )
    try:
        omega_val = float(omega_rad_s)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"ARC returned a non-numeric laser Rabi frequency "
            f"({omega_rad_s!r}) for {transition}. F05: no fallback."
        ) from exc
    if not np.isfinite(omega_val):
        raise PhysicsCalculationError(
            f"ARC returned a non-finite laser Rabi frequency "
            f"({omega_val!r}) for {transition}. F05: no fallback."
        )
    if omega_val <= 0:
        raise PhysicsCalculationError(
            f"ARC returned a non-positive laser Rabi frequency "
            f"({omega_val!r} rad/s) for the driven transition {transition} "
            f"(mj1={from_level.mj}, q={q}, mj2=mj1+q). The model explicitly "
            "drives this transition, so zero or negative is a calculation "
            "failure, not a forbidden transition. "
            "F05: no dipole-based fallback and no zero substitution."
        )
    logger.debug(
        f"ARC calculated Rabi frequency for {from_level.label} -> "
        f"{to_level.label}: {omega_val:.4e} rad/s "
        f"({omega_val*1e-6/2/np.pi:.4f} MHz)")
    return omega_val  # Angular Rabi frequency [rad/s]



def calculate_hfs_rabi_frequency(atom, from_level: AtomicLevel,
                                 to_level: AtomicLevel,
                                 laser_params: LaserParams) -> float:
    """PRIMARY ARC calculation for an HFS-resolved optical transition (QN-001A).

    Physics
    -------
    A laser driving ``|1> -> |2>`` with peak electric-field amplitude
    ``E_peak`` produces the angular Rabi frequency

        Omega = |d_12| * E_peak / hbar                         [rad/s]

    where ``d_12`` is the DESIGNATED hyperfine dipole matrix element obtained
    from ARC for the CONFIGURED polarisation q

        d_12 = ARC.getDipoleMatrixElementHFS(
                   n1,l1,j1, F1,mF1, n2,l2,j2, F2,mF2, q)      [e·a0]

    and the Gaussian-beam peak field of the configured power P and 1/e²
    intensity radius w0:

        I_peak = 2 P / (pi w0^2)                               [W/m²]
        E_peak = sqrt(2 I_peak / (eps0 c))                     [V/m]

    Fail-fast contract (F05 / QN-001A): a raised ARC exception, a ``None`` /
    non-finite / exactly-zero dipole, unusable laser parameters or an
    unconfigured polarisation all raise :class:`PhysicsCalculationError` /
    :class:`PhysicsConfigurationError`. There is deliberately NO fallback: no
    generic ``getRabiFrequency`` call, no other mJ / mF / q trial, no cached or
    hard-coded element, no zero.

    Args:
        atom: ARC atom object (``None`` is rejected by the caller).
        from_level: lower level; must carry F and mF.
        to_level: upper level; must carry F and mF.
        laser_params: ``power_mw``, ``beam_waist_mm`` (1/e² radius) and
            ``polarization``.

    Returns:
        Angular Rabi frequency in rad/s (C-001 unit contract: the single
        s -> us conversion happens in ``atomic_config``).
    """
    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")

    if not (_is_hfs_resolved(from_level) and _is_hfs_resolved(to_level)):
        raise PhysicsConfigurationError(
            f"calculate_hfs_rabi_frequency requires F and mF on BOTH levels "
            f"for {transition}: from_level (F={from_level.f}, "
            f"mF={from_level.mf}), to_level (F={to_level.f}, "
            f"mF={to_level.mf}). QN-001A: no implicit hyperfine sublevel."
        )

    q = _configured_polarization(laser_params, transition)
    power_w = laser_params.power_mw * 1e-3        # mW -> W
    waist_m = laser_params.beam_waist_mm * 1e-3   # mm -> m (1/e² radius)
    if waist_m <= 0 or power_w <= 0:
        raise PhysicsConfigurationError(
            f"Invalid laser parameters for {transition}: "
            f"power_mw={laser_params.power_mw}, "
            f"beam_waist_mm={laser_params.beam_waist_mm} (both must be > 0). "
            "F05: no fallback is permitted."
        )

    # ---- designated primary ARC calculation ---------------------------------
    try:
        dipole_ea0 = atom.getDipoleMatrixElementHFS(
            from_level.n, from_level.l, from_level.j,
            from_level.f, from_level.mf,
            to_level.n, to_level.l, to_level.j,
            to_level.f, to_level.mf,
            q
        )
    except PhysicsError:
        # F05: a Physics* failure is already the correct contextual fail-fast
        # error; never replace it with a fallback value.
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC hyperfine dipole matrix element failed for the driven "
            f"transition {transition} (F1={from_level.f}, mF1={from_level.mf} "
            f"-> F2={to_level.f}, mF2={to_level.mf}, q={q}; "
            f"laserPower={power_w} W, laserWaist={waist_m} m): "
            f"{type(exc).__name__}: {exc}. F05/QN-001A: the HFS element is the "
            "designated PRIMARY path — no generic getRabiFrequency call, no "
            "other mJ or q trial, no cached or hard-coded element."
        ) from exc

    if dipole_ea0 is None:
        raise PhysicsCalculationError(
            f"ARC returned no hyperfine dipole matrix element for "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"F2={to_level.f}, mF2={to_level.mf}, q={q}). "
            "F05: no fallback is permitted."
        )
    try:
        dipole_atomic = float(dipole_ea0)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"ARC returned a non-numeric hyperfine dipole matrix element for "
            f"{transition} ({dipole_ea0!r}). F05: no fallback."
        ) from exc
    if not np.isfinite(dipole_atomic):
        raise PhysicsCalculationError(
            f"ARC returned a non-finite hyperfine dipole matrix element for "
            f"{transition} ({dipole_atomic!r} e·a0). F05: no fallback."
        )
    if dipole_atomic == 0.0:
        raise PhysicsCalculationError(
            f"ARC returned exactly zero for the driven hyperfine transition "
            f"{transition} (F1={from_level.f}, mF1={from_level.mf} -> "
            f"F2={to_level.f}, mF2={to_level.mf}, q={q}). The model drives "
            "this transition explicitly, so zero means the element "
            "calculation failed (or the configured sublevel/polarisation does "
            "not couple) — never a value to be replaced by another path. "
            "F05/QN-001A: no zero substitution and no generic retry."
        )

    dipole_si = abs(dipole_atomic) * EA0_TO_CM          # e·a0 -> C·m

    # Gaussian-beam peak field of the configured power and 1/e² radius
    e_peak = _gaussian_peak_field(power_w, waist_m)                      # V/m
    omega_rad_s = dipole_si * e_peak / const.HBAR                        # rad/s

    if not np.isfinite(omega_rad_s) or omega_rad_s <= 0:
        raise PhysicsCalculationError(
            f"Omega = |d| E_peak / hbar for {transition} is not a positive "
            f"finite value ({omega_rad_s!r} rad/s; |d|={dipole_si:.6e} C·m, "
            f"E_peak={e_peak:.6e} V/m, P={power_w} W, w0={waist_m} m). "
            "F05: no fallback."
        )

    logger.debug(
        "QN-001A HFS Rabi for %s (F=%.1f, mF=%g -> F=%.1f, mF=%g, q=%d): "
        "|d|=%.6e e·a0 = %.6e C·m, E_peak=%.6f V/m, "
        "Omega=%.6e rad/s (%.6f rad/us)",
        transition, from_level.f, from_level.mf, to_level.f, to_level.mf, q,
        dipole_atomic, dipole_si, e_peak, omega_rad_s, omega_rad_s * 1e-6)
    return omega_rad_s  # Angular Rabi frequency [rad/s]


def calculate_hfs_to_fs_rabi_frequency(atom, from_level: AtomicLevel,
                                       to_level: AtomicLevel,
                                       laser_params: LaserParams) -> float:
    """PRIMARY ARC calculation for an explicit HFS -> FS optical transition (QN-001B).

    Physics
    -------
    A laser driving ``|F1 mF1> -> |j2 mJ2>`` with polarisation q and peak
    field ``E_peak`` produces the angular Rabi frequency

        Omega = |d| * E_peak / hbar                             [rad/s]

    where ``d`` is the SIGNED dipole matrix element of exactly the DECLARED
    magnetic channel (see :func:`calculate_hfs_to_fs_dipole_matrix_element`)

        d = ARC.getDipoleMatrixElementHFStoFS(
               n1,l1,j1, F1,mF1, n2,l2,j2, mJ2, q)              [e·a0]

    and the Gaussian peak field of the configured power P and 1/e² intensity
    radius w0:

        I_peak = 2 P / (pi w0²)                                 [W/m²]
        E_peak = sqrt(2 I_peak / (eps0 c))                      [V/m]

    QN-001B contract
    ----------------
    This is a PRIMARY calculation path built on ARC's DEDICATED HFS->FS
    element — NOT ARC's ``getRabiFrequency`` (which requires an electronic mj
    on the SOURCE state and cannot represent an HFS-resolved source) and NOT
    the project's generic max-over-mJ dipole helper. It computes exactly ONE
    declared channel: no mJ inference, no mJ scan, no q scan and no maximum
    over magnetic sublevels.

    Fail-fast (F05): a missing source F/mF, a missing destination mJ, an
    ambiguous destination declaration, an absent or invalid q, a structurally
    forbidden magnetic channel (QN-001B.1: implied mJ1/mI projections outside
    their allowed ranges, checked before ARC), unusable laser parameters, a
    dark (exactly zero) declared channel, an ARC failure and a
    non-positive/non-finite Omega all raise the dedicated hierarchy. There is
    deliberately NO fallback and NO secondary calculation attempt.

    Args:
        atom: ARC atom object (``None`` is rejected by the DME primitive).
        from_level: source level; must carry F and mF.
        to_level: destination level; must carry an explicit mJ.
        laser_params: ``power_mw``, ``beam_waist_mm`` (1/e² radius) and
            ``polarization`` (q).

    Returns:
        Angular Rabi frequency in rad/s (C-001 unit contract: the single
        s -> us conversion happens in ``atomic_config``).
    """
    transition = (f"{getattr(from_level, 'label', from_level)} -> "
                  f"{getattr(to_level, 'label', to_level)}")

    if not _is_hfs_resolved(from_level):
        raise PhysicsConfigurationError(
            "calculate_hfs_to_fs_rabi_frequency requires F and mF on the "
            f"SOURCE level for {transition}: from_level (F={from_level.f}, "
            f"mF={from_level.mf}). QN-001B: no implicit hyperfine sublevel."
        )

    if to_level.mj is None:
        # The EXPECTED current blocker: the Experimental optimizer output
        # (26D5/2) declares no mJ. An explicit destination FS magnetic
        # sublevel (Rydberg mJ / Zeeman-channel contract) is REQUIRED; QN-001B
        # computes only explicitly declared channels.
        raise PhysicsConfigurationError(
            f"the driven laser transition {transition} couples a hyperfine "
            "source to a fine-structure destination (HFS->FS) but the "
            "destination declares no mJ (to_level: "
            f"n={to_level.n}, l={to_level.l}, j={to_level.j}, "
            f"f={to_level.f}, mf={to_level.mf}, mj={to_level.mj}). An "
            "explicit Rydberg mJ / Zeeman-channel contract is required: no "
            "mJ is inferred and the code does not scan mJ for a maximum. "
            "F05: no implicit magnetic sublevel."
        )

    q = _configured_polarization(laser_params, transition)
    power_w = laser_params.power_mw * 1e-3        # mW -> W
    waist_m = laser_params.beam_waist_mm * 1e-3   # mm -> m (1/e² radius)
    if waist_m <= 0 or power_w <= 0:
        raise PhysicsConfigurationError(
            f"Invalid laser parameters for {transition}: "
            f"power_mw={laser_params.power_mw}, "
            f"beam_waist_mm={laser_params.beam_waist_mm} (both must be > 0). "
            "F05: no fallback is permitted."
        )

    # ---- designated primary ARC calculation (exactly ONE channel) ----------
    # The signed element is obtained by the dedicated HFS->FS primitive; that
    # primitive is the ONLY calculation path attempted (no generic
    # getRabiFrequency retry, no max-over-mJ dipole helper).
    dipole_ea0 = calculate_hfs_to_fs_dipole_matrix_element(
        atom, from_level, to_level, q)
    dipole_si = abs(dipole_ea0) * EA0_TO_CM      # |signed e·a0| -> C·m

    e_peak = _gaussian_peak_field(power_w, waist_m)                      # V/m
    omega_rad_s = dipole_si * e_peak / const.HBAR                        # rad/s

    if not np.isfinite(omega_rad_s) or omega_rad_s <= 0:
        raise PhysicsCalculationError(
            f"Omega = |d| E_peak / hbar for {transition} is not a positive "
            f"finite value ({omega_rad_s!r} rad/s; |d|={dipole_si:.6e} C·m, "
            f"E_peak={e_peak:.6e} V/m, P={power_w} W, w0={waist_m} m, "
            f"q={q}). F05: no fallback."
        )

    logger.debug(
        "QN-001B HFS->FS Rabi for %s (F=%.1f, mF=%g -> mJ=%g, q=%d): "
        "d=%.6e e·a0 = %.6e C·m, E_peak=%.6f V/m, "
        "Omega=%.6e rad/s (%.6f rad/us)",
        transition, from_level.f, from_level.mf, to_level.mj, q,
        dipole_ea0, dipole_si, e_peak, omega_rad_s, omega_rad_s * 1e-6)
    return omega_rad_s  # Angular Rabi frequency [rad/s]


def calculate_rf_rabi_frequency(atom, level1, level2, rf_params) -> float:
    """
    Calculate the RF (Rydberg -> Rydberg) Rabi frequency for ONE declared
    magnetic channel (QN-001C).

    Contract (QN-001C)
    ------------------
    The RF transition is an explicit FS -> FS magnetic channel
    ``|n1 l1 j1 mJ1> -> |n2 l2 j2 mJ2>`` with the configured RF polarisation
    q = ``rf_params.polarization``. Its dipole comes from the dedicated
    primitive :func:`calculate_fs_to_fs_dipole_matrix_element` (exactly ONE
    ARC ``getDipoleMatrixElement`` call), NOT from the generic
    max-over-mJ helper that silently selected the strongest channel and
    ignored the polarisation. Both levels must carry an EXPLICIT mJ; no mJ
    and no q is inferred, defaulted or scanned. Then (C-001):

        Omega = |d| * E_RF / hbar                                [rad/s]

    Args:
        atom: ARC atom object
        level1, level2: Atomic levels (dict or AtomicLevel objects); both
            must carry an explicit ``mj`` and no hyperfine labels
        rf_params: RF parameters (dict or RFParams object); must carry
            ``electric_field_vm`` and ``polarization``

    Returns:
        Angular Rabi frequency in rad/s (C-001 unit contract: the single
        s->us conversion to rad/us happens in atomic_config; Omega [rad/us]
        = 2pi * f_Rabi [MHz] is a display relation only).
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the RF Rabi frequency for "
            f"{getattr(level2, 'label', level2)} cannot be calculated. "
            "F05: no fallback is permitted."
        )

    # Convert dictionary parameters to AtomicLevel and RFParams objects if needed
    level1_obj = AtomicLevel(**level1) if isinstance(level1, dict) else level1
    level2_obj = AtomicLevel(**level2) if isinstance(level2, dict) else level2

    transition = (f"{getattr(level1_obj, 'label', level1_obj)} -> "
                  f"{getattr(level2_obj, 'label', level2_obj)}")
    e_field = (rf_params['electric_field_vm'] if isinstance(rf_params, dict)
               else rf_params.electric_field_vm)
    if e_field is None:
        raise PhysicsConfigurationError(
            f"RF electric field is None for {transition}. "
            "F05: no fallback is permitted."
        )

    # QN-001C: the polarisation is a REQUIRED input of the RF configuration.
    q = _configured_rf_polarization(rf_params, transition)

    # ---- designated primary ARC calculation (the ONE declared channel) ----
    # The signed element is obtained by the dedicated FS->FS primitive; that
    # primitive is the ONLY calculation path attempted (no generic
    # max-over-mJ dipole helper, no q scan, no mJ scan).
    dipole_ea0 = calculate_fs_to_fs_dipole_matrix_element(
        atom, level1_obj, level2_obj, q)
    # ---- F05 validation of the primitive's SIGNED element [e·a0] -----------
    if dipole_ea0 is None:
        raise PhysicsCalculationError(
            f"Dipole returned for the RF transition {transition} is None. "
            "F05: no fallback is permitted."
        )
    try:
        dipole_ea0 = float(dipole_ea0)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError(
            f"Dipole returned for the RF transition {transition} is not "
            f"numeric ({dipole_ea0!r}). F05: no fallback."
        ) from exc
    if not np.isfinite(dipole_ea0):
        raise PhysicsCalculationError(
            f"Dipole returned for the RF transition {transition} is not "
            f"finite ({dipole_ea0!r}). F05: no fallback."
        )
    dipole = abs(dipole_ea0) * EA0_TO_CM      # |signed e·a0| -> C·m
    if dipole == 0.0:
        raise PhysicsCalculationError(
            f"ARC returned a zero dipole moment for the driven RF transition "
            f"{transition} (mJ1={level1_obj.mj} -> mJ2={level2_obj.mj}, "
            f"q={q}). The model explicitly drives this transition, so zero "
            "means the dipole calculation failed - never a value to "
            "substitute. F05: no zero substitution."
        )

    # Canonical definition (C-001): Omega = mu * E / hbar [rad/s].
    # Exactly ONE s->us conversion happens downstream in atomic_config;
    # there is NO 2pi in this chain.
    omega_rabi = dipole * e_field / const.HBAR

    if not np.isfinite(omega_rabi) or omega_rabi <= 0:
        raise PhysicsCalculationError(
            f"RF Rabi frequency for {transition} is not a positive finite "
            f"value ({omega_rabi!r} rad/s) although dipole="
            f"{dipole:.3e} C*m and E_field={e_field} V/m "
            f"(mJ1={level1_obj.mj}, mJ2={level2_obj.mj}, q={q}). "
            "F05: no zero substitution."
        )

    logger.debug(
        "QN-001C RF Rabi for %s (mJ1=%g -> mJ2=%g, q=%d): d=%.6e e·a0 = "
        "%.6e C·m, E=%.6f V/m, Omega=%.6e rad/s (%.6f rad/us)",
        transition, level1_obj.mj, level2_obj.mj, q,
        dipole_ea0, dipole, e_field, omega_rabi, omega_rabi * 1e-6)
    return omega_rabi  # Angular Rabi frequency [rad/s]
