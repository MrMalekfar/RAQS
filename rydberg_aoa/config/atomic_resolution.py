"""MODE-001 — atomic-resolution mode boundary: PRESCRIBED vs EXPERIMENTAL.

Why this module exists
----------------------
Until MODE-001 the four-manifold Rydberg ladder (G, E, R1, R2) could only be
resolved one way: :class:`SingleStageLevelOptimizer` searched ARC for the
levels that best matched the configured laser/RF parameters.  That automatic
path is a reasonable *experimental* workflow, but it cannot express a
*measurement* in which the operator already knows which four states and which
magnetic channels are being driven, and in which the applied wavelengths /
frequencies are legitimate inputs that must survive untouched even when they
are far from resonance.

MODE-001 therefore splits atomic resolution into two explicit modes:

PRESCRIBED
    The user declares all four states (including the Rydberg ``mJ``) and the
    field parameters.  Nothing is searched, ranked, scanned, inferred or
    repaired; the declared channel is validated and then used.

EXPERIMENTAL
    The historical automatic resolution.  Its physics and search algorithm are
    deliberately NOT redesigned here (that is the separate AUTO-001 task); it
    is isolated behind this mode so it can later be replaced safely.

Design constraints honoured by this module
------------------------------------------
* One canonical, user-facing switch — :data:`ATOMIC_RESOLUTION_MODE` plus
  :data:`PRESCRIBED_LADDER` — instead of a boolean scattered through the
  configuration layer (``parameters.py`` is intentionally NOT touched; it
  holds unrelated user edits).
* Minimal dependencies: only :mod:`rydberg_aoa.arc.types` (the ``AtomicLevel``
  declaration) and :mod:`rydberg_aoa.exceptions`.  No import of
  ``atomic_config`` or ``laser_rf_config``, so no circular import is possible
  and this module can be imported by either.
* Ambiguous configuration fails loudly (F05): an EXPERIMENTAL request that
  carries a ladder spec, or a PRESCRIBED request without one, is a
  configuration error — never a silent mode switch.
* The EXPERIMENTAL default is preserved: ``initialize_atomic_physics()`` with
  no arguments keeps behaving exactly as before.
* No Zeeman machinery (no B field, no quantization axis, no g factors): the
  scalar ``q`` in ``{-1, 0, +1}`` polarisation contract is unchanged
  (ZEE-001 is a later task).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Optional, Tuple

from ..arc.types import AtomicLevel
from ..exceptions import PhysicsConfigurationError

__all__ = [
    "ATOMIC_RESOLUTION_MODE",
    "DEFAULT_ATOMIC_RESOLUTION_MODE",
    "PRESCRIBED_LADDER",
    "AtomicResolutionMode",
    "AtomicResolutionRequest",
    "PrescribedLadderSpec",
    "describe_resolution_request",
    "get_configured_resolution_request",
    "prescribed_cache_identity",
    "resolution_request_identity",
    "resolve_resolution_request",
    "set_prescribed_ladder",
    "set_resolution_mode",
    "use_experimental_mode",
]

# Exact projection arithmetic on integers / half-integers (no physics
# tolerance: the E1 rules below are exact statements, not measurements).
_E1_TOL = 1e-9

_L_SYMBOLS = ("S", "P", "D", "F", "G", "H", "K")


class AtomicResolutionMode(str, Enum):
    """How the four-manifold Rydberg ladder is resolved (MODE-001).

    A dedicated enum is used instead of a boolean such as ``AUTO_LEVELS``
    because the two modes have genuinely different contracts: PRESCRIBED
    *declares* the states, EXPERIMENTAL *searches* for them.
    """

    PRESCRIBED = "prescribed"
    EXPERIMENTAL = "experimental"


def _coerce_mode(mode) -> AtomicResolutionMode:
    """Accept the enum or its exact string value; reject everything else."""
    if isinstance(mode, AtomicResolutionMode):
        return mode
    if isinstance(mode, str):
        try:
            return AtomicResolutionMode(mode.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(repr(m.value) for m in AtomicResolutionMode)
            raise PhysicsConfigurationError(
                f"unknown atomic-resolution mode {mode!r}; expected one of "
                f"{allowed}. MODE-001: the mode is never guessed."
            ) from exc
    raise PhysicsConfigurationError(
        f"atomic-resolution mode must be an AtomicResolutionMode (or its "
        f"string value), got {mode!r} ({type(mode).__name__}). "
        "MODE-001: the mode is never guessed."
    )


def _default_level_label(level: AtomicLevel) -> str:
    """Canonical ``<n><L><j>`` label (same style the ARC summary prints)."""
    l_value = int(round(float(level.l)))
    symbol = (_L_SYMBOLS[l_value] if 0 <= l_value < len(_L_SYMBOLS)
              else f"l{l_value}")
    return f"{int(round(float(level.n)))}{symbol}{level.j}"


def _coerce_level(level, role: str) -> AtomicLevel:
    """Accept an ``AtomicLevel`` or a plain mapping of its fields."""
    if isinstance(level, AtomicLevel):
        return level
    if isinstance(level, dict):
        try:
            return AtomicLevel(**level)
        except TypeError as exc:
            raise PhysicsConfigurationError(
                f"the prescribed {role} level mapping {level!r} does not match "
                f"the AtomicLevel declaration ({type(exc).__name__}: {exc}). "
                "MODE-001: no field is defaulted."
            ) from exc
    raise PhysicsConfigurationError(
        f"the prescribed {role} level must be an AtomicLevel instance or a "
        f"mapping of its fields, got {level!r} ({type(level).__name__}). "
        "MODE-001: the declared states are never inferred."
    )


def _labelled(level: AtomicLevel) -> AtomicLevel:
    """Return the level with a canonical label — a COPY, never the input.

    The caller's object is left untouched (PRESCRIBED mode must not mutate the
    user's declaration); only a level with an empty label is copied so that
    error messages and the ARC summary can name it.
    """
    if level.label:
        return level
    return replace(level, label=_default_level_label(level))


def _require_orbital_quantum_numbers(level: AtomicLevel, role: str) -> None:
    """PRESCRIBED structural check: ``n`` and ``l`` must be physical integers."""
    for name, value in (("n", level.n), ("l", level.l)):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PhysicsConfigurationError(
                f"the prescribed {role} level declares a non-numeric {name}="
                f"{value!r}. MODE-001: no default quantum number."
            ) from exc
        if numeric != round(numeric):
            raise PhysicsConfigurationError(
                f"the prescribed {role} level declares a non-integer {name}="
                f"{value!r}. MODE-001: no default quantum number."
            )
    if int(round(float(level.n))) < 1:
        raise PhysicsConfigurationError(
            f"the prescribed {role} level declares n={level.n}; the principal "
            "quantum number must be >= 1. MODE-001: no repair of a declared "
            "state."
        )
    if int(round(float(level.l))) < 0:
        raise PhysicsConfigurationError(
            f"the prescribed {role} level declares l={level.l}; the orbital "
            "quantum number must be >= 0. MODE-001: no repair of a declared "
            "state."
        )
    if int(round(float(level.l))) > int(round(float(level.n))) - 1:
        raise PhysicsConfigurationError(
            f"the prescribed {role} level declares l={level.l} with n="
            f"{level.n}; an orbital angular momentum requires l <= n - 1. "
            "MODE-001: no repair of a declared state."
        )


def _require_e1_pair(lower: AtomicLevel, upper: AtomicLevel,
                     transition: str) -> None:
    """Structural electric-dipole rule for a declared transition.

    Exact statements, never a numerical threshold: ``delta_l = +-1``,
    ``delta_j <= 1`` and ``j = 0 -> j = 0`` forbidden.  A declared transition
    that violates them cannot be driven by an E1 field at any detuning, so it
    is rejected before any ARC evaluation — and never replaced.
    """
    delta_l = abs(int(round(float(upper.l))) - int(round(float(lower.l))))
    delta_j = abs(float(upper.j) - float(lower.j))
    if delta_l != 1:
        raise PhysicsConfigurationError(
            f"the prescribed {transition} transition violates the "
            f"electric-dipole selection rule (delta_l = {delta_l} != 1) for "
            f"{lower.label} -> {upper.label}. MODE-001: a declared state is "
            "never replaced by a search."
        )
    if delta_j > 1.0 + _E1_TOL:
        raise PhysicsConfigurationError(
            f"the prescribed {transition} transition violates the "
            f"electric-dipole selection rule (delta_j = {delta_j} > 1) for "
            f"{lower.label} -> {upper.label}. MODE-001: a declared state is "
            "never replaced by a search."
        )
    if abs(float(lower.j)) <= _E1_TOL and abs(float(upper.j)) <= _E1_TOL:
        raise PhysicsConfigurationError(
            f"the prescribed {transition} transition violates the "
            "electric-dipole selection rule (j = 0 -> j = 0 is forbidden) for "
            f"{lower.label} -> {upper.label}. MODE-001: a declared state is "
            "never replaced by a search."
        )
@dataclass(frozen=True)
class PrescribedLadderSpec:
    """The four physical manifolds of this project, DECLARED by the user.

    This is deliberately NOT a generic N-state framework: the sensor remains a
    four-manifold ladder ``G -> E -> R1 -> R2`` (probe, coupling, LO/signal
    RF).  Each field is an :class:`AtomicLevel`; a mapping of its fields is
    accepted and converted, and the caller's objects are never mutated.

    Contract enforced at construction (before any ARC call):

    ``ground`` / ``excited``
        HFS-resolved: ``F`` and ``mF`` are both required (QN-001A separates
        ``mF`` from ``mJ``; neither is ever inferred from the other and
        ``mJ = 0`` is not a default).
    ``rydberg1`` / ``rydberg2``
        Fine-structure: an EXPLICIT ``mJ`` is required on both, and hyperfine
        quantum numbers must NOT be declared alongside it (the ambiguity
        QN-001C refuses).
    E1 structure
        ``G -> E``, ``E -> R1`` and ``R1 -> R2`` must each satisfy the exact
        electric-dipole rules.

    No energy ordering is imposed: the project has a known, deliberately
    unresolved ladder-index vs energy-order question (the current R1/R2 pair
    may be a downward-energy RF transition), and MODE-001 must not silently
    impose a Hamiltonian or decay direction.
    """

    ground: AtomicLevel
    excited: AtomicLevel
    rydberg1: AtomicLevel
    rydberg2: AtomicLevel

    def __post_init__(self):
        for role in ("ground", "excited", "rydberg1", "rydberg2"):
            object.__setattr__(
                self, role, _labelled(_coerce_level(getattr(self, role), role)))

        for role in ("ground", "excited", "rydberg1", "rydberg2"):
            _require_orbital_quantum_numbers(getattr(self, role), role)

        # -- HFS-resolved probe pair (QN-001A) ---------------------------------
        for role in ("ground", "excited"):
            level = getattr(self, role)
            if level.f is None or level.mf is None:
                raise PhysicsConfigurationError(
                    f"the prescribed {role} level {level.label} must be "
                    f"HFS-resolved: F and mF are both required (got F="
                    f"{level.f}, mF={level.mf}). MODE-001: mF is never inferred "
                    "from mJ and mJ is never inferred from mF (QN-001A)."
                )

        # -- explicitly fine-structure Rydberg pair (QN-001B/C) ----------------
        for role in ("rydberg1", "rydberg2"):
            level = getattr(self, role)
            if level.mj is None:
                raise PhysicsConfigurationError(
                    f"the prescribed {role} level {level.label} declares no "
                    f"explicit mJ (F={level.f}, mF={level.mf}). MODE-001: the "
                    "Rydberg magnetic sublevel is a REQUIRED user input; it is "
                    "never inferred and never scanned (QN-001A/QN-001C)."
                )
            if level.f is not None or level.mf is not None:
                raise PhysicsConfigurationError(
                    f"the prescribed {role} level {level.label} declares an "
                    f"explicit mJ={level.mj} AND hyperfine quantum numbers "
                    f"(F={level.f}, mF={level.mf}); the RF/coupling channel "
                    "couples unresolved-HFS fine-structure states and must not "
                    "declare both. MODE-001: refusing the ambiguous "
                    "declaration instead of choosing an interpretation."
                )

        # -- structural E1 rules for the three driven transitions --------------
        _require_e1_pair(self.ground, self.excited, "probe (ground -> excited)")
        _require_e1_pair(self.excited, self.rydberg1,
                         "coupling (excited -> rydberg1)")
        _require_e1_pair(self.rydberg1, self.rydberg2,
                         "RF (rydberg1 -> rydberg2)")

    def levels(self) -> Dict[str, AtomicLevel]:
        """The four levels in the project's canonical role order."""
        return {
            "ground": self.ground,
            "excited": self.excited,
            "rydberg1": self.rydberg1,
            "rydberg2": self.rydberg2,
        }
@dataclass(frozen=True)
class AtomicResolutionRequest:
    """An explicit, self-validating atomic-resolution request (MODE-001).

    Exactly two consistent shapes exist:

        AtomicResolutionRequest(mode=EXPERIMENTAL)                    # no spec
        AtomicResolutionRequest(mode=PRESCRIBED, prescribed=<spec>)   # spec

    Any other combination is a configuration error: a request that both asks
    for the automatic search and supplies declared states (or asks for
    prescribed resolution without states) is ambiguous, and MODE-001 never
    resolves an ambiguity by silently preferring one branch.
    """

    mode: AtomicResolutionMode = AtomicResolutionMode.EXPERIMENTAL
    prescribed: Optional[PrescribedLadderSpec] = None

    def __post_init__(self):
        mode = _coerce_mode(self.mode)
        object.__setattr__(self, "mode", mode)

        if mode is AtomicResolutionMode.EXPERIMENTAL:
            if self.prescribed is not None:
                raise PhysicsConfigurationError(
                    "an EXPERIMENTAL atomic-resolution request must not carry a "
                    "prescribed ladder spec: the two branches resolve the "
                    "levels differently and the request would be ambiguous. "
                    "MODE-001: no silent branch preference."
                )
            return

        if self.prescribed is None:
            raise PhysicsConfigurationError(
                "a PRESCRIBED atomic-resolution request requires the four "
                "declared states (PrescribedLadderSpec); the mode cannot "
                "resolve anything without them and MODE-001 never falls back "
                "to the automatic search."
            )
        if not isinstance(self.prescribed, PrescribedLadderSpec):
            raise PhysicsConfigurationError(
                "request.prescribed must be a PrescribedLadderSpec, got "
                f"{type(self.prescribed).__name__}. MODE-001: the declared "
                "states are never coerced into a different structure."
            )

    @property
    def is_prescribed(self) -> bool:
        return self.mode is AtomicResolutionMode.PRESCRIBED


# ---------------------------------------------------------------------------
# THE canonical user-facing switch.
#
# This module is the ONE place where the resolution mode is selected.  Nothing
# in ``parameters.py`` / ``atomic_config.py`` / ``main.py`` / ``runner.py``
# carries a second mode flag.  Edit ``ATOMIC_RESOLUTION_MODE`` (and supply
# ``PRESCRIBED_LADDER`` when switching to PRESCRIBED), or call
# ``set_prescribed_ladder(...)`` / ``use_experimental_mode()`` at runtime.
#
# The default is EXPERIMENTAL on purpose: existing callers of
# ``initialize_atomic_physics()`` must keep the current behaviour until the
# user deliberately opts in (MODE-001 backward-compatibility requirement).
# ---------------------------------------------------------------------------
DEFAULT_ATOMIC_RESOLUTION_MODE = AtomicResolutionMode.EXPERIMENTAL

ATOMIC_RESOLUTION_MODE = DEFAULT_ATOMIC_RESOLUTION_MODE

PRESCRIBED_LADDER: Optional[PrescribedLadderSpec] = None


def get_configured_resolution_request() -> AtomicResolutionRequest:
    """Build the request from the configured switch (never a hidden default).

    Raises:
        PhysicsConfigurationError: if the switch is inconsistent with the
            supplied ladder (e.g. PRESCRIBED with ``PRESCRIBED_LADDER = None``).
    """
    return AtomicResolutionRequest(
        mode=ATOMIC_RESOLUTION_MODE, prescribed=PRESCRIBED_LADDER)


def set_resolution_mode(mode, prescribed=None) -> None:
    """Set the configured switch, validating the resulting request first.

    The configuration is only mutated AFTER the request validated, so a bad
    combination cannot leave the process in a half-switched state.
    """
    global ATOMIC_RESOLUTION_MODE, PRESCRIBED_LADDER
    request = AtomicResolutionRequest(mode=mode, prescribed=prescribed)
    ATOMIC_RESOLUTION_MODE = request.mode
    PRESCRIBED_LADDER = request.prescribed


def set_prescribed_ladder(spec: PrescribedLadderSpec) -> None:
    """Switch to PRESCRIBED mode with the given declared ladder."""
    set_resolution_mode(AtomicResolutionMode.PRESCRIBED, spec)


def use_experimental_mode() -> None:
    """Switch back to the EXPERIMENTAL (legacy automatic) mode."""
    set_resolution_mode(AtomicResolutionMode.EXPERIMENTAL, None)
def resolve_resolution_request(resolution_request=None) -> AtomicResolutionRequest:
    """THE canonical resolution entry point used by ``initialize_atomic_physics``.

    An explicit ``resolution_request`` always wins; when it is omitted the
    configured switch is consulted, so ``initialize_atomic_physics()`` with no
    arguments keeps the historical EXPERIMENTAL behaviour.

    Args:
        resolution_request: an :class:`AtomicResolutionRequest`, a mode (enum
            or its string value), or ``None`` for "use the configured switch".

    Returns:
        A validated :class:`AtomicResolutionRequest`.

    Raises:
        PhysicsConfigurationError: for an unknown mode, a bare PRESCRIBED mode
            without a declared ladder, or any other ambiguous configuration.
    """
    if resolution_request is None:
        return get_configured_resolution_request()
    if isinstance(resolution_request, AtomicResolutionRequest):
        return resolution_request
    if isinstance(resolution_request, (str, AtomicResolutionMode)):
        # A bare mode string is accepted for ergonomics; PRESCRIBED without a
        # ladder fails inside AtomicResolutionRequest (never silently searched).
        return AtomicResolutionRequest(mode=resolution_request, prescribed=None)
    raise PhysicsConfigurationError(
        "resolution_request must be an AtomicResolutionRequest, an "
        "AtomicResolutionMode (or its string value), or None; got "
        f"{resolution_request!r} ({type(resolution_request).__name__}). "
        "MODE-001: the mode is never guessed."
    )


def _level_identity(level: AtomicLevel) -> Tuple:
    """Hashable identity of ONE declared level (quantum numbers only).

    ``energy`` and ``label`` are deliberately excluded: they are cosmetic /
    derived annotations and must not make two identical declarations look
    different.
    """
    def _num(value):
        return None if value is None else round(float(value), 9)

    return (int(round(float(level.n))), int(round(float(level.l))),
            _num(level.j), _num(level.f), _num(level.mf), _num(level.mj))


def resolution_request_identity(request: AtomicResolutionRequest) -> Tuple:
    """Hashable identity of a request: the mode plus the four declared states.

    Used for idempotency (same request -> no recomputation; a materially
    different request -> refuse-and-rebuild) and as the leading component of
    the prescribed cache identity.
    """
    if not isinstance(request, AtomicResolutionRequest):
        raise PhysicsConfigurationError(
            "resolution_request_identity requires an AtomicResolutionRequest, "
            f"got {type(request).__name__}. MODE-001."
        )
    if request.mode is AtomicResolutionMode.EXPERIMENTAL:
        return ("experimental",)
    spec = request.prescribed
    return ("prescribed",
            _level_identity(spec.ground), _level_identity(spec.excited),
            _level_identity(spec.rydberg1), _level_identity(spec.rydberg2))


def _polarisation_identity(value, name: str) -> int:
    """Validate and hash a configured polarisation q in {-1, 0, +1}."""
    try:
        q = int(value)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"the configured {name} polarisation is missing or is not an "
            f"integer ({value!r}); q must be one of -1, 0, +1. MODE-001: the "
            "polarisation is a required part of the resolution identity."
        ) from exc
    if float(value) != float(q) or q not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"the configured {name} polarisation must be one of -1, 0, +1 "
            f"(got {value!r}). MODE-001: no q fallback."
        )
    return q


def prescribed_cache_identity(request: AtomicResolutionRequest,
                              probe_params: Dict,
                              coupling_params: Dict,
                              lo_rf_params: Dict,
                              signal_rf_params: Dict) -> Tuple:
    """Complete cache identity of a PRESCRIBED resolution.

    Two prescribed resolutions may only share a cache entry when they agree on
    EVERY input that can change the resolved physics: the declared states
    (including both Rydberg ``mJ`` values), the probe/coupling wavelengths and
    polarisations, and the LO/signal RF frequency, field and polarisation.
    Otherwise switching a prescribed magnetic channel could return stale data.

    The identity is APPENDED to the existing five-field parameter key, so a
    prescribed entry can never collide with an experimental one (tuple arity
    differs) and the experimental key stays bit-identical.
    """
    if not isinstance(request, AtomicResolutionRequest) or not request.is_prescribed:
        raise PhysicsConfigurationError(
            "prescribed_cache_identity requires a PRESCRIBED "
            "AtomicResolutionRequest. MODE-001."
        )

    def _field(params, key, name, places):
        try:
            value = params[key]
        except (KeyError, TypeError) as exc:
            raise PhysicsConfigurationError(
                f"the configured {name} '{key}' is missing from {params!r}. "
                "MODE-001: the prescribed cache identity requires every field "
                "parameter; no default is substituted."
            ) from exc
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PhysicsConfigurationError(
                f"the configured {name} '{key}' is not numeric ({value!r}). "
                "MODE-001: no default is substituted."
            ) from exc
        return round(numeric, places)

    return resolution_request_identity(request) + (
        _field(probe_params, 'wavelength_nm', 'probe wavelength', 6),
        _polarisation_identity(probe_params.get('polarization'), 'probe'),
        _field(coupling_params, 'wavelength_nm', 'coupling wavelength', 6),
        _polarisation_identity(coupling_params.get('polarization'), 'coupling'),
        _field(lo_rf_params, 'frequency_hz', 'LO RF frequency', 3),
        _field(lo_rf_params, 'electric_field_vm', 'LO RF field', 9),
        _polarisation_identity(lo_rf_params.get('polarization'), 'LO RF'),
        _field(signal_rf_params, 'frequency_hz', 'signal RF frequency', 3),
        _field(signal_rf_params, 'electric_field_vm', 'signal RF field', 9),
        _polarisation_identity(signal_rf_params.get('polarization'),
                               'signal RF'),
    )


def describe_resolution_request(request: AtomicResolutionRequest) -> str:
    """One-line human-readable summary (reporting / ARC summary printing)."""
    if request.mode is AtomicResolutionMode.EXPERIMENTAL:
        return ("EXPERIMENTAL atomic resolution — the automatic ARC level "
                "optimizer selects the four levels")
    lines = ["PRESCRIBED atomic resolution — the four declared states are used "
             "verbatim (no level search, no mJ scan, no q scan)"]
    for role, level in request.prescribed.levels().items():
        lines.append(
            f"  {role:<9} {level.label:<8} n={level.n}, l={level.l}, j={level.j}"
            f", F={level.f}, mF={level.mf}, mJ={level.mj}")
    return "\n".join(lines)