"""ARCH-001T — shared support for tests that need RESOLVED atomic physics.

ARCH-001 changed the runtime contract: importing
``rydberg_aoa.config.atomic_config`` publishes NOTHING; resolved quantities
(``ATOMIC_LEVELS``, ``OMEGA_P``/``OMEGA_C``/``OMEGA_RF_BASE``,
``OMEGA_SIGNAL_RF_BASE``, ``GAMMA_21/32/43``, cached detunings/dipoles and
the derived absorption constants) exist only after the explicit
``initialize_atomic_physics()`` seam succeeded.  Legacy test suites predate
that seam and relied on import-time physics; this module provides the ONE
opt-in mechanism for them.

Contents
--------
* ``ensure_resolved_atomic_physics()`` — for module-scope fixtures and
  direct call sites: attempts the canonical seam (idempotent), and ONLY when
  the exact, documented QN-001A coupling boundary is hit, raises
  ``pytest.skip`` with a ``BLOCKED: QN-001A`` reason.  Any other failure
  propagates unchanged (F05: a real physics/configuration failure must
  never be converted into a skip).
* ``snapshot_published_atomic_state`` / ``restore_published_atomic_state`` —
  the ARCH-001 published-state surface (flag + compatibility globals +
  ``params.OMEGA_S_BASE`` / ``params.OMEGA_LO_BASE``), used by the
  function-scoped ``atomic_physics`` fixture in ``tests/conftest.py`` so a
  test that initializes atomic physics leaves the runtime equivalent to its
  pre-test state.
* ``is_documented_qn001a_coupling_block`` — the strict predicate the skip
  path is gated on (exception type + three exact message substrings).

Importing this module performs NO atomic initialization and no other
physics side effect.
"""

import pytest

from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config import parameters as params
from rydberg_aoa.exceptions import PhysicsConfigurationError

__all__ = [
    "PUBLISHED_ATOMIC_GLOBALS",
    "QN001A_BLOCK_SKIP_REASON",
    "ensure_resolved_atomic_physics",
    "is_documented_qn001a_coupling_block",
    "restore_published_atomic_state",
    "snapshot_published_atomic_state",
]

# Published compatibility state that initialize_atomic_physics() may write
# (same surface as the ARCH-001 module's own isolation helper).
PUBLISHED_ATOMIC_GLOBALS = (
    "ATOMIC_LEVELS",
    "ARC_RF_DIPOLE_EA0",
    "GAMMA_21",
    "GAMMA_32",
    "GAMMA_43",
    "OMEGA_P",
    "OMEGA_C",
    "OMEGA_RF_BASE",
    "OMEGA_SIGNAL_RF_BASE",
    "DELTA_P",
    "DELTA_C",
    "DELTA_RF",
)

QN001A_BLOCK_SKIP_REASON = (
    "BLOCKED: QN-001A — the coupling laser's HFS->FS destination (e.g. "
    "26D5/2) declares no explicit Rydberg mJ, so the ARCH-001 seam stops "
    "before publishing any resolved atomic quantity (no mJ is inferred and "
    "none is scanned; F05). Re-enable automatically once the explicit "
    "Rydberg mJ contract is configured."
)


def is_documented_qn001a_coupling_block(exc: BaseException) -> bool:
    """True ONLY for the exact documented QN-001A HFS->FS coupling boundary.

    The production seam raises ``PhysicsConfigurationError`` whose message
    names the HFS->FS boundary, the missing Rydberg mJ, and the
    no-inference/no-scan contract.  A skip may be produced only for an
    exception satisfying every one of those conditions; anything else
    (wrong type, generic configuration error, calculation error) must
    propagate so a genuine failure is never masked as a skip.
    """
    if not isinstance(exc, PhysicsConfigurationError):
        return False
    message = str(exc)
    return (
        "HFS->FS" in message
        and "no mJ is inferred" in message
        and "does not scan mJ" in message
    )


def snapshot_published_atomic_state() -> dict:
    """Capture the ARCH-001 published-state surface (flag, globals, bases).

    MODE-001: the initialized resolution REQUEST is captured as well, so a
    test that initialized with an explicit ``resolution_request=`` cannot leak
    that mode into later no-argument readers via the stored request.
    NUM-SS-001-R3: the initialized RESOLUTION IDENTITY
    (``atomic_config._INITIALIZED_RESOLUTION_IDENTITY``) is captured too and
    restored to its EXACT entry value (never reset unconditionally to None),
    so driving a fixture that initializes cannot leave a stale identity behind
    after the initialized flag has already returned to False.
    """
    return {
        "flag": acfg._ATOMIC_PHYSICS_INITIALIZED,
        "globals": {name: getattr(acfg, name) for name in PUBLISHED_ATOMIC_GLOBALS},
        "initialized_request": acfg._INITIALIZED_REQUEST,
        "resolution_identity": acfg._INITIALIZED_RESOLUTION_IDENTITY,
        "atom_identity": acfg._INITIALIZED_ATOM_IDENTITY,
        "rf_orientation": acfg._RF_ORIENTATION,
        "omega_s_base": params.OMEGA_S_BASE,
        "omega_lo_base": params.OMEGA_LO_BASE,
    }


def restore_published_atomic_state(state: dict) -> None:
    """Restore a snapshot taken by :func:`snapshot_published_atomic_state`."""
    for name, value in state["globals"].items():
        setattr(acfg, name, value)
    acfg._ATOMIC_PHYSICS_INITIALIZED = state["flag"]
    acfg._INITIALIZED_REQUEST = state["initialized_request"]
    acfg._INITIALIZED_RESOLUTION_IDENTITY = state["resolution_identity"]
    acfg._INITIALIZED_ATOM_IDENTITY = state['atom_identity']
    acfg._RF_ORIENTATION = state["rf_orientation"]
    params.OMEGA_S_BASE = state["omega_s_base"]
    params.OMEGA_LO_BASE = state["omega_lo_base"]


# Process-level memoization of the documented BLOCKED verdict: the QN-001A
# boundary is a stable property of the (unmodified) configuration, so the
# pipeline needs to be attempted at most once per session.  This mirrors the
# established once-per-session pattern of the legacy self-heal helper in
# test_probe_wavelength_cache_regression.py and keeps repeated fixture uses
# from re-running the ARC pipeline.  Only the exact documented block is
# memoized; failures of any other kind are always re-raised, never cached.
_BLOCKED_BY_QN001A = False


def ensure_resolved_atomic_physics():
    """Opt a test/fixture into the ARCH-001 explicit initialization.

    * already initialized -> return immediately (idempotent, cheap);
    * documented QN-001A coupling boundary -> ``pytest.skip`` with the
      explicit ``BLOCKED: QN-001A`` reason (the pipeline is transactional,
      so the runtime is unchanged and nothing needs restoring; the verdict
      is memoized for the session after one real attempt);
    * any other failure -> re-raised unchanged (F05).

    Returns the ``atomic_config`` module object for convenience.
    """
    global _BLOCKED_BY_QN001A

    if acfg.is_atomic_physics_initialized():
        return acfg
    if _BLOCKED_BY_QN001A:
        pytest.skip(QN001A_BLOCK_SKIP_REASON)

    try:
        acfg.initialize_atomic_physics()
    except PhysicsConfigurationError as exc:
        if is_documented_qn001a_coupling_block(exc):
            _BLOCKED_BY_QN001A = True
            pytest.skip(QN001A_BLOCK_SKIP_REASON)
        raise

    return acfg
