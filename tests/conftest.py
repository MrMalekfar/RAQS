"""ARCH-001T shared fixtures — explicit opt-in to RESOLVED atomic physics.

Importing this module (or ``tests.atomic_init_support``) performs NO atomic
initialization and no other physics side effect: ARCH-001 requires that
``import != initialize`` for every test module.  There is deliberately NO
autouse fixture here — tests request exactly the state they need.

Fixtures
--------
``atomic_physics``
    The canonical explicit opt-in (function-scoped).  Runs the production
    seam once (idempotent), skips with an explicit ``BLOCKED: QN-001A``
    reason at the documented coupling boundary, and restores the ARCH-001
    published-state surface afterwards so the test leaves the runtime
    equivalent to its pre-test state.

``published_atomic_state``
    A minimal, controlled "initialized" state for CONSUMER unit tests that
    exercise guarded readers with their own synthetic inputs (e.g. the
    RF-003 detuning-cache contract).  It publishes only the seam flag and
    uses ``monkeypatch`` so restoration is automatic.  It must never be
    used to fake the initialization PROCESS itself.
"""

import pytest

from rydberg_aoa.config import atomic_config as acfg
from tests.atomic_init_support import (
    ensure_resolved_atomic_physics,
    restore_published_atomic_state,
    snapshot_published_atomic_state,
)


@pytest.fixture
def atomic_physics():
    """Explicitly resolve atomic physics, or skip with the QN-001A reason.

    On success the ARCH-001 published state is restored after the test, so
    no resolved value can leak into later tests regardless of execution
    order.  At the documented QN-001A boundary the seam is transactional
    (nothing published), so the skip path leaves the runtime untouched.
    """
    state = snapshot_published_atomic_state()
    ensure_resolved_atomic_physics()   # pytest.skip propagates on the block
    yield acfg
    restore_published_atomic_state(state)


@pytest.fixture
def published_atomic_state(monkeypatch):
    """Minimal known "initialized" state for guarded-reader contract tests.

    Flips ONLY the seam flag (auto-restored by monkeypatch); no physics
    values are published, so consumers keep whatever synthetic inputs the
    test itself supplies.

    MODE-001: the stored initialized REQUEST is neutralized as well, so a
    leftover request from an earlier real initialization cannot steer these
    no-argument readers away from the test's own synthetic cache entries.
    RF-DIR-001B: the published RF-edge orientation is neutralized for the
    same reason (get_rf_orientation() must read as "nothing published").
    """
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", True)
    monkeypatch.setattr(acfg, "_INITIALIZED_REQUEST", None)
    # RF-DIR-001B: neutralize the published RF-edge orientation as well, so a
    # leftover value from an earlier real initialization cannot masquerade as
    # this synthetic "nothing published" state (get_rf_orientation() -> None).
    monkeypatch.setattr(acfg, "_RF_ORIENTATION", None)
    return acfg
