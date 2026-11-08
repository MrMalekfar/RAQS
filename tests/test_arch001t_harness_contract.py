"""ARCH-001T — the test harness itself must not undo ARCH-001.

The shared fixtures added by ARCH-001T live in ``tests/conftest.py`` and
``tests/atomic_init_support.py``.  Importing them must never initialize
atomic physics — otherwise every future test session would silently return
to import-time physics and the ARCH-001 "import != initialize" contract
would be defeated from inside the test infrastructure.

The ``BLOCKED: QN-001A`` skip path is gated on the EXACT documented
coupling boundary (exception type + three message substrings).  Those
gates are pinned here so a future message change cannot silently convert a
real initialization failure into a skip.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from rydberg_aoa.exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
)
from tests.atomic_init_support import is_documented_qn001a_coupling_block

REPO_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_in_fresh_interpreter(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_importing_the_test_harness_initializes_nothing():
    """ARCH-001: importing the new fixtures/helpers publishes NO physics.

    Runs in a fresh interpreter so it observes the real import path,
    unaffected by this session's module state.
    """
    result = _run_in_fresh_interpreter(
        """
        import tests.atomic_init_support  # noqa: F401
        import tests.conftest             # noqa: F401

        from rydberg_aoa.config import atomic_config as acfg
        print("STATE", acfg.is_atomic_physics_initialized(), acfg.ATOMIC_LEVELS)
        """
    )
    assert result.returncode == 0, (
        "importing the ARCH-001T harness failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "STATE False None" in result.stdout, (
        "importing the ARCH-001T harness initialized atomic physics:\n"
        f"{result.stdout}"
    )
    combined = result.stdout + result.stderr
    assert "Optimizing atomic levels" not in combined, (
        "harness import triggered the ARC level optimization"
    )


def test_predicate_accepts_the_real_documented_block():
    """The skip gate matches the REAL production QN-001A message.

    Runs one explicit seam attempt in a fresh interpreter (transactional:
    nothing is published and the session state here is untouched) and
    applies the same predicate the fixture uses.
    """
    result = _run_in_fresh_interpreter(
        """
        import rydberg_aoa.config.atomic_config as acfg
        from rydberg_aoa.exceptions import PhysicsConfigurationError
        from tests.atomic_init_support import (
            is_documented_qn001a_coupling_block,
        )

        try:
            acfg.initialize_atomic_physics()
        except PhysicsConfigurationError as exc:
            print("HARNESS_PREDICATE", is_documented_qn001a_coupling_block(exc))
        else:
            print("HARNESS_INITIALIZED")
        """
    )
    assert result.returncode == 0, (
        f"probe failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if "HARNESS_INITIALIZED" in result.stdout:
        pytest.skip(
            "QN-001A coupling boundary resolved — the real blocked message no "
            "longer exists to pin (predicate coverage will resume if it returns)"
        )
    assert "HARNESS_PREDICATE True" in result.stdout, (
        "the skip gate did not recognize the real documented QN-001A "
        f"message:\n{result.stdout}"
    )


def test_predicate_rejects_near_miss_failures():
    """Anything but the exact documented boundary must NOT become a skip."""
    canonical = (
        "the driven laser transition 5P3/2 F=4 -> 26D2.5 couples a hyperfine "
        "source to a fine-structure destination (HFS->FS) but the destination "
        "declares no mJ. An explicit Rydberg mJ / Zeeman-channel contract is "
        "required: no mJ is inferred and the code does not scan mJ for a "
        "maximum. F05: no implicit magnetic sublevel."
    )
    # Canonical message and type -> recognized.
    assert is_documented_qn001a_coupling_block(
        PhysicsConfigurationError(canonical))

    # Any single missing substring -> NOT the documented boundary.
    for dropped in ("HFS->FS", "no mJ is inferred", "does not scan mJ"):
        broken = canonical.replace(dropped, "PLACEHOLDER")
        assert not is_documented_qn001a_coupling_block(
            PhysicsConfigurationError(broken)), (
            f"predicate accepted a message missing {dropped!r}"
        )

    # Wrong exception types -> never a skip, even with the exact text.
    assert not is_documented_qn001a_coupling_block(RuntimeError(canonical))
    assert not is_documented_qn001a_coupling_block(
        PhysicsCalculationError(canonical))
