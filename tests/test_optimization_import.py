"""
Import verification for the legacy two-stage optimizer.

Audit B.3 fix: this was a script-style test executing ``exit(1)`` at module level
during pytest collection, which raised SystemExit inside Tk's mainloop and killed
the entire test runner. The deprecated ``rydberg_aoa.optimization`` module has been
removed from the project (audit Part B lean inventory), so this test now degrades
to a documented skip instead of aborting the suite.
"""

import pytest

try:
    from rydberg_aoa.optimization import TwoStageLevelOptimizer  # noqa: F401
    _LEGACY_OPTIMIZER_AVAILABLE = True
except ImportError:
    _LEGACY_OPTIMIZER_AVAILABLE = False


@pytest.mark.skipif(
    not _LEGACY_OPTIMIZER_AVAILABLE,
    reason="Deprecated rydberg_aoa.optimization module removed (audit Part B / B.3)",
)
def test_two_stage_optimizer_import():
    """The legacy two-stage API must still be importable while it exists."""
    from rydberg_aoa.optimization import TwoStageLevelOptimizer, OptimizationCriteria

    # Basic initialization contract of the deprecated entry point
    optimizer = TwoStageLevelOptimizer()
    criteria = OptimizationCriteria()
    assert optimizer is not None
    assert criteria.max_dipole_strength > 0
