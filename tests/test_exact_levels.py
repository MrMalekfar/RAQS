"""ATOMIC_LEVELS structural contract: the four-level cascade is populated.

Replaces a print-only collection-time script with asserted structure checks:
all four cascade levels exist, carry labels, and are energetically ordered
E_ground < E_excited < min(E_rydberg1, E_rydberg2) (hyperfine ground manifold
to the near-degenerate RF-dressed Rydberg pair). Numerical values against the
frozen snapshot live in test_physics_baseline_regression.py; this file guards
structure only.
"""
from rydberg_aoa.config import atomic_config as acfg

_CASCADE = ('ground', 'excited', 'rydberg1', 'rydberg2')


def test_all_four_cascade_levels_present_with_labels(atomic_physics):
    """Each cascade slot must exist and carry a human-readable ARC label."""
    for key in _CASCADE:
        assert key in acfg.ATOMIC_LEVELS, f"missing cascade level {key}"
        assert acfg.ATOMIC_LEVELS[key].get('label'), f"{key} has no label"


def test_optical_ladder_below_near_degenerate_rydberg_pair(atomic_physics):
    """E_ground < E_excited < min(E_ryd1, E_ryd2).

    The optical ladder (probe + coupling) is strictly monotonic, but the RF
    pair is near-degenerate by construction — the transition scan places the
    two Rydberg states |E_ryd1 - E_ryd2| = h·f_RF apart, so their internal
    ordering is set by the RF scan, not by the ladder.
    """
    e_ground = acfg.ATOMIC_LEVELS['ground']['energy']
    e_excited = acfg.ATOMIC_LEVELS['excited']['energy']
    e_ryd = [acfg.ATOMIC_LEVELS['rydberg1']['energy'],
             acfg.ATOMIC_LEVELS['rydberg2']['energy']]
    assert e_ground < e_excited < min(e_ryd), (
        f"ladder violated: ground={e_ground}, excited={e_excited}, ryd={e_ryd}")
    # Near-degeneracy sanity: the RF pair splitting stays sub-optical, i.e.
    # far below the optical transition energy scale.
    assert abs(e_ryd[0] - e_ryd[1]) < abs(e_excited - e_ground)


def test_rydberg_pair_labels_are_distinct(atomic_physics):
    """The RF-dressed pair must be two different Rydberg states."""
    r1 = acfg.ATOMIC_LEVELS['rydberg1']['label']
    r2 = acfg.ATOMIC_LEVELS['rydberg2']['label']
    assert r1 != r2, "rydberg1 and rydberg2 resolved to the same state"

