"""RF-003 regression tests for :func:`get_cached_detunings`.

RF-003 requires that a genuinely-missing detuning entry FAILS LOUDLY
(RuntimeError) instead of silently returning (0,0,0), while a *present* entry
that legitimately holds zero (e.g. coupling on resonance) continues to be
returned without error. The valid-zero vs failed distinction must be made by
entry presence, not by value.

ARCH-001T: the guarded public reader requires the explicit ARCH-001
initialized state. The valid-cache test opts into the real seam
(``atomic_physics``); the two synthetic-cache contract tests use the minimal
``published_atomic_state`` fixture so that the RF-003 cache-miss path ITSELF
is what raises — previously the uninitialized-state guard satisfied
``pytest.raises(RuntimeError)`` accidentally.
"""

import pytest

from rydberg_aoa.config import atomic_config as acfg


@pytest.fixture(autouse=True)
def _restore_detuning_cache():
    """Snapshot and restore the detuning cache around each test."""
    import copy
    snapshot = copy.deepcopy(acfg._DETUNING_CACHE)
    yield
    acfg._DETUNING_CACHE.clear()
    acfg._DETUNING_CACHE.update(snapshot)


def test_valid_cache_returns_stored_values_no_error(atomic_physics):
    """A populated cache returns the real values without raising."""
    dp, dc, drf = acfg.get_cached_detunings()
    # The import-time pipeline populates this cache; the call must not raise.
    assert isinstance(dp, float)
    assert isinstance(dc, float)
    assert isinstance(drf, float)


def test_present_zero_entry_returns_zero_without_error(
        published_atomic_state, monkeypatch):
    """A present entry of all zeros is a legitimate zero, not a failure."""
    key = ("present-zero-entry",)
    entry = {'delta_p_rad_us': 0.0, 'delta_c_rad_us': 0.0, 'delta_rf_rad_us': 0.0}
    acfg._DETUNING_CACHE[key] = entry
    monkeypatch.setattr(acfg, "_generate_cache_key", lambda *a, **k: key)

    dp, dc, drf = acfg.get_cached_detunings()
    assert (dp, dc, drf) == (0.0, 0.0, 0.0)


def test_cache_miss_raises_runtime_error(published_atomic_state, monkeypatch):
    """A missing entry must raise RuntimeError, not silently return zeros."""
    sentinel_key = ("no-such-entry-key",)
    # Guarantee the sentinel is absent from the cache.
    acfg._DETUNING_CACHE.pop(sentinel_key, None)
    monkeypatch.setattr(acfg, "_generate_cache_key", lambda *a, **k: sentinel_key)

    with pytest.raises(RuntimeError):
        acfg.get_cached_detunings()