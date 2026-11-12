"""
Regression tests: dipole-matrix cache crash on wavelength change (Issues #4 / N-1).

Covers:
- Cache key must respond to BOTH probe and coupling wavelengths (N-1 fix).
- get_cached_dipole_matrix() must self-heal after a cache clear.
- calculate_absorption_constant() must recover via self-heal and must fail loudly
  (RuntimeError, not TypeError) when no dipole matrix can be produced.

NOTE: importing ``rydberg_aoa.config.atomic_config`` no longer runs physics
(ARCH-001): resolved quantities exist only after the explicit
``initialize_atomic_physics()`` seam succeeded.  The self-heal class below
therefore uses the explicit prepared-channel fixture.
"""

import numpy as np
import pytest

from tests.test_physics_baseline_regression import isolated, prepared_atomic_physics


class TestCacheKeyWavelengthSensitivity:
    """N-1 regression: λ_p must invalidate the cache independently of λ_c."""

    def test_probe_wavelength_changes_key(self):
        # D2 vs D1 probe lines differ ONLY in λ_p → keys must differ
        from rydberg_aoa.config.atomic_config import _generate_cache_key
        coupling = {'wavelength_nm': 480.0, 'power_mw': 10.0}
        rf = {'frequency_hz': 10.0e9, 'electric_field_vm': 0.5}
        probe_d2 = {'wavelength_nm': 780.2412, 'power_mw': 0.001}
        probe_d1 = {'wavelength_nm': 794.9790, 'power_mw': 0.001}
        assert _generate_cache_key(probe_d2, coupling, rf) != \
               _generate_cache_key(probe_d1, coupling, rf)

    def test_coupling_wavelength_changes_key(self):
        from rydberg_aoa.config.atomic_config import _generate_cache_key
        probe = {'wavelength_nm': 780.2412, 'power_mw': 0.001}
        rf = {'frequency_hz': 10.0e9, 'electric_field_vm': 0.5}
        coupling_480 = {'wavelength_nm': 480.0, 'power_mw': 10.0}
        coupling_483 = {'wavelength_nm': 483.0, 'power_mw': 10.0}
        assert _generate_cache_key(probe, coupling_480, rf) != \
               _generate_cache_key(probe, coupling_483, rf)


class TestSelfHealAndAbsorptionConstant:
    """Issue #4 regression: consumers must never receive an unguarded None."""

    def test_self_heal_after_full_cache_clear(self, prepared_atomic_physics):
        acfg = prepared_atomic_physics
        reference = acfg.get_cached_dipole_matrix()
        assert reference is not None and 'probe' in reference

        # Simulate the post-wavelength-change cache miss, then verify self-heal
        acfg._DIPOLE_MATRIX_CACHE.clear()
        healed = acfg.get_cached_dipole_matrix()

        assert healed is not None and 'probe' in healed
        # Same optimized levels ⇒ bitwise-consistent μ₂₁ [C·m] via the canonical path
        assert np.isclose(healed['probe']['dipole_si'], reference['probe']['dipole_si'])

    def test_absorption_constant_survives_cache_clear(self, prepared_atomic_physics):
        from rydberg_aoa.physics.susceptibility.absorption_const import (
            calculate_absorption_constant,
        )
        prepared_atomic_physics._DIPOLE_MATRIX_CACHE.clear()
        od_factor, K_chi = calculate_absorption_constant()
        assert np.isfinite(od_factor) and od_factor > 0   # πL/λ_p strictly positive
        assert np.isfinite(K_chi) and K_chi < 0           # −2N₀μ²/(ε₀ħω_p) strictly negative

    def test_missing_dipole_raises_runtime_error_not_type_error(self, monkeypatch):
        from rydberg_aoa.physics.susceptibility import absorption_const
        # Force the worst case (None) to verify the explicit guard (Issue #4)
        monkeypatch.setattr(absorption_const.acfg, 'get_cached_dipole_matrix', lambda: None)
        with pytest.raises(RuntimeError):
            absorption_const.calculate_absorption_constant()
