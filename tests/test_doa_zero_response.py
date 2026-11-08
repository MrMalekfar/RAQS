"""Beamforming must not turn absent or unrepresentable power into an angle."""

import numpy as np
import pytest

from rydberg_aoa.doa import ArrayGeometry, DOAError, PhaseBeamformingEstimator


RF_HZ = 132.6491e9
C_M_S = 299792458.0


def _plane_wave(angle_deg):
    """Independent plane-wave oracle: x-axis ULA, angle from broadside."""
    spacing = C_M_S / RF_HZ / 8
    x_m = np.arange(6) * spacing
    geometry = ArrayGeometry.from_xy(np.column_stack((x_m, np.zeros_like(x_m))))
    phase = 2 * np.pi * RF_HZ / C_M_S * x_m * np.sin(np.deg2rad(angle_deg))
    return geometry, np.exp(1j * phase)


@pytest.mark.parametrize("dtype", [float, complex])
def test_zero_response_raises_instead_of_returning_first_scan_angle(dtype):
    geometry, _ = _plane_wave(-30)
    estimator = PhaseBeamformingEstimator(scan_azimuth_deg=[-70, 0, 45])
    with pytest.raises(DOAError, match="zero.*response"):
        estimator.estimate(np.zeros(geometry.m, dtype=dtype), geometry,
                           {"frequency_hz": RF_HZ})


@pytest.mark.parametrize("angle_deg", [-30, 25])
@pytest.mark.parametrize("amplitude", [1.0, 1e-12, 1e-100])
def test_nonzero_plane_wave_keeps_angle_without_amplitude_threshold(angle_deg,
                                                                  amplitude):
    geometry, wave = _plane_wave(angle_deg)
    data = amplitude * wave
    before = data.copy()
    result = PhaseBeamformingEstimator().estimate(
        data, geometry, {"frequency_hz": RF_HZ})
    assert result.azimuth_deg == angle_deg
    assert np.max(result.spectrum) > 0
    assert np.all(np.isfinite(result.spectrum))
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("amplitude", [1e-200, 1e200])
def test_unrepresentable_power_fails_instead_of_fabricating_an_angle(amplitude):
    geometry, wave = _plane_wave(-30)
    with pytest.raises(DOAError, match="power"):
        PhaseBeamformingEstimator().estimate(
            amplitude * wave, geometry, {"frequency_hz": RF_HZ})


def test_valid_spectrum_retains_unnormalized_power_and_phase_convention():
    geometry, wave = _plane_wave(-30)
    grid = np.array([-30, 0, 30])
    amplitude = 0.25
    result = PhaseBeamformingEstimator(scan_azimuth_deg=grid).estimate(
        amplitude * wave, geometry, {"frequency_hz": RF_HZ})
    x_m = geometry.positions_m[:, 0]
    expected = []
    for angle in grid:
        phase = 2 * np.pi * RF_HZ / C_M_S * x_m * np.sin(np.deg2rad(angle))
        expected.append(abs(np.sum(np.exp(-1j * phase) * amplitude * wave)) ** 2)
    np.testing.assert_allclose(result.spectrum, expected, rtol=1e-14, atol=0)
    assert result.spectrum[0] == pytest.approx((geometry.m * amplitude) ** 2)
    assert result.azimuth_deg == -30
