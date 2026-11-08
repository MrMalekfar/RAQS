"""Exact-frequency IF extraction against independent sinusoid/array oracles."""

import numpy as np
import pytest

from rydberg_aoa.doa import ArrayGeometry, PhaseBeamformingEstimator
from rydberg_aoa.utils.numerical import extract_if_component


@pytest.mark.parametrize("frequency", [0.1037, -0.1037, 0.1])
@pytest.mark.parametrize("count", [137, 300, 600, 1000])
def test_known_tone_with_dc_recovers_amplitude_and_phase(frequency, count):
    t = 7.3 + 0.1 * np.arange(count)
    trace = 3.7 + 0.75 * np.cos(2 * np.pi * frequency * t + 0.4)
    amplitude, _, _ = extract_if_component(trace, t, frequency)
    expected = 0.75 * np.exp(1j * (0.4 + 2 * np.pi * frequency * t[0]))
    assert amplitude == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("frequency", [0.1037, -0.1037])
def test_transient_cut_uses_first_retained_sample_as_phase_origin(frequency):
    t = 0.1 * np.arange(600)
    trace = -0.5 + 1.2 * np.cos(2 * np.pi * frequency * t - 0.8)
    amplitude, _, _ = extract_if_component(trace, t, frequency, delay_us=17.35)
    first = t[t >= 17.35][0]
    assert amplitude == pytest.approx(
        1.2 * np.exp(1j * (2 * np.pi * frequency * first - 0.8)), abs=1e-12)


@pytest.mark.parametrize("frequency", [0.1037, -0.1037])
def test_complex_trace_preserves_two_sided_factor_of_two(frequency):
    t = 0.1 * np.arange(137)
    trace = (1 + 2j) + (0.4 - 0.3j) * np.exp(2j * np.pi * frequency * t)
    amplitude, _, _ = extract_if_component(trace, t, frequency)
    assert amplitude == pytest.approx(2 * (0.4 - 0.3j), abs=1e-12)


@pytest.mark.parametrize("frequency", [0.1037, -0.1037])
@pytest.mark.parametrize("angle", [-30, 25])
def test_off_bin_extraction_preserves_known_array_angle(frequency, angle):
    rf_hz, c_m_s = 132.6491e9, 299792458.0
    positions = np.arange(6) * c_m_s / rf_hz / 8
    geometry = ArrayGeometry.from_xy(np.column_stack((positions, positions * 0)))
    spatial_phase = 2 * np.pi * rf_hz / c_m_s * positions * np.sin(np.deg2rad(angle))
    t = 0.1 * np.arange(137)
    traces = [0.2 + 0.75 * np.cos(2 * np.pi * frequency * t + phase)
              for phase in spatial_phase]
    responses = np.array([extract_if_component(y, t, frequency)[0] for y in traces])
    np.testing.assert_allclose(responses, 0.75 * np.exp(1j * spatial_phase), atol=1e-12)
    result = PhaseBeamformingEstimator().estimate(
        responses, geometry, {"frequency_hz": rf_hz})
    assert result.azimuth_deg == angle


def test_unresolved_near_dc_basis_is_rejected_without_fft_fallback():
    t = 0.1 * np.arange(100)
    with pytest.raises(ValueError, match="rank"):
        extract_if_component(np.ones(t.size), t, 1e-20)


def test_fit_failure_preserves_cause_without_fft_fallback(monkeypatch):
    failure = np.linalg.LinAlgError("SVD failed")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(np.linalg, "lstsq", fail)
    t = 0.1 * np.arange(100)
    with pytest.raises(ValueError, match="IF.*fit") as error:
        extract_if_component(np.cos(2 * np.pi * 0.1037 * t), t, 0.1037)
    assert error.value.__cause__ is failure
