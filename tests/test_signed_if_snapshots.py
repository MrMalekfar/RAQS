"""Signed IF projection and its existing RF-orientation/AoA boundary."""

import numpy as np
import pytest

from rydberg_aoa.doa import IFSnapshotData, extract_if_snapshots


def _record():
    t = np.arange(1000) * 0.1
    phases = np.array([0.3, -0.7])
    amplitudes = np.array([0.75, 1.25])
    traces = amplitudes[:, None] * np.cos(
        2 * np.pi * 0.1 * t[None, :] + phases[:, None])
    return t, traces, amplitudes, phases


@pytest.mark.parametrize("frequency", [0.1, -0.1])
def test_signed_projection_preserves_known_amplitude_phase_and_metadata(frequency):
    t, traces, amplitudes, phases = _record()
    before = traces.copy()
    snapshots = extract_if_snapshots(t, traces, frequency, start_us=0)
    expected = amplitudes * np.exp(1j * np.sign(frequency) * phases)
    np.testing.assert_allclose(snapshots.X, np.broadcast_to(
        expected[:, None], snapshots.X.shape), rtol=1e-12, atol=1e-12)
    assert snapshots.if_frequency_mhz == frequency
    assert snapshots.samples_per_window == 200
    assert snapshots.window_duration_us > 0
    assert snapshots.diagnostics["samples_per_if_cycle"] == pytest.approx(100)
    np.testing.assert_array_equal(traces, before)


def test_opposite_if_signs_conjugate_real_trace_projection_without_changing_windows():
    t, traces, _, _ = _record()
    positive = extract_if_snapshots(t, traces, 0.1, start_us=0)
    negative = extract_if_snapshots(t, traces, -0.1, start_us=0)
    np.testing.assert_array_equal(negative.X, positive.X.conj())
    np.testing.assert_array_equal(negative.time_centers_us, positive.time_centers_us)
    assert negative.diagnostics == positive.diagnostics
    assert negative.samples_per_window == positive.samples_per_window


def test_positive_if_retains_existing_projection_exactly():
    t, traces, _, _ = _record()
    snapshots = extract_if_snapshots(t, traces, 0.1, start_us=0)
    windows = traces.astype(complex).reshape(2, 5, 200)
    windows = windows - windows.mean(axis=2, keepdims=True)
    oscillator = np.exp(-1j * 2 * np.pi * 0.1 * t.reshape(5, 200))
    expected = (2.0 / 200) * (windows * oscillator[None, :, :]).sum(axis=2)
    np.testing.assert_array_equal(snapshots.X, expected)


@pytest.mark.parametrize("frequency", [0.0, np.nan, np.inf, -np.inf])
def test_invalid_if_is_rejected_by_extractor_and_container(frequency):
    t, traces, _, _ = _record()
    with pytest.raises(ValueError, match="nonzero finite"):
        extract_if_snapshots(t, traces, frequency, start_us=0)
    with pytest.raises(ValueError, match="nonzero finite"):
        IFSnapshotData(X=np.ones((2, 1), dtype=complex),
                       time_centers_us=np.array([1.0]),
                       if_frequency_mhz=frequency, n_paths=2,
                       window_duration_us=1.0, samples_per_window=10)


@pytest.mark.parametrize("frequency", [0.1, -0.1])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("method", ["bartlett", "music"])
def test_signed_if_and_rf_orientation_recover_spatial_angle(
        frequency, direction, method, monkeypatch):
    """Synthetic boundary oracle; does not claim a solved atomic response."""
    from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.simulation.doa_pipeline import estimate_doa_from_results

    monkeypatch.setattr(params, "IF_FREQ", frequency)
    carrier = params.RF_SIGNAL_FREQUENCY
    spacing = 299792458.0 / carrier / 8
    x_m = np.arange(6) * spacing
    positions = np.column_stack((x_m, np.zeros_like(x_m)))
    angle_deg = -30.0
    spatial_phase = 2 * np.pi * carrier / 299792458.0 * x_m * np.sin(
        np.deg2rad(angle_deg))
    t = np.arange(1000) * 0.1
    # Real{exp(i*phi) exp(i*direction*f*t)} equals
    # cos(2*pi*f*t + direction*phi). Signed projection recovers
    # direction*phi, then the existing downward boundary conjugates once.
    traces = np.cos(2 * np.pi * frequency * t[None, :] +
                    direction * spatial_phase[:, None])
    result, snapshots = estimate_doa_from_results(
        positions, np.exp(1j * spatial_phase), t, traces, method=method,
        rf_orientation=resolve_rf_orientation(direction * 132.649e9))
    assert result.azimuth_deg == pytest.approx(angle_deg, abs=1e-9)
    assert snapshots.if_frequency_mhz == frequency
    np.testing.assert_allclose(snapshots.X, np.broadcast_to(
        np.exp(1j * spatial_phase)[:, None], snapshots.X.shape),
        rtol=1e-12, atol=1e-12)
