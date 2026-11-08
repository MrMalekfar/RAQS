"""Reusing a processed snapshot must not apply the RF phase conversion twice."""

from dataclasses import replace

import numpy as np
import pytest

from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import extract_if_snapshots
from rydberg_aoa.simulation import doa_pipeline


def _run_data(frequency, direction):
    carrier = params.RF_SIGNAL_FREQUENCY
    x_m = np.arange(6) * (299792458.0 / carrier / 8)
    positions = np.column_stack((x_m, np.zeros_like(x_m)))
    phases = 2 * np.pi * carrier / 299792458.0 * x_m * np.sin(np.deg2rad(-30))
    t = np.arange(1000) * 0.1
    traces = np.cos(2 * np.pi * frequency * t[None, :] +
                    direction * phases[:, None])
    raw = extract_if_snapshots(t, traces, frequency, start_us=40)
    return positions, np.exp(1j * phases), t, traces, raw


@pytest.mark.parametrize("frequency", [0.1, -0.1])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("first,second", [("bartlett", "music"),
                                         ("music", "bartlett")])
def test_processed_snapshot_reuse_keeps_angle_and_phase(
        frequency, direction, first, second, monkeypatch):
    monkeypatch.setattr(params, "IF_FREQ", frequency)
    positions, response, t, traces, raw = _run_data(frequency, direction)
    original = raw.X.copy()
    orientation = resolve_rf_orientation(direction * 132.649e9)
    result1, processed = doa_pipeline.estimate_doa_from_results(
        positions, response, t, traces, method=first,
        if_snapshots=raw, rf_orientation=orientation)
    result2, reused = doa_pipeline.estimate_doa_from_results(
        positions, response, t, traces, method=second,
        if_snapshots=processed, rf_orientation=orientation)
    assert result1.azimuth_deg == pytest.approx(-30)
    assert result2.azimuth_deg == pytest.approx(-30)
    np.testing.assert_array_equal(reused.X, processed.X)
    np.testing.assert_array_equal(raw.X, original)
    assert raw.normalization_polarity is None
    assert processed.normalization_polarity == direction
    assert reused is processed


def test_reuse_does_not_reextract_or_use_new_traces(monkeypatch):
    monkeypatch.setattr(params, "IF_FREQ", 0.1)
    positions, response, t, traces, raw = _run_data(0.1, -1)
    orientation = resolve_rf_orientation(-132.649e9)
    _, processed = doa_pipeline.estimate_doa_from_results(
        positions, response, t, traces, method="bartlett",
        if_snapshots=raw, rf_orientation=orientation)

    def forbidden(*args, **kwargs):
        raise AssertionError("A supplied snapshot must not be extracted again")

    monkeypatch.setattr(doa_pipeline, "extract_snapshots_once", forbidden)
    result, reused = doa_pipeline.estimate_doa_from_results(
        positions, response, None, None, method="music",
        if_snapshots=processed, rf_orientation=orientation)
    assert result.azimuth_deg == pytest.approx(-30)
    assert reused is processed


@pytest.mark.parametrize("direction,opposite", [
    (-1, None), (-1, resolve_rf_orientation(132.649e9)),
    (1, resolve_rf_orientation(-132.649e9)),
])
def test_processed_snapshot_rejects_incompatible_rf_polarity(
        direction, opposite, monkeypatch):
    monkeypatch.setattr(params, "IF_FREQ", 0.1)
    positions, response, t, traces, raw = _run_data(0.1, direction)
    _, processed = doa_pipeline.estimate_doa_from_results(
        positions, response, t, traces, method="bartlett", if_snapshots=raw,
        rf_orientation=resolve_rf_orientation(direction * 132.649e9))
    with pytest.raises(ValueError, match="polarity"):
        doa_pipeline.estimate_doa_from_results(
            positions, response, t, traces, method="music",
            if_snapshots=processed, rf_orientation=opposite)


def test_supplied_snapshot_rejects_different_signed_if(monkeypatch):
    monkeypatch.setattr(params, "IF_FREQ", 0.1)
    positions, response, t, traces, raw = _run_data(0.1, 1)
    monkeypatch.setattr(params, "IF_FREQ", -0.1)
    with pytest.raises(ValueError, match="IF"):
        doa_pipeline.estimate_doa_from_results(
            positions, response, t, traces, method="bartlett", if_snapshots=raw)


@pytest.mark.parametrize("invalid", [0, 2, True, "-1", 0.5])
def test_snapshot_rejects_invalid_normalization_provenance(invalid):
    _, _, _, _, raw = _run_data(0.1, 1)
    with pytest.raises(ValueError, match="normalization_polarity"):
        replace(raw, normalization_polarity=invalid)
