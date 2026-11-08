"""Input and transient-window contracts for the production FFT IF extractor."""

import numpy as np
import pytest
from scipy.fft import fft, fftfreq

from rydberg_aoa.utils.numerical import extract_if_component


def _record():
    t_us = np.arange(1000) * 0.1
    trace = 0.75 * np.cos(2 * np.pi * 0.1 * t_us + 0.4)
    return t_us, trace


def test_delay_beyond_record_is_rejected_without_reusing_tail():
    t, trace = _record()
    with pytest.raises(ValueError, match="No samples.*delay"):
        extract_if_component(trace, t, 0.1, delay_us=200)


def test_too_few_samples_after_delay_are_rejected():
    t, trace = _record()
    with pytest.raises(ValueError, match="Insufficient data points"):
        extract_if_component(trace, t, 0.1, delay_us=t[-5])


@pytest.mark.parametrize("frequency,phase_sign", [(0.1, 1), (-0.1, -1)])
def test_signed_if_recovers_known_amplitude_and_phase(frequency, phase_sign):
    t, trace = _record()
    amplitude, _, _ = extract_if_component(trace, t, frequency)
    assert amplitude == pytest.approx(0.75 * np.exp(phase_sign * 0.4j),
                                     rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("delay", [0.0, 40.0])
def test_valid_fft_output_is_unchanged_and_inputs_are_not_mutated(delay):
    t, trace = _record()
    before_t, before_trace = t.copy(), trace.copy()
    mask = t >= delay
    reference_fft = fft(trace[mask])
    reference_frequency = fftfreq(mask.sum(), t[mask][1] - t[mask][0])
    index = np.argmin(np.abs(reference_frequency - 0.1))
    result, frequencies, spectrum = extract_if_component(
        trace.tolist(), t.tolist(), 0.1, delay_us=delay)
    assert result == pytest.approx((reference_fft[index] / mask.sum()) * 2,
                                 rel=1e-12, abs=1e-12)
    np.testing.assert_array_equal(frequencies, reference_frequency)
    np.testing.assert_array_equal(spectrum, reference_fft)
    np.testing.assert_array_equal(t, before_t)
    np.testing.assert_array_equal(trace, before_trace)


@pytest.mark.parametrize("which,bad", [
    ("time", np.nan), ("time", np.inf),
    ("signal", np.nan), ("signal", np.inf),
])
def test_nonfinite_samples_are_rejected(which, bad):
    t, trace = _record()
    (t if which == "time" else trace)[20] = bad
    with pytest.raises(ValueError, match="finite"):
        extract_if_component(trace, t, 0.1)


@pytest.mark.parametrize("case", ["duplicate", "decreasing", "nonuniform"])
def test_invalid_time_grid_is_rejected(case):
    t, trace = _record()
    if case == "duplicate":
        t[20] = t[19]
    elif case == "decreasing":
        t = t[::-1]
    else:
        t[20] += 0.01
    with pytest.raises(ValueError, match="increasing|uniform"):
        extract_if_component(trace, t, 0.1)


@pytest.mark.parametrize("case", ["time_2d", "signal_2d", "mismatch", "empty"])
def test_invalid_array_shapes_are_rejected(case):
    t, trace = _record()
    if case == "time_2d":
        t = t.reshape(10, 100)
    elif case == "signal_2d":
        trace = trace.reshape(10, 100)
    elif case == "mismatch":
        trace = trace[:-1]
    else:
        t, trace = np.array([]), np.array([])
    with pytest.raises(ValueError):
        extract_if_component(trace, t, 0.1)


@pytest.mark.parametrize("frequency", [0.0, 5.0, -5.0, 6.0, -6.0,
                                      np.nan, np.inf, True, "0.1", 0.1j])
def test_invalid_if_is_rejected(frequency):
    t, trace = _record()
    with pytest.raises(ValueError):
        extract_if_component(trace, t, frequency)


@pytest.mark.parametrize("delay", [-1.0, np.nan, np.inf, True, "40", 1j])
def test_invalid_delay_is_rejected(delay):
    t, trace = _record()
    with pytest.raises(ValueError):
        extract_if_component(trace, t, 0.1, delay_us=delay)


def test_complex_time_is_rejected_without_discarding_imaginary_part():
    t, trace = _record()
    with pytest.raises(ValueError, match="real"):
        extract_if_component(trace, t.astype(complex) + 1j, 0.1)


def test_nonfinite_fft_output_is_rejected():
    t, _ = _record()
    with pytest.raises(ValueError, match="FFT.*finite"):
        extract_if_component(np.full(t.size, 1e308), t, 0.1)
