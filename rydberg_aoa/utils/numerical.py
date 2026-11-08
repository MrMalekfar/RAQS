"""
Numerical utilities for signal processing and FFT analysis.
"""

import numpy as np
from scipy.fft import fft, fftfreq


def _finite_real_scalar(value, name):
    """Validate FFT scalar inputs without coercing strings, bools or complex."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def _if_fit_grid(t_list, if_freq_mhz, delay_us):
    """Shared retained grid/design for production fitting and noise weights."""
    t_list = np.asarray(t_list)
    if (t_list.ndim != 1 or t_list.dtype.kind not in 'iuf' or
            not np.all(np.isfinite(t_list))):
        raise ValueError('Time samples must be a finite real numeric vector')
    delay_us = _finite_real_scalar(delay_us, 'delay_us')
    frequency = _finite_real_scalar(if_freq_mhz, 'if_freq_mhz')
    if delay_us < 0:
        raise ValueError('delay_us must be nonnegative')
    if frequency == 0:
        raise ValueError('Zero IF requires a DC measurement contract, not beat-note extraction')
    if t_list.size < 10:
        raise ValueError('Insufficient data points; at least 10 are required')
    steps = np.diff(t_list.astype(float))
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError('Time samples must be strictly increasing with finite steps')
    if not np.allclose(steps, steps[0], rtol=1e-7, atol=0):
        raise ValueError('Time samples must be uniformly spaced')
    if abs(frequency) >= 0.5 / steps[0]:
        raise ValueError('IF must lie strictly inside the Nyquist interval')
    selected = np.flatnonzero(t_list >= delay_us) if delay_us > 0 else np.arange(len(t_list))
    if not len(selected):
        raise ValueError(f'No samples remain at or after delay_us={delay_us:g}')
    if len(selected) < 10:
        raise ValueError(f'Insufficient data points after delay ({len(selected)}). Increase simulation time or reduce delay.')
    retained = t_list[selected]
    try:
        with np.errstate(over='raise', invalid='raise'):
            phase = (2 * np.pi * frequency) * (retained.astype(float) - float(retained[0]))
            design = np.column_stack((np.ones(len(retained)), np.cos(phase), np.sin(phase)))
    except FloatingPointError as exc:
        raise ValueError('IF sinusoidal fit failed') from exc
    return selected, retained, design


def if_fit_noise_weights(t_list, if_freq_mhz, delay_us=0.0):
    """Return retained indices, complex linear weights and white-noise gains.

    For real samples y, A_IF = sum(w*y), with the same DC/cos/sin design as
    extract_if_component. No conjugation is applied in that dot product.
    With independent real sample variance v, E|A_noise|^2 = v*sum(|w|^2).
    ENBW here uses one-sided white input PSD S and v=S*fs/2; the complex
    peak-amplitude convention sums both quadratures (not an analog bandwidth).
    """
    selected, retained, design = _if_fit_grid(t_list, if_freq_mhz, delay_us)
    try:
        weights, _, rank, singular = np.linalg.lstsq(
            design.T, np.array([0., 1., -1j]), rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError('IF noise-weight fit failed') from exc
    if rank != 3:
        raise ValueError('IF sinusoidal fit is rank deficient; DC and IF are unresolved')
    if not np.isfinite(weights).all() or not np.isfinite(singular).all():
        raise ValueError('IF noise weights must be finite')
    fs = 1e6 / (retained[1] - retained[0])
    with np.errstate(over='raise', invalid='raise'):
        try:
            iq_gain = np.array([[np.dot(weights.real, weights.real), np.dot(weights.real, weights.imag)],
                                [np.dot(weights.imag, weights.real), np.dot(weights.imag, weights.imag)]])
            iq_enbw = fs * .5 * iq_gain
            enbw = float(np.trace(iq_enbw))
        except FloatingPointError as exc:
            raise ValueError('IF noise gain overflow') from exc
    if not np.isfinite(iq_enbw).all() or not np.isfinite(enbw) or enbw <= 0:
        raise ValueError('IF noise gain must be finite and positive')
    return selected, weights, {
        'retained_samples': len(selected), 'time_origin_us': float(retained[0]),
        'sample_rate_hz': float(fs), 'signed_if_mhz': float(if_freq_mhz),
        'complex_peak_amplitude_white_enbw_hz': enbw,
        'iq_white_enbw_matrix_hz': iq_enbw.tolist(),
        'noise_definition': 'E|A_noise|^2 = one-sided white input PSD * complex_peak_amplitude_white_enbw_hz',
        'scope': 'DC/cos/sin fit on retained samples; not sensor bandwidth or detector boxcar ENBW',
    }


def extract_if_component(signal_trace, t_list, if_freq_mhz, delay_us=0.0):
    """
    Extract a known intermediate frequency by a sinusoidal least-squares fit,
    optionally applying a delay to remove atomic transients.

    Args:
        signal_trace: Time-domain signal array
        t_list: Time array (microseconds)
        if_freq_mhz: Signed, nonzero intermediate frequency in MHz, strictly
            inside the Nyquist interval. Zero IF is a DC measurement, not a
            beat-note amplitude with the factor-of-two convention used here.
        delay_us: Nonnegative transient-cut time in microseconds (default: 0.0).

    Returns:
        tuple: (complex_amplitude, frequency_array, fft_array)

    Raises:
        ValueError: For invalid samples, grid, IF, delay, insufficient retained
            data, rank-deficient/failed fit, or non-finite output. An empty
            transient window is never replaced by earlier samples.

    Fit signal = dc + a*cos(2*pi*f*tau) + b*sin(2*pi*f*tau), where tau is
    time relative to the first retained sample. Return a - 1j*b: a real
    cosine of amplitude A and phase phi returns A*exp(1j*phi). Complex
    traces retain the historical twice-the-signed-frequency-coefficient
    convention. The DC term avoids its leakage into the fitted IF.
    The FFT arrays remain unmodified diagnostic outputs; no nearest-bin
    fallback is used. Other tones, unresolved transients and noise can still
    bias the fit; this is not a multi-tone separation or noise model.
    """
    t_list = np.asarray(t_list)
    signal_trace = np.asarray(signal_trace)
    if t_list.ndim != 1 or signal_trace.ndim != 1:
        raise ValueError("Time and signal must be one-dimensional arrays")
    if t_list.size != signal_trace.size:
        raise ValueError("Time and signal must have the same number of samples")
    if not np.issubdtype(t_list.dtype, np.number) or np.iscomplexobj(t_list):
        raise ValueError("Time samples must be real numeric values")
    if not np.issubdtype(signal_trace.dtype, np.number):
        raise ValueError("Signal samples must be numeric values")
    if not np.all(np.isfinite(t_list)) or not np.all(np.isfinite(signal_trace)):
        raise ValueError("Time and signal samples must be finite")

    selected, t_slice, design = _if_fit_grid(t_list, if_freq_mhz, delay_us)
    signal_slice = signal_trace[selected]
    N = len(t_slice)
    if N < 10:  # Minimum points for meaningful FFT
        raise ValueError(f"Insufficient data points after delay ({N}). Increase simulation time or reduce delay.")

    dt = t_slice[1] - t_slice[0]
    yf = fft(signal_slice)
    xf = fftfreq(N, dt)
    if not np.all(np.isfinite(yf)) or not np.all(np.isfinite(xf)):
        raise ValueError("FFT output and frequency bins must be finite")
    # Use the requested signed frequency, not a quantized FFT-bin frequency.
    # Subtract in floating point to avoid integer overflow in relative time.
    try:
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design, signal_slice, rcond=None)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("IF sinusoidal fit failed") from exc
    if rank != 3:
        raise ValueError("IF sinusoidal fit is rank deficient; DC and IF are unresolved")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(singular_values)):
        raise ValueError("IF sinusoidal fit output must be finite")
    complex_amplitude = coefficients[1] - 1j * coefficients[2]
    if not np.isfinite(complex_amplitude):
        raise ValueError("IF fitted amplitude must be finite")

    return complex_amplitude, xf, yf

