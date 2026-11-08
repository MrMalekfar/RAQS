"""Reject corrupted IF/covariance data before a direction estimator consumes it."""

import warnings

import numpy as np
import pytest

from rydberg_aoa.doa.snapshots import (
    IFSnapshotData, extract_if_snapshots, sample_covariance,
)


def _container(**changes):
    values = dict(X=np.ones((2, 1), complex), time_centers_us=[1.0],
                  if_frequency_mhz=0.1, n_paths=2,
                  window_duration_us=1.0, samples_per_window=10)
    return IFSnapshotData(**(values | changes))


def _record():
    t = 0.1 * np.arange(1000)
    return t, [np.cos(2 * np.pi * 0.1 * t + 0.4)]


@pytest.mark.parametrize("bad", [True, np.bool_(True), "0.1", b"0.1",
                                     0.1j, np.complex128(0.1 + 2j),
                                     np.array(0.1), np.array([0.1])])
def test_if_type_is_checked_before_conversion_in_both_entry_points(bad):
    t, traces = _record()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="if_frequency_mhz"):
            _container(if_frequency_mhz=bad)
        with pytest.raises(ValueError, match="if_frequency_mhz"):
            extract_if_snapshots(t, traces, bad, start_us=0)


@pytest.mark.parametrize("field", ["n_paths", "samples_per_window"])
@pytest.mark.parametrize("bad", [True, np.bool_(True), 2.8, "2", 2 + 0j,
                                     np.array(2), np.nan, np.inf])
def test_container_counts_cannot_be_coerced_or_truncated(field, bad):
    with pytest.raises(ValueError, match=field):
        _container(**{field: bad})


@pytest.mark.parametrize("field,bad", [
    ("window_duration_us", "1.0"), ("window_duration_us", True),
    ("time_centers_us", [1 + 2j]), ("time_centers_us", ["1.0"]),
    ("X", [["1"], ["2"]]), ("X", [[True], [False]]),
    ("X", [[np.nan], [1]]), ("X", [[1], [np.inf]]),
])
def test_container_rejects_invalid_arrays_and_duration_without_warnings(field, bad):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=field):
            _container(**{field: bad})


@pytest.mark.parametrize("field,bad", [
    ("cycles_per_snapshot", "2"), ("cycles_per_snapshot", True),
    ("start_us", "0"), ("stop_us", 100 + 0j),
    ("min_samples", 8.7), ("min_samples", True),
    ("remove_dc", "False"), ("remove_dc", 1),
])
def test_extractor_rejects_scalar_coercion(field, bad):
    t, traces = _record()
    kwargs = dict(start_us=0)
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        extract_if_snapshots(t, traces, 0.1, **kwargs)


@pytest.mark.parametrize("kind", ["complex_time", "string_time", "string_trace"])
def test_extractor_rejects_lossy_or_non_numeric_arrays(kind):
    t, traces = _record()
    if kind == "complex_time":
        t = t.astype(complex) + 1j
    elif kind == "string_time":
        t = t.astype(str)
    else:
        traces = [traces[0].astype(str)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="t_us|trace"):
            extract_if_snapshots(t, traces, 0.1, start_us=0)


@pytest.mark.parametrize("frequency", [5.0, -5.0, 6.0, -6.0])
def test_aliasing_is_rejected_even_when_many_cycles_form_a_window(frequency):
    t, traces = _record()
    with pytest.raises(ValueError, match="Nyquist"):
        extract_if_snapshots(t, traces, frequency, start_us=0, cycles_per_snapshot=20)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e200])
def test_covariance_rejects_nonfinite_input_or_overflow(bad):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite|overflow"):
            sample_covariance(np.full((2, 3), bad))


@pytest.mark.parametrize("bad", [[["1", "2"]], [[True, False]]])
def test_covariance_does_not_accept_non_numeric_coercion(bad):
    with pytest.raises(ValueError, match="X"):
        sample_covariance(bad)


def test_projection_overflow_is_visible_without_invalid_snapshots():
    t, traces = _record()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="projection|finite|overflow"):
            extract_if_snapshots(t, [np.full_like(traces[0], 1e308)], 0.1, start_us=0)


@pytest.mark.parametrize("frequency,cycles", [(1e-320, 2.), (0.1, 1e308)])
def test_nonfinite_derived_window_geometry_is_rejected(frequency, cycles):
    t, traces = _record()
    with pytest.raises(ValueError, match="geometry.*finite"):
        extract_if_snapshots(t, traces, frequency, start_us=0, cycles_per_snapshot=cycles)


def test_finite_times_with_overflowing_difference_are_rejected():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite.*steps"):
            extract_if_snapshots([-1e308, 1e308], [[0., 1.]], 0.1, start_us=0)


@pytest.mark.parametrize("field", ["if_frequency_mhz", "window_duration_us"])
def test_scalar_conversion_overflow_preserves_cause(field):
    with pytest.raises(ValueError, match=field) as error:
        _container(**{field: 10**1000})
    assert isinstance(error.value.__cause__, OverflowError)


def test_valid_numpy_scalars_and_covariance_are_preserved_without_mutation():
    x = np.array([[1 + 2j, -3j], [4, 5 - 1j]])
    before = x.copy()
    snapshots = IFSnapshotData(
        X=x, time_centers_us=np.array([1., 2.]),
        if_frequency_mhz=np.float32(-0.1), n_paths=np.int64(2),
        window_duration_us=np.float64(1.), samples_per_window=np.int32(10))
    assert snapshots.if_frequency_mhz < 0
    expected = x @ x.conj().T / x.shape[1]
    np.testing.assert_array_equal(sample_covariance(x), expected)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(snapshots.X, before)


def test_zero_covariance_is_valid_data_without_an_invented_floor():
    np.testing.assert_array_equal(sample_covariance(np.zeros((2, 3))), np.zeros((2, 2)))
