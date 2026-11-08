"""DOA-DATA-001 — complex IF snapshot extraction (synthetic, fast).

Validates the snapshot layer WITHOUT physics, ARC, Doppler or any
simulation runner.  The convention under test is traced from the production
chain:

    time-domain worker (Im<sigma21>(t)) ->
    extract_if_component (FFT-bin (yf[N]/N)*2 at +f_IF, numpy sign) ->
    runner.extract_phases (unwrap(angle(...)))

For a real observable s(t) = A*cos(2*pi*f*t + phi) with f in MHz and t in
us, the complex IF coefficient is Z = A*exp(+j*phi): the coherent
projection (2/N)*sum([s - mean]*exp(-j*2*pi*f_IF*t)) must reproduce the
production FFT-bin coefficient to numerical precision for windows spanning
an integer number of IF cycles.
"""

import inspect
import re

import numpy as np
import pytest

from rydberg_aoa.config.parameters import ATOMIC_TRANSIENT_DELAY_US
from rydberg_aoa.doa import (
    IFSnapshotData,
    extract_if_snapshots,
    sample_covariance,
)
from rydberg_aoa.doa import snapshots as snapshots_module
from rydberg_aoa.utils.numerical import extract_if_component as production_if_extractor

# Synthetic waveform units (f in MHz, t in us): NO 1e6 factor anywhere.
F_IF_MHZ = 0.1              # 0.1 MHz == 0.1 cycles/us == 10 us period
DT_US = 0.1                 # exactly 100 samples per IF cycle
DEG_TO_RAD = np.pi / 180.0
TWO_PI = 2.0 * np.pi


def _time_grid(n_cycles, f=F_IF_MHZ, dt=DT_US, start=0.0):
    """Uniform grid with an exact integer number of IF cycles."""
    n = max(1, int(round(n_cycles / (f * dt))))
    return start + dt * np.arange(n, dtype=float)


def _traces(m, phases_rad, amps=None, n_cycles=8, f=F_IF_MHZ, dt=DT_US):
    """Real traces s_m(t) = A_m * cos(2*pi*f*t + phi_m)."""
    t = _time_grid(n_cycles, f=f, dt=dt)
    if amps is None:
        amps = np.ones(m)
    amps = np.broadcast_to(np.asarray(amps, dtype=float), (m,))
    traces = []
    for i in range(m):
        traces.append(amps[i] * np.cos(TWO_PI * f * t + phases_rad[i]))
    return t, traces


def _angle_diff_abs(a, b):
    """Smallest absolute circular difference between two angle arrays."""
    return np.abs(np.angle(np.exp(1j * (a - b))))


# --- A. M dynamic -----------------------------------------------------------
@pytest.mark.parametrize("m", [3, 6, 10])
def test_m_dynamic_shapes_and_k(m):
    t, traces = _traces(m, phases_rad=np.zeros(m), n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    assert snap.X.shape == (m, 4)          # 8 cycles / 2-cycle windows
    assert snap.time_centers_us.shape == (4,)
    assert snap.n_paths == m
    assert snap.n_snapshots == 4
    assert snap.if_frequency_mhz == F_IF_MHZ
    assert snap.samples_per_window == 200
    assert snap.window_duration_us == pytest.approx(199 * DT_US)


def test_single_path_supported():
    t, (trace,) = _traces(1, phases_rad=[0.0], n_cycles=8)
    snap = extract_if_snapshots(
        t, [trace], F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    assert snap.X.shape == (1, 4)


def test_arbitrary_k_single_window():
    t, traces = _traces(4, phases_rad=np.zeros(4), n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=8
    )
    assert snap.X.shape == (4, 1)          # K = 1 supported


# --- B. known phase progression ----------------------------------------------
def test_known_phase_progression_minus_22_5_deg_per_path():
    m = 6                                   # test uses six paths on purpose
    phases = np.arange(m) * (-22.5 * DEG_TO_RAD)
    t, traces = _traces(m, phases_rad=phases, n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    x_bar = snap.X.mean(axis=1)             # coherent mean snapshot (M,)
    rec = np.unwrap(np.angle(x_bar))
    rec = rec - rec[0]                      # common global offset removed
    expected = phases - phases[0]
    np.testing.assert_allclose(rec, expected, rtol=0.0, atol=1.0e-9)
    progression = np.rad2deg(np.diff(rec))
    np.testing.assert_allclose(progression, -22.5, rtol=0.0, atol=1.0e-9)
# --- C. arbitrary per-path phases --------------------------------------------
@pytest.mark.parametrize(
    "phases_deg",
    [
        [0.0, 17.0, -43.0, 88.0, 121.0, -179.0],
        [45.0, 45.0, 45.0, 45.0],
        [-120.0, 60.0, 0.0, 150.0, -30.0],
    ],
)
def test_arbitrary_phases_recovered_up_to_common_offset(phases_deg):
    m = len(phases_deg)
    phases = np.deg2rad(np.asarray(phases_deg, dtype=float))
    t, traces = _traces(m, phases_rad=phases, n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    x_bar = snap.X.mean(axis=1)
    rec = np.unwrap(np.angle(x_bar))
    rel = rec - rec[0]
    expected_rel = phases - phases[0]
    assert np.max(_angle_diff_abs(rel, expected_rel)) < 1.0e-9


# --- D. amplitude tracking ---------------------------------------------------
def test_amplitudes_tracked_by_snapshot_magnitudes():
    amps = np.array([1.0, 2.0, 0.5, 3.0])
    m = amps.shape[0]
    phases = np.arange(m) * 0.3
    t, traces = _traces(m, phases_rad=phases, amps=amps, n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    # per-window magnitudes track the true amplitudes exactly
    expected = np.broadcast_to(amps[:, None], snap.X.shape)  # (M, K)
    np.testing.assert_allclose(
        np.abs(snap.X), expected, rtol=0.0, atol=1.0e-9
    )
# --- E. multiple windows ------------------------------------------------------
def test_multiple_windows_shape_contract():
    m = 6
    phases = np.arange(m) * (-22.5 * DEG_TO_RAD)
    t, traces = _traces(m, phases_rad=phases, n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=1
    )
    assert snap.X.shape == (m, 8)          # eight 1-cycle windows
    assert snap.time_centers_us.shape == (8,)
    # time centers must cover the record and be ordered
    assert snap.time_centers_us[0] >= 0.0
    assert snap.time_centers_us[-1] <= t[-1]
    assert np.all(np.diff(snap.time_centers_us) > 0.0)


# --- F. covariance -----------------------------------------------------------
def test_sample_covariance_hermitian_finite_psd():
    m = 6
    phases = np.arange(m) * (-22.5 * DEG_TO_RAD)
    amps = [1.0, 2.0, 1.5, 0.5, 3.0, 1.2]
    t, traces = _traces(m, phases_rad=phases, amps=amps, n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=1
    )
    rxx = sample_covariance(snap.X)
    assert rxx.shape == (m, m)
    assert np.all(np.isfinite(rxx))
    np.testing.assert_allclose(rxx, rxx.conj().T, rtol=0.0, atol=1.0e-12)
    symmetrized = 0.5 * (rxx + rxx.conj().T)
    eigenvalues = np.linalg.eigvalsh(symmetrized)
    assert np.all(eigenvalues >= -1.0e-10)


# --- G. global phase invariance -----------------------------------------------
def test_global_phase_invariance():
    m = 5
    phases = np.arange(m) * 30.0 * DEG_TO_RAD
    t, traces = _traces(m, phases_rad=phases, n_cycles=8)
    theta = 137.0 * DEG_TO_RAD
    rotated = [np.asarray(tr, dtype=complex) * np.exp(1j * theta) for tr in traces]

    plain = extract_if_snapshots(t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2)
    rot = extract_if_snapshots(t, rotated, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2)

    rel_plain = np.unwrap(np.angle(plain.X.mean(axis=1)))
    rel_rot = np.unwrap(np.angle(rot.X.mean(axis=1)))
    rel_plain -= rel_plain[0]
    rel_rot -= rel_rot[0]
    assert np.max(_angle_diff_abs(rel_plain, rel_rot)) < 1.0e-9
    # covariance is exactly invariant under one common global phase
    np.testing.assert_allclose(
        sample_covariance(plain.X), sample_covariance(rot.X), atol=1.0e-12
    )


# --- H. unit contract: t_us and f_MHz, no 1e6 ---------------------------------
@pytest.mark.parametrize("phase_rad", [0.37, 2.1, -1.9])
def test_unit_contract_us_mhz_no_scale_factor(phase_rad):
    f = 0.1          # MHz
    dt = 0.1         # us
    amplitude = 1.3
    t = dt * np.arange(200, dtype=float)      # two full IF cycles
    s = amplitude * np.cos(TWO_PI * f * t + phase_rad)
    snap = extract_if_snapshots(t, [s], f, start_us=0.0, cycles_per_snapshot=2)
    recovered = np.angle(snap.X[0, 0])
    assert np.abs(np.angle(np.exp(1j * (recovered - phase_rad)))) < 1.0e-9
    assert abs(abs(snap.X[0, 0]) - amplitude) < 1.0e-9
# --- parity with the production FFT extractor --------------------------------
def test_snapshot_parity_with_production_fft_extractor():
    m = 6
    phases = np.arange(m) * (-22.5 * DEG_TO_RAD)
    amps = np.linspace(1.0, 2.0, m)
    t, traces = _traces(m, phases_rad=phases, amps=amps, n_cycles=8)
    production = np.array(
        [production_if_extractor(np.asarray(tr), t, F_IF_MHZ, 0.0)[0]
         for tr in traces]
    )  # (M,) complex, bin-aligned FFT at +f_IF == exact projection
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    x_bar = snap.X.mean(axis=1)             # (M,)
    np.testing.assert_allclose(x_bar, production, rtol=1.0e-9, atol=1.0e-12)


# --- transient default follows production -------------------------------------
def test_default_start_us_reuses_production_transient_cut():
    t, traces = _traces(4, phases_rad=np.zeros(4), n_cycles=16)
    snap = extract_if_snapshots(t, traces, F_IF_MHZ, cycles_per_snapshot=2)
    assert snap.diagnostics["start_us"] == pytest.approx(ATOMIC_TRANSIENT_DELAY_US)
    assert snap.time_centers_us.min() >= ATOMIC_TRANSIENT_DELAY_US - 1.0e-12
    assert snap.samples_per_window == 200


# --- windowing contracts -------------------------------------------------------
def test_trailing_partial_window_never_used():
    m = 3
    t = DT_US * np.arange(850, dtype=float)   # 8.5 IF cycles of data
    traces = [np.cos(TWO_PI * F_IF_MHZ * t) for _ in range(m)]
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=1
    )
    assert snap.X.shape == (m, 8)             # full 1-cycle windows only
    assert snap.diagnostics["trailing_samples_dropped"] == 50
    assert snap.diagnostics["n_samples_selected"] == 850


def test_rejects_windows_with_too_few_samples():
    # dt = 4 us -> ~2.5 samples per IF cycle -> 2-cycle window far below 8
    t = np.arange(0.0, 12.0, 4.0)
    s = np.cos(TWO_PI * 0.1 * t)
    with pytest.raises(ValueError, match="minimum"):
        extract_if_snapshots(t, [s], 0.1, start_us=0.0, cycles_per_snapshot=2)


def test_rejects_record_too_short_for_one_full_window():
    # enough samples per cycle but record shorter than one 2-cycle window
    t = np.linspace(0.0, 5.0, 51)             # 5 us at dt=0.1 -> 50 < 200
    s = np.cos(TWO_PI * 0.1 * t)
    with pytest.raises(ValueError, match="Too few samples"):
        extract_if_snapshots(t, [s], 0.1, start_us=0.0, cycles_per_snapshot=2)


def test_rejects_stop_before_start():
    t, traces = _traces(2, phases_rad=np.zeros(2), n_cycles=8)
    with pytest.raises(ValueError, match="stop_us"):
        extract_if_snapshots(
            t, traces, F_IF_MHZ, start_us=20.0, stop_us=10.0
        )


def test_rejects_non_uniform_time_grid():
    t = np.linspace(0.0, 80.0, 801)
    t = np.concatenate([t[:400], t[400:] + 5.0])   # jump at sample 400
    s = np.cos(TWO_PI * 0.1 * t)
    with pytest.raises(ValueError, match="uniform"):
        extract_if_snapshots(t, [s], F_IF_MHZ, start_us=0.0)


# --- container contract --------------------------------------------------------
def test_snapshot_data_container_is_immutable_and_validated():
    t, traces = _traces(4, phases_rad=np.zeros(4), n_cycles=8)
    snap = extract_if_snapshots(
        t, traces, F_IF_MHZ, start_us=0.0, cycles_per_snapshot=2
    )
    with pytest.raises(AttributeError):       # frozen dataclass
        snap.X = np.zeros((1, 1), dtype=complex)

    with pytest.raises(ValueError):
        IFSnapshotData(
            X=np.zeros((2, 3), dtype=complex),
            time_centers_us=np.zeros(2),      # length mismatch (needs 3)
            if_frequency_mhz=0.1,
            n_paths=2,
            window_duration_us=1.0,
            samples_per_window=10,
        )
    with pytest.raises(ValueError):
        IFSnapshotData(
            X=np.zeros((2, 3), dtype=complex),
            time_centers_us=np.zeros(3),
            if_frequency_mhz=0.1,
            n_paths=7,                        # n_paths != X.shape[0]
            window_duration_us=1.0,
            samples_per_window=10,
        )


# --- I. no hard-coded six ------------------------------------------------------
def test_no_hardcoded_six_in_snapshot_extraction():
    source = inspect.getsource(snapshots_module.extract_if_snapshots)
    assert not re.search(r"(shape\s*==\s*\(?\s*6\b|\[\s*6\s*\]|range\(\s*6\s*\)|==\s*6\b|!=\s*6\b)", source)
    assert "n_paths" in source or "n_avail" in source