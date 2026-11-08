"""DOA-ALG-001 — Bartlett / MVDR-Capon / MUSIC (synthetic, fast).

Deterministic NumPy-only coverage for the three covariance estimators on
top of the validated ArrayGeometry / steering_matrix / IFSnapshotData
X(M, K) stack.  No SimulationRunner, ARC, Doppler, Lindblad,
multiprocessing or full-pytest use.  All synthetic sources are generated
with the PROJECT steering_vector(); no second steering convention exists.

Data-polarity note: production snapshots carry x = exp(+j phi) while the
project steering function returns s = exp(-j phi).  The covariance cores
therefore evaluate the Hermitian form on the conjugate manifold
m = conj(s); the focused tests use the same production polarity.
"""
import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import (
    ArrayGeometry,
    BartlettEstimator,
    IFSnapshotData,
    MUSICEstimator,
    MVDREstimator,
    PhaseBeamformingEstimator,
    arrival_phases,
    get_doa_estimator,
    sample_covariance,
    steering_vector,
)

RF_FREQ_HZ = params.RF_FREQUENCY
TRUE_AOA_DEG = params.TARGET_AOA_DEG
SPACING_M = params.SPACING
IF_FREQ_HZ = params.IF_FREQ * 1.0e6
GRID_DEG = np.linspace(-90.0, 90.0, 181)


def _ula(n_elements):
    return ArrayGeometry.from_ula(n_elements=n_elements, spacing_m=SPACING_M)


def _production_response(geometry, azimuth_deg):
    """Production-polarity snapshot column x = exp(+j phi)."""
    return np.exp(1j * arrival_phases(geometry, RF_FREQ_HZ, azimuth_deg))


def _snapshots(geometry, azimuth_deg, n_snapshots, seed=0):
    """Deterministic K-snapshot block from one coherent source."""
    base = _production_response(geometry, azimuth_deg)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(n_snapshots))
    gains = rng.uniform(0.5, 1.5, size=int(n_snapshots))
    return base[:, None] * (gains * np.exp(1j * phases))[None, :]


def _context():
    return {"frequency_hz": RF_FREQ_HZ}


def _cov_estimators(n_sources=1):
    return {
        "bartlett": BartlettEstimator(),
        "mvdr": MVDREstimator(),
        "music": MUSICEstimator(n_sources=n_sources),
    }


def _local_peak_azimuths(grid, spectrum, count):
    """Deterministic strict-local-maximum peak picker (no SciPy needed)."""
    order = []
    for idx in range(1, spectrum.shape[0] - 1):
        if spectrum[idx] > spectrum[idx - 1] and spectrum[idx] >= spectrum[idx + 1]:
            order.append(idx)
    order.sort(key=lambda i: float(spectrum[i]), reverse=True)
    return np.array([float(grid[i]) for i in order[:count]])


def test_bartlett_single_source_ula():
    geometry = _ula(8 - 2)
    data = _production_response(geometry, TRUE_AOA_DEG)
    result = BartlettEstimator().estimate(data, geometry, _context())
    assert result.algorithm == "bartlett"
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG)
    assert result.scan_grid.shape == GRID_DEG.shape
    assert float(np.max(result.spectrum)) == pytest.approx(1.0)
    assert result.diagnostics["n_elements"] == geometry.m
    assert result.diagnostics["n_snapshots"] == 1
def test_music_single_source_ula_reports_subspace_diagnostics():
    geometry = _ula(8 - 2)
    data = _snapshots(geometry, TRUE_AOA_DEG, 32)
    result = MUSICEstimator(n_sources=1).estimate(data, geometry, _context())
    assert result.algorithm == "music"
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG)
    assert result.diagnostics["n_sources"] == 1
    assert result.diagnostics["eigenvalues"].shape == (geometry.m,)
    assert np.all(np.diff(result.diagnostics["eigenvalues"]) <= 0.0)
    assert result.diagnostics["covariance_rank"] == 1
    assert np.isfinite(result.diagnostics["covariance_condition_number"])


def test_mvdr_single_source_ula_reports_loading_diagnostics():
    geometry = _ula(8 - 2)
    data = _snapshots(geometry, TRUE_AOA_DEG, 32)
    estimator = MVDREstimator()
    result = estimator.estimate(data, geometry, _context())
    assert result.algorithm == "mvdr"
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG)
    assert result.diagnostics["loading_rel"] == pytest.approx(estimator.loading_rel)
    assert result.diagnostics["loading_absolute"] > 0.0
    assert result.diagnostics["rank_before_loading"] == 1
    assert np.isfinite(result.diagnostics["condition_after_loading"])
    assert (
        result.diagnostics["condition_after_loading"]
        < result.diagnostics["condition_before_loading"]
    )


def test_all_algorithms_agree_at_minus_thirty_degrees():
    geometry = _ula(8 - 2)
    data = _snapshots(geometry, TRUE_AOA_DEG, 32)
    legacy = PhaseBeamformingEstimator().estimate(
        _production_response(geometry, TRUE_AOA_DEG), geometry, _context())
    assert legacy.azimuth_deg == pytest.approx(TRUE_AOA_DEG)
    for name, estimator in _cov_estimators().items():
        result = estimator.estimate(data, geometry, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG), name


@pytest.mark.parametrize("m", [3, 8 - 2, 10])
def test_arbitrary_path_counts(m):
    geometry = _ula(m)
    data = _snapshots(geometry, TRUE_AOA_DEG, 32, seed=m)
    for name, estimator in _cov_estimators().items():
        result = estimator.estimate(data, geometry, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG), (name, m)
        assert result.scan_grid.shape == GRID_DEG.shape
        assert result.spectrum.shape == GRID_DEG.shape

def test_translated_geometry_invariance():
    geometry = _ula(8 - 2)
    offset = np.array([0.1234, -5.6789, 2.3456])
    moved = ArrayGeometry(geometry.positions_m + offset)
    data = _snapshots(geometry, TRUE_AOA_DEG, 24)
    moved_data = _snapshots(moved, TRUE_AOA_DEG, 24)
    for name, estimator in _cov_estimators().items():
        same = estimator.estimate(data, moved, _context())
        rebuilt = estimator.estimate(moved_data, moved, _context())
        assert same.azimuth_deg == pytest.approx(TRUE_AOA_DEG), name
        assert rebuilt.azimuth_deg == pytest.approx(TRUE_AOA_DEG), name


def test_nonuniform_linear_geometry():
    positions = np.zeros((5, 3))
    positions[:, 0] = np.array([0.0, 0.7, 1.9, 2.2, 4.1]) * SPACING_M
    geometry = ArrayGeometry(positions)
    data = _snapshots(geometry, TRUE_AOA_DEG, 48, seed=3)
    for name, estimator in _cov_estimators().items():
        result = estimator.estimate(data, geometry, _context())
        assert abs(float(result.azimuth_deg) - TRUE_AOA_DEG) <= 1.0, name


def test_planar_geometry_fixed_elevation():
    rng = np.random.default_rng(7)
    positions = np.column_stack([
        rng.uniform(-2.0, 2.0, size=7) * SPACING_M,
        rng.uniform(-2.0, 2.0, size=7) * SPACING_M,
        np.zeros(7),
    ])
    geometry = ArrayGeometry(positions)
    elevation = 10.0
    target = -20.0
    base = np.exp(
        1j * arrival_phases(geometry, RF_FREQ_HZ, target, elevation_deg=elevation))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=64)
    data = base[:, None] * np.exp(1j * phases)[None, :]
    for name in ("bartlett", "mvdr", "music"):
        estimator = get_doa_estimator(name, elevation_deg=elevation)
        result = estimator.estimate(data, geometry, _context())
        assert abs(float(result.azimuth_deg) - target) <= 1.0, name
        assert result.elevation_deg == pytest.approx(elevation)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_music_n_sources_ctor_validation(bad):
    with pytest.raises(ValueError):
        MUSICEstimator(n_sources=bad)


def test_music_n_sources_estimate_validation():
    geometry = _ula(8 - 2)
    data = _production_response(geometry, TRUE_AOA_DEG)
    estimator = MUSICEstimator(n_sources=1)
    estimator.n_sources = geometry.m
    with pytest.raises(ValueError):
        estimator.estimate(data, geometry, _context())


def test_music_two_sources():
    n_elem = 8
    geometry = _ula(n_elem)
    true_a = -20.0
    true_b = 30.0
    n_snap = 256
    noise_sigma = 0.1
    m1 = np.conjugate(steering_vector(geometry.positions_m, RF_FREQ_HZ, true_a))
    m2 = np.conjugate(steering_vector(geometry.positions_m, RF_FREQ_HZ, true_b))
    mixing = np.column_stack([m1, m2])
    rng = np.random.default_rng(1)
    symbols = (rng.standard_normal((2, n_snap))
               + 1j * rng.standard_normal((2, n_snap))) / np.sqrt(2.0)
    noise = noise_sigma * (rng.standard_normal((n_elem, n_snap))
                           + 1j * rng.standard_normal((n_elem, n_snap))) / np.sqrt(2.0)
    data = mixing @ symbols + noise
    result = MUSICEstimator(n_sources=2).estimate(data, geometry, _context())
    peaks = _local_peak_azimuths(result.scan_grid, result.spectrum, 2)
    assert peaks.shape == (2,)
    assert min(abs(float(p) - true_a) for p in peaks) <= 1.0
    assert min(abs(float(p) - true_b) for p in peaks) <= 1.0
    assert result.diagnostics["n_sources"] == 2
    assert result.diagnostics["covariance_rank"] == n_elem


def test_mvdr_singular_covariance_loading_path():
    geometry = _ula(8 - 2)
    data = _production_response(geometry, TRUE_AOA_DEG)
    rxx = sample_covariance(np.asarray(data).reshape(geometry.m, 1))
    assert int(np.linalg.matrix_rank(rxx)) == 1
    assert float(np.linalg.cond(rxx)) > 1.0e12
    result = MVDREstimator().estimate(data, geometry, _context())
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG)
    assert result.diagnostics["rank_before_loading"] == 1
    assert np.isfinite(result.diagnostics["condition_after_loading"])


def test_mvdr_zero_trace_fails_loudly():
    geometry = _ula(8 - 2)
    data = np.zeros((geometry.m, 4), dtype=complex)
    with pytest.raises(Exception):
        MVDREstimator().estimate(data, geometry, _context())


def test_deterministic_repeatability():
    geometry = _ula(8 - 2)
    data = _snapshots(geometry, TRUE_AOA_DEG, 24)
    for name, estimator in _cov_estimators().items():
        first = estimator.estimate(data, geometry, _context())
        second = estimator.estimate(data, geometry, _context())
        np.testing.assert_array_equal(first.spectrum, second.spectrum)
        assert first.azimuth_deg == second.azimuth_deg


def test_registry_names_and_capon_alias():
    expected = {
        "phase_beamforming": "phase_beamforming",
        "bartlett": "bartlett",
        "mvdr": "mvdr",
        "capon": "mvdr",
        "music": "music",
    }
    for name, algorithm in expected.items():
        assert get_doa_estimator(name).algorithm == algorithm
    assert isinstance(get_doa_estimator("capon"), MVDREstimator)
    with pytest.raises(ValueError):
        get_doa_estimator("definitely_not_registered")
    # DOA-ALG-002: the search-free estimators are now registered.
    assert get_doa_estimator("esprit").algorithm == "esprit"
    assert get_doa_estimator("root_music").algorithm == "root_music"


def test_rf_carrier_contract_for_covariance_estimators():
    geometry = _ula(8 - 2)
    data = _production_response(geometry, TRUE_AOA_DEG)
    for name, estimator in _cov_estimators().items():
        result_rf = estimator.estimate(data, geometry, _context())
        assert result_rf.azimuth_deg == pytest.approx(TRUE_AOA_DEG), name
        result_if = estimator.estimate(data, geometry, {"frequency_hz": IF_FREQ_HZ})
        assert result_if.azimuth_deg != pytest.approx(TRUE_AOA_DEG), name

def test_snapshot_container_input_and_scan_resolution():
    geometry = _ula(8 - 2)
    data = _snapshots(geometry, TRUE_AOA_DEG, 16)
    snap = IFSnapshotData(
        X=data,
        time_centers_us=np.arange(data.shape[1], dtype=float),
        if_frequency_mhz=params.IF_FREQ,
        n_paths=geometry.m,
        window_duration_us=1.0,
        samples_per_window=10,
    )
    coarse = np.linspace(-90.0, 90.0, 37)
    cases = (
        ("bartlett", BartlettEstimator(scan_azimuth_deg=coarse)),
        ("mvdr", MVDREstimator(scan_azimuth_deg=coarse)),
        ("music", MUSICEstimator(scan_azimuth_deg=coarse, n_sources=1)),
    )
    for name, estimator in cases:
        result = estimator.estimate(snap, geometry, _context())
        assert result.scan_grid.shape == (37,)
        assert result.spectrum.shape == (37,)
        assert abs(float(result.azimuth_deg) - TRUE_AOA_DEG) <= 2.5, name

