"""DOA-ALG-002 — ESPRIT / Root-MUSIC on structured arrays (synthetic, fast).

Deterministic NumPy-only coverage for the two search-free estimators added
on top of the validated ArrayGeometry / steering / IFSnapshotData stack.
No SimulationRunner, ARC, Doppler, Lindblad, multiprocessing or full-pytest
use.  All synthetic sources are generated with the PROJECT
arrival_phases() production polarity x = exp(+j phi); no second steering
convention exists.

Geometry-capability contract exercised here:
  * ESPRIT  requires capabilities {"shift_invariant"} (translational
    invariance) and a valid ShiftInvariantPair with 1 <= n_sources < pair size.
  * Root-MUSIC requires capabilities {"ula"} (Vandermonde manifold).
Incompatible geometries must raise IncompatibleGeometryError / DOAError —
never a silent fallback to a scan-based method.
"""
import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import (
    ArrayGeometry,
    BartlettEstimator,
    DOAError,
    ESPRITEstimator,
    IFSnapshotData,
    IncompatibleGeometryError,
    RootMUSICEstimator,
    arrival_phases,
    get_doa_estimator,
)

RF_FREQ_HZ = params.RF_FREQUENCY
TRUE_AOA_DEG = params.TARGET_AOA_DEG          # -30.0 (frozen production target)
SPACING_M = params.SPACING                    # lambda_RF / 8
SPEED_OF_LIGHT = 299792458.0
WAVELENGTH_M = SPEED_OF_LIGHT / RF_FREQ_HZ
TIGHT = 1.0e-4                                # deg, noiseless synthetic data


def _ula(n_elements, spacing_m=SPACING_M):
    return ArrayGeometry.from_ula(n_elements=n_elements, spacing_m=spacing_m)


def _production_response(geometry, azimuth_deg):
    """Production-polarity snapshot column x = exp(+j phi)."""
    phi = arrival_phases(geometry, RF_FREQ_HZ, azimuth_deg)
    return np.exp(1j * np.asarray(phi))


def _column(geometry, azimuth_deg):
    return _production_response(geometry, azimuth_deg).reshape(geometry.m, 1)


def _context():
    return {"frequency_hz": RF_FREQ_HZ}


def _two_source_snapshots(geometry, azimuths_deg, n_snapshots=256,
                          seed=20260913, noise_std=1.0e-3):
    """Deterministic K-snapshot block: two coherent sources + tiny noise.

    The noise is TEST STIMULUS ONLY (identifiability of a rank-2 subspace);
    no project physical noise modeling is involved.
    """
    rng = np.random.default_rng(seed)
    m = geometry.m
    x = np.zeros((m, int(n_snapshots)), dtype=complex)
    for az in azimuths_deg:
        base = _production_response(geometry, az)
        gains = rng.uniform(0.7, 1.3, size=int(n_snapshots))
        phis = rng.uniform(0.0, 2.0 * np.pi, size=int(n_snapshots))
        x += base[:, None] * (gains * np.exp(1j * phis))[None, :]
    if noise_std > 0.0:
        x += noise_std * (
            rng.standard_normal((m, int(n_snapshots)))
            + 1j * rng.standard_normal((m, int(n_snapshots)))
        ) / np.sqrt(2.0)
    return x


def _nonuniform_arbitrary():
    return ArrayGeometry(np.array([
        [0.0, 0.0, 0.0],
        [0.7 * SPACING_M, 0.0, 0.0],
        [1.9 * SPACING_M, 0.1 * SPACING_M, 0.0],
        [2.6 * SPACING_M, 0.2 * SPACING_M, 0.0],
    ]))


def _unequal_spacing_linear():
    """Collinear but nonuniform: unequal translated-subarray spacing."""
    return ArrayGeometry(np.array([
        [0.0, 0.0, 0.0],
        [SPACING_M, 0.0, 0.0],
        [2.0 * SPACING_M, 0.0, 0.0],
        [3.5 * SPACING_M, 0.0, 0.0],
    ]))


def _ring_geometry():
    angles = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    return ArrayGeometry(np.column_stack([
        np.cos(angles), np.sin(angles), np.zeros(7)]))


TWO_SRC_TRUE = (-35.0, 15.0)

# --- 1/2. current frozen ULA, single source -----------------------------------
def test_esprit_current_ula_minus30():
    geometry = _ula(6)
    result = ESPRITEstimator().estimate(_column(geometry, TRUE_AOA_DEG),
                                        geometry, _context())
    assert result.algorithm == "esprit"
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)
    assert result.spectrum is None and result.scan_grid is None
    diag = result.diagnostics
    # lambda/8 ULA at |az| = 30 deg, elevation 0:
    # |psi| = 2*pi*(1/8)*sin(30 deg) = pi/8
    assert diag["spatial_phase_shifts"][0] == pytest.approx(-np.pi / 8, rel=1e-6)
    assert diag["signal_subspace_rank"] == 1
    assert diag["n_sources"] == 1
    assert diag["all_azimuths_deg"][0] == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)


def test_root_music_current_ula_minus30():
    geometry = _ula(6)
    result = RootMUSICEstimator().estimate(_column(geometry, TRUE_AOA_DEG),
                                           geometry, _context())
    assert result.algorithm == "root_music"
    assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)
    diag = result.diagnostics
    assert diag["ula_spacing_m"] == pytest.approx(SPACING_M, rel=1e-9)
    assert np.allclose(diag["ula_axis"], [1.0, 0.0, 0.0])
    assert len(diag["selected_roots"]) == 1
    assert diag["root_magnitudes"][0] == pytest.approx(1.0, abs=1e-6)


# --- 3. path-count sweep: no hard-coded six -----------------------------------
@pytest.mark.parametrize("n_elements", [4, 6, 10])
def test_path_counts_m4_m6_m10(n_elements):
    geometry = _ula(n_elements)
    data = _column(geometry, TRUE_AOA_DEG)
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(data, geometry, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT), (
            estimator.algorithm, n_elements)


# --- 4. translated ULA invariance ---------------------------------------------
def test_translated_ula_invariance():
    reference = _ula(6)
    shifted = ArrayGeometry.from_ula(
        n_elements=6, spacing_m=SPACING_M, offset_xyz=(3.1, -2.7, 5.0))
    assert shifted.capabilities["shift_invariant"] is True
    pair_ref = reference.shift_invariant_pair
    pair_new = shifted.shift_invariant_pair
    assert np.allclose(pair_new.displacement_m, pair_ref.displacement_m)
    assert pair_new.indices_1 == pair_ref.indices_1
    assert pair_new.indices_2 == pair_ref.indices_2
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(_column(shifted, TRUE_AOA_DEG),
                                    shifted, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT), (
            estimator.algorithm)


# --- 5. rotated ULA (3D-relevant): actual displacement vector must be used ----
def test_rotated_ula_esprit_uses_actual_displacement():
    axis_deg = 40.0
    axis = np.array([np.cos(np.deg2rad(axis_deg)),
                     np.sin(np.deg2rad(axis_deg)), 0.0])
    positions = np.array([m * SPACING_M * axis for m in range(6)])
    geometry = ArrayGeometry(positions)
    caps = geometry.capabilities
    assert caps["ula"] is True and caps["shift_invariant"] is True
    pair = geometry.shift_invariant_pair
    assert np.allclose(pair.displacement_m, SPACING_M * axis, atol=1e-12)
    assert pair.indices_1 == tuple(range(5))
    assert pair.indices_2 == tuple(range(1, 6))
    data = _column(geometry, TRUE_AOA_DEG)
    esprit = ESPRITEstimator().estimate(data, geometry, _context())
    assert esprit.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)
    # Root-MUSIC option A: support rotated ULA via its own ULA axis.
    rmusic = RootMUSICEstimator().estimate(data, geometry, _context())
    assert rmusic.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=1e-3)


# --- 6/7. geometry rejection (fail loudly, no fallback) ------------------------
@pytest.mark.parametrize("bad_geometry", [
    _nonuniform_arbitrary(), _unequal_spacing_linear(), _ring_geometry()])
def test_esprit_rejects_incompatible_geometry(bad_geometry):
    assert bad_geometry.capabilities["shift_invariant"] is False
    data = _column(bad_geometry, TRUE_AOA_DEG)
    with pytest.raises(IncompatibleGeometryError):
        ESPRITEstimator().estimate(data, bad_geometry, _context())


@pytest.mark.parametrize("bad_geometry", [
    _nonuniform_arbitrary(), _unequal_spacing_linear(), _ring_geometry()])
def test_root_music_rejects_non_ula(bad_geometry):
    assert bad_geometry.capabilities["ula"] is False
    data = _column(bad_geometry, TRUE_AOA_DEG)
    with pytest.raises(IncompatibleGeometryError):
        RootMUSICEstimator().estimate(data, bad_geometry, _context())


# --- 8. n_sources validation ---------------------------------------------------
@pytest.mark.parametrize("bad_n", [0, -1, 2.5])
def test_n_sources_constructor_validation(bad_n):
    with pytest.raises(ValueError):
        ESPRITEstimator(n_sources=bad_n)
    with pytest.raises(ValueError):
        RootMUSICEstimator(n_sources=bad_n)


def test_n_sources_too_large_for_subarray():
    geometry = _ula(6)
    data = _column(geometry, TRUE_AOA_DEG)
    # ESPRIT subarray size is M-1 = 5: n_sources >= 5 must fail loudly.
    for n_sources in (5, 6):
        with pytest.raises(ValueError):
            ESPRITEstimator(n_sources=n_sources).estimate(
                data, geometry, _context())
    # Root-MUSIC operates on the full M = 6: n_sources >= 6 fails loudly.
    with pytest.raises(ValueError):
        RootMUSICEstimator(n_sources=6).estimate(data, geometry, _context())


def test_m_too_small_fails_loudly():
    tiny = _ula(2)
    data = _column(tiny, TRUE_AOA_DEG)
    with pytest.raises(IncompatibleGeometryError):
        ESPRITEstimator().estimate(data, tiny, _context())
    with pytest.raises(IncompatibleGeometryError):
        RootMUSICEstimator().estimate(data, tiny, _context())


# --- 9/10. two-source recovery -------------------------------------------------
def test_two_source_esprit():
    geometry = _ula(8)
    data = _two_source_snapshots(geometry, TWO_SRC_TRUE)
    result = ESPRITEstimator(n_sources=2).estimate(data, geometry, _context())
    got = np.sort(np.asarray(result.diagnostics["all_azimuths_deg"], dtype=float))
    assert got.shape == (2,)
    np.testing.assert_allclose(got, np.sort(TWO_SRC_TRUE), atol=0.5)


def test_two_source_root_music():
    geometry = _ula(8)
    data = _two_source_snapshots(geometry, TWO_SRC_TRUE)
    result = RootMUSICEstimator(n_sources=2).estimate(data, geometry, _context())
    got = np.sort(np.asarray(result.diagnostics["all_azimuths_deg"], dtype=float))
    assert got.shape == (2,)
    np.testing.assert_allclose(got, np.sort(TWO_SRC_TRUE), atol=0.5)
    assert len(result.diagnostics["selected_roots"]) == 2


# --- 11. deterministic repeatability ------------------------------------------
def test_deterministic_repeatability():
    geometry = _ula(6)
    data = _two_source_snapshots(geometry, TWO_SRC_TRUE, n_snapshots=64)
    for estimator in (ESPRITEstimator(n_sources=2),
                      RootMUSICEstimator(n_sources=2)):
        first = estimator.estimate(data, geometry, _context())
        second = type(estimator)(n_sources=2).estimate(
            data, geometry, _context())
        assert first.azimuth_deg == second.azimuth_deg
        np.testing.assert_array_equal(
            np.asarray(first.diagnostics["all_azimuths_deg"], dtype=float),
            np.asarray(second.diagnostics["all_azimuths_deg"], dtype=float))


# --- 12. no scan-grid dependency ----------------------------------------------
def test_no_scan_grid_dependency():
    geometry = _ula(6)
    off_grid = TRUE_AOA_DEG - 0.5      # not representable on a 1-degree grid
    data = _column(geometry, off_grid)
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(data, geometry, _context())
        assert result.spectrum is None
        assert result.scan_grid is None
        # Sub-degree accuracy proves no 1-degree scan grid is in play.
        assert result.azimuth_deg == pytest.approx(off_grid, abs=TIGHT)
    # Search-free estimators must reject any scan grid explicitly.
    with pytest.raises(ValueError):
        ESPRITEstimator().scan_azimuth_deg = np.linspace(-90.0, 90.0, 181)
    with pytest.raises(ValueError):
        RootMUSICEstimator().scan_azimuth_deg = np.linspace(-90.0, 90.0, 181)


# --- 13. lambda/8 spacing handled with the actual carrier ----------------------
def test_lambda_over_eight_spacing_phase_contract():
    geometry = _ula(6)
    assert SPACING_M == pytest.approx(WAVELENGTH_M / 8.0, rel=1e-9)
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(_column(geometry, TRUE_AOA_DEG),
                                    geometry, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)
        assert result.diagnostics["frequency_hz"] == pytest.approx(RF_FREQ_HZ)
    # +30 deg must mirror the phase sign of -30 deg (project convention).
    plus = ESPRITEstimator().estimate(_column(geometry, +30.0),
                                      geometry, _context())
    assert plus.azimuth_deg == pytest.approx(+30.0, abs=TIGHT)
    assert plus.diagnostics["spatial_phase_shifts"][0] == pytest.approx(
        +np.pi / 8, rel=1e-6)


# --- 14. aliasing / incompatible spacing handled explicitly --------------------
def test_aliased_spacing_fails_loudly():
    """d = 0.75*lambda creates a second in-domain solution: must not guess."""
    aliased = _ula(6, spacing_m=0.75 * WAVELENGTH_M)
    data = _column(aliased, -80.0)
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        with pytest.raises(DOAError):
            estimator.estimate(data, aliased, _context())


def test_half_wave_and_quarter_wave_spacing_remain_unique():
    for spacing in (0.25 * WAVELENGTH_M, 0.5 * WAVELENGTH_M):
        geometry = _ula(6, spacing_m=spacing)
        for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
            result = estimator.estimate(_column(geometry, TRUE_AOA_DEG),
                                        geometry, _context())
            assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)


# --- 15. registry integration --------------------------------------------------
def test_registry_integration():
    assert get_doa_estimator("esprit").algorithm == "esprit"
    assert get_doa_estimator("root_music").algorithm == "root_music"
    assert isinstance(get_doa_estimator("esprit"), ESPRITEstimator)
    assert isinstance(get_doa_estimator("root_music"), RootMUSICEstimator)
    with pytest.raises(ValueError):
        get_doa_estimator("definitely_not_registered")


# --- 16. sign-convention parity with existing estimators -----------------------
def test_sign_convention_parity_with_existing_estimators():
    geometry = _ula(6)
    data = _column(geometry, TRUE_AOA_DEG)
    bartlett = BartlettEstimator().estimate(data, geometry, _context())
    assert bartlett.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=1e-9)
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(data, geometry, _context())
        # Same polarity, same RF carrier: all estimators must agree in sign
        # and value.  (No estimator is permitted to fix a mismatch by
        # multiplying the final answer by -1.)
        assert result.azimuth_deg == pytest.approx(
            bartlett.azimuth_deg, abs=TIGHT), estimator.algorithm
    # Negative control: the IF carrier carries no resolvable spatial phase;
    # structured estimators must fail loudly, not invert a degenerate phase.
    if_context = {"frequency_hz": float(params.IF_FREQ) * 1.0e6}
    with pytest.raises(DOAError):
        ESPRITEstimator().estimate(data, geometry, if_context)


# --- IFSnapshotData container input (production data model parity) -------------
def test_ifsnapshot_container_input():
    geometry = _ula(6)
    data = _column(geometry, TRUE_AOA_DEG)
    snap = IFSnapshotData(
        X=np.repeat(data, 8, axis=1),
        time_centers_us=np.arange(8, dtype=float),
        if_frequency_mhz=params.IF_FREQ,
        n_paths=geometry.m,
        window_duration_us=1.0,
        samples_per_window=10,
    )
    for estimator in (ESPRITEstimator(), RootMUSICEstimator()):
        result = estimator.estimate(snap, geometry, _context())
        assert result.azimuth_deg == pytest.approx(TRUE_AOA_DEG, abs=TIGHT)


# --- OPTIONAL doa_py oracle (DOA-ALG-002 §13) -----------------------------------
# doa_py is a research/reference package, never a production dependency.
# If absent, this test skips cleanly.  Convention adaptation is explicit:
#   doa_py UniformLinearArray lies along +y with steering exp(-j k d i sin az),
#   project ULA lies along +x with data x = exp(+j k d i sin theta_project),
# hence theta_project = -az_doapy for identical element spacing.
def test_optional_doa_py_oracle_cross_check():
    doa_py = pytest.importorskip("doa_py", reason="optional oracle not installed")
    import importlib.metadata as md
    from doa_py.algorithm.esprit_based import esprit as oracle_esprit
    from doa_py.algorithm.music_based import root_music as oracle_root_music
    from doa_py.arrays import UniformLinearArray

    # Record the exact oracle version used for the cross-check.
    oracle_version = md.version("doa-py")
    assert tuple(int(t) for t in oracle_version.split(".")[:2]) >= (0, 5)

    m = 8
    geometry = _ula(m)
    oracle_array = UniformLinearArray(m, SPACING_M)
    data = _two_source_snapshots(geometry, TWO_SRC_TRUE)
    adapted = -np.asarray(oracle_esprit(
        data, 2, oracle_array, RF_FREQ_HZ))
    np.testing.assert_allclose(
        np.sort(adapted), np.sort(TWO_SRC_TRUE), atol=0.5)
    adapted_rm = -np.asarray(oracle_root_music(
        data, 2, oracle_array, RF_FREQ_HZ))
    np.testing.assert_allclose(
        np.sort(adapted_rm), np.sort(TWO_SRC_TRUE), atol=0.5)
    # Project estimators must agree with the adapted oracle on the SAME data.
    for estimator in (ESPRITEstimator(n_sources=2),
                      RootMUSICEstimator(n_sources=2)):
        result = estimator.estimate(data, geometry, _context())
        got = np.sort(np.asarray(
            result.diagnostics["all_azimuths_deg"], dtype=float))
        oracle_sorted = np.sort(
            adapted if estimator.algorithm == "esprit" else adapted_rm)
        np.testing.assert_allclose(got, oracle_sorted, atol=0.5)
