"""DOA-ARCH-001 — geometry-agnostic AoA architecture (synthetic and fast).

Validates the estimator/geometry/steering decoupling WITHOUT physics, ARC,
Doppler or multiprocessing.  The frozen production result at the current
six-path ULA must be preserved exactly:

    true AoA          = -30 deg
    estimated AoA     = -30 deg
    phase progression = -22.5 deg/path
"""

import inspect

import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.doa import (
    ArrayGeometry,
    DOAEstimator,
    IncompatibleGeometryError,
    PhaseBeamformingEstimator,
    arrival_phases,
    get_doa_estimator,
    steering_matrix,
    steering_vector,
    wave_number,
)
from rydberg_aoa.doa import estimators as estimators_module
from rydberg_aoa.signal_processing.array_geometry import get_array_phases
from rydberg_aoa.tools.analysis import estimate_aoa_beamforming as legacy_beamforming

RF_FREQ_HZ = params.RF_FREQUENCY
TRUE_AOA_DEG = params.TARGET_AOA_DEG        # -30.0
SPACING_M = params.SPACING                  # lambda_RF / 8
IF_FREQ_HZ = params.IF_FREQ * 1.0e6         # 0.1 MHz expressed in Hz
GRID_DEG = np.linspace(-90.0, 90.0, 181)
ESTIMATOR_NAME = "phase_beamforming"


def _ula(n_elements):
    return ArrayGeometry.from_ula(n_elements=n_elements, spacing_m=SPACING_M)


def _ideal_response(geometry, azimuth_deg, frequency_hz=RF_FREQ_HZ):
    """Ideal noiseless element responses carrying the true arrival phases."""
    return np.exp(1j * arrival_phases(geometry, frequency_hz, azimuth_deg))


def _estimate(data, geometry, frequency_hz=RF_FREQ_HZ, scan_azimuth_deg=None):
    estimator = PhaseBeamformingEstimator(scan_azimuth_deg=scan_azimuth_deg)
    return estimator.estimate(data, geometry, {"frequency_hz": frequency_hz})


# --- A. frozen six-path ULA production parity --------------------------------
def test_ula_six_paths_recovers_true_aoa_exactly():
    geometry = _ula(6)
    assert geometry.m == 6
    result = _estimate(_ideal_response(geometry, TRUE_AOA_DEG), geometry)
    assert result.azimuth_deg == -30.0                    # exact grid bin
    assert result.scan_grid[np.argmax(result.spectrum)] == result.azimuth_deg
    assert result.elevation_deg == 0.0


def test_ula_six_paths_phase_progression_minus_22_5_deg():
    geometry = _ula(6)
    phases = arrival_phases(geometry, RF_FREQ_HZ, TRUE_AOA_DEG)
    progression = np.rad2deg(np.diff(phases))
    np.testing.assert_allclose(progression, -22.5, rtol=0.0, atol=1.0e-9)
    # steering is the conjugate phase ramp: +22.5 deg/path progression
    s = steering_vector(geometry.positions_m, RF_FREQ_HZ, TRUE_AOA_DEG)
    steering_progression = np.rad2deg(np.diff(np.unwrap(np.angle(s))))
    np.testing.assert_allclose(steering_progression, 22.5, rtol=0.0, atol=1.0e-9)


# --- B. different path counts ------------------------------------------------
@pytest.mark.parametrize("m", [3, 4, 8])
def test_ula_arbitrary_path_counts(m):
    geometry = _ula(m)
    assert geometry.m == m
    result = _estimate(_ideal_response(geometry, TRUE_AOA_DEG), geometry)
    assert result.azimuth_deg == -30.0
    assert result.diagnostics["n_elements"] == m


# --- C. translation invariance ----------------------------------------------
def test_common_xyz_offset_does_not_change_aoa():
    geometry = _ula(6)
    offset = np.array([0.1234, -5.6789, 2.3456])
    moved = ArrayGeometry(geometry.positions_m + offset)

    # original data estimated on the translated geometry (pure estimator check)
    result = _estimate(_ideal_response(geometry, TRUE_AOA_DEG), moved)
    assert result.azimuth_deg == -30.0

    # physically consistent variant: data rebuilt at the translated coordinates
    result_moved = _estimate(_ideal_response(moved, TRUE_AOA_DEG), moved)
    assert result_moved.azimuth_deg == -30.0


# --- D. arbitrary non-uniform positions -------------------------------------
def test_arbitrary_nonuniform_positions_geometry_agnostic():
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(5, 3)) * 0.01            # metres, non-uniform
    geometry = ArrayGeometry(positions)
    result = _estimate(_ideal_response(geometry, TRUE_AOA_DEG), geometry)
    assert np.all(np.isfinite(result.spectrum))
    assert result.azimuth_deg == -30.0

    scan = np.linspace(-90.0, 90.0, 37)
    matrix = steering_matrix(geometry.positions_m, RF_FREQ_HZ, scan)
    assert matrix.shape == (37, 5)
    assert np.all(np.isfinite(matrix))
    vector = steering_vector(geometry.positions_m, RF_FREQ_HZ, 15.0)
    assert vector.shape == (5,)
    assert np.all(np.isfinite(vector))


# --- E. RF carrier contract --------------------------------------------------
def test_steering_uses_rf_carrier_never_if_frequency():
    geometry = _ula(6)
    assert wave_number(RF_FREQ_HZ) == pytest.approx(
        2.0 * np.pi * RF_FREQ_HZ / 299792458.0)

    prog_rf = np.rad2deg(np.diff(
        arrival_phases(geometry, RF_FREQ_HZ, TRUE_AOA_DEG)))
    np.testing.assert_allclose(prog_rf, -22.5, rtol=0.0, atol=1.0e-9)

    # The IF beat note carries no resolvable differential phase: flat ramp.
    prog_if = np.rad2deg(np.diff(
        arrival_phases(geometry, IF_FREQ_HZ, TRUE_AOA_DEG)))
    assert np.max(np.abs(prog_if)) < 1.0e-4

    data = _ideal_response(geometry, TRUE_AOA_DEG)
    result_rf = _estimate(data, geometry, frequency_hz=RF_FREQ_HZ)
    assert result_rf.azimuth_deg == -30.0

    result_if = _estimate(data, geometry, frequency_hz=IF_FREQ_HZ)
    assert result_if.azimuth_deg != -30.0
    peak = np.max(result_if.spectrum)
    at_true = result_if.spectrum[result_if.scan_grid == -30.0][0]
    assert (peak / at_true) - 1.0 < 1.0e-4     # flat spectrum: no spatial info


# --- F. no hard-coded element count inside the DOA estimator -----------------
def test_estimator_has_no_hardcoded_six():
    source = inspect.getsource(estimators_module)
    assert "6" not in source


# --- registry -----------------------------------------------------------------
def test_registry_returns_phase_beamforming():
    estimator = get_doa_estimator(ESTIMATOR_NAME)
    assert isinstance(estimator, PhaseBeamformingEstimator)
    assert estimator.algorithm == ESTIMATOR_NAME


def test_registry_reads_config_lazily(monkeypatch):
    monkeypatch.setattr(params, "DOA_METHOD", "phase_beamforming", raising=False)
    estimator = get_doa_estimator()          # no import-time snapshot
    assert estimator.algorithm == "phase_beamforming"


def test_registry_rejects_unknown_method():
    with pytest.raises(ValueError):
        get_doa_estimator("definitely_not_a_doa_method")   # explicit error


# --- capability contract -----------------------------------------------------
def test_geometry_capability_flags():
    ula = _ula(4)
    caps = ula.capabilities
    assert caps["arbitrary"] is True
    assert caps["shift_invariant"] is True
    assert caps["ula"] is True
    assert caps["ura"] is False
    assert caps["uca"] is False

    rng = np.random.default_rng(1)
    odd = ArrayGeometry(rng.normal(size=(4, 3)))
    assert odd.capabilities["ula"] is False
    assert odd.capabilities["ura"] is False
    assert odd.capabilities["uca"] is False


def test_incompatible_geometry_rejected_explicitly():
    """A future algorithm/geometry mismatch must raise, never fall back."""
    class _UcaOnlyEstimator(DOAEstimator):
        algorithm = "test_uca_only"
        required_capabilities = frozenset({"uca"})

        def _estimate_impl(self, data, geometry, context=None):
            raise AssertionError("estimate core must never be reached")

    geometry = _ula(4)
    estimator = _UcaOnlyEstimator()
    data = _ideal_response(geometry, -10.0)
    assert estimator.supports(geometry, data) is False
    with pytest.raises(IncompatibleGeometryError):
        estimator.estimate(data, geometry, {"frequency_hz": RF_FREQ_HZ})


# --- legacy parity -----------------------------------------------------------
def test_legacy_beamforming_parity_bit_identical():
    """The wrapped estimator reproduces the legacy production estimator exactly."""
    geometry = _ula(6)
    pos2d = geometry.positions_m[:, :2]

    phases_legacy = get_array_phases(pos2d, RF_FREQ_HZ, TRUE_AOA_DEG)
    phases_new = arrival_phases(geometry, RF_FREQ_HZ, TRUE_AOA_DEG)
    np.testing.assert_array_equal(phases_legacy, phases_new)

    data = np.exp(1j * phases_legacy)
    angles_legacy, power_legacy = legacy_beamforming(data, pos2d, RF_FREQ_HZ, GRID_DEG)
    result = _estimate(data, geometry, scan_azimuth_deg=GRID_DEG)
    np.testing.assert_array_equal(angles_legacy, result.scan_grid)
    np.testing.assert_array_equal(power_legacy, result.spectrum)
    assert result.azimuth_deg == angles_legacy[np.argmax(power_legacy)]
    assert result.azimuth_deg == -30.0


# --- input validation --------------------------------------------------------
def test_geometry_validation():
    with pytest.raises(ValueError):
        ArrayGeometry(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        ArrayGeometry(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        ArrayGeometry.from_ula(0, SPACING_M)
    with pytest.raises(ValueError):
        ArrayGeometry.from_ula(4, -0.1)