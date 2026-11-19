"""Sub-grid peak refinement of the grid DoA estimators.

The grid estimators used to report the scan-grid angle, so every arrival
between two grid points was quantized to the grid.  These tests show the
accuracy for OFF-GRID arrivals, that the historical grid angle is kept in
the diagnostics and in the run report, and that ``refine_peak=False``
restores the old behaviour.  The synthetic cases use an ideal plane wave
and need no atomic physics; the last test runs the real simulation chain.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rydberg_aoa.config.run_profile import parse_run_profile
from rydberg_aoa.doa.geometry import ArrayGeometry
from rydberg_aoa.doa.peak_refinement import refine_quadratic_form_peak
from rydberg_aoa.doa.registry import get_doa_estimator
from rydberg_aoa.doa.steering import arrival_phases
from tests.test_lab007_profile_integration import isolated  # noqa: F401 - explicit full state isolation

ROOT = Path(__file__).resolve().parents[1]
FREQUENCY_HZ = 132.6491e9
SPACING_M = 1.13e-3            # about half a wavelength
GRID_ESTIMATORS = ('phase_beamforming', 'bartlett', 'mvdr', 'music')
OFF_GRID_DEG = (-73.37, -30.4, -0.25, 12.5, 47.3, 61.81)


def _array(count=6):
    return ArrayGeometry(np.column_stack((np.arange(count) * SPACING_M, np.zeros((count, 2)))))


def _data(name, geometry, azimuth_deg, snapshots=8, noise=0.0, seed=7):
    rng = np.random.default_rng(seed)
    response = np.exp(1j * arrival_phases(geometry.positions_m, FREQUENCY_HZ, azimuth_deg))
    if name == 'phase_beamforming':
        return response
    matrix = response[:, None] * np.exp(1j * rng.uniform(0., 2 * np.pi, snapshots))[None, :]
    if noise:
        matrix = matrix + noise * (rng.normal(size=matrix.shape) + 1j * rng.normal(size=matrix.shape))
    return matrix


def _estimate(name, geometry, data, **kwargs):
    return get_doa_estimator(name, **kwargs).estimate(data, geometry, {'frequency_hz': FREQUENCY_HZ})


@pytest.mark.parametrize('name', GRID_ESTIMATORS)
@pytest.mark.parametrize('true_deg', OFF_GRID_DEG)
def test_off_grid_arrival_is_recovered_far_below_the_grid_spacing(name, true_deg):
    geometry = _array()
    result = _estimate(name, geometry, _data(name, geometry, true_deg))
    grid_angle = result.diagnostics['grid_azimuth_deg']
    assert result.diagnostics['peak_refined'] is True
    assert result.azimuth_deg == pytest.approx(true_deg, abs=1e-6)
    # the historical answer is still reported and is the nearest grid point
    assert grid_angle == result.scan_grid[result.diagnostics['peak_index']]
    assert abs(grid_angle - true_deg) <= 0.5
    assert abs(grid_angle - true_deg) > 100 * abs(result.azimuth_deg - true_deg)


@pytest.mark.parametrize('name', GRID_ESTIMATORS)
def test_refine_peak_false_returns_the_historical_grid_angle(name):
    geometry = _array()
    data = _data(name, geometry, -30.4)
    refined = _estimate(name, geometry, data)
    legacy = _estimate(name, geometry, data, refine_peak=False)
    assert legacy.azimuth_deg == -30.0
    assert legacy.diagnostics['peak_refined'] is False
    assert legacy.diagnostics['grid_azimuth_deg'] == refined.diagnostics['grid_azimuth_deg'] == -30.0
    assert np.array_equal(legacy.spectrum, refined.spectrum)
    assert np.array_equal(legacy.scan_grid, refined.scan_grid)


@pytest.mark.parametrize('name', GRID_ESTIMATORS)
@pytest.mark.parametrize('on_grid_deg', (-30.0, 0.0, 25.0))
def test_on_grid_arrival_stays_exactly_on_the_grid(name, on_grid_deg):
    geometry = _array()
    result = _estimate(name, geometry, _data(name, geometry, on_grid_deg))
    assert result.azimuth_deg == on_grid_deg


def test_noisy_off_grid_error_is_below_the_grid_quantization_floor():
    """300 random arrivals: refined RMSE is well below 1 deg / sqrt(12)."""
    geometry = _array()
    rng = np.random.default_rng(11)
    refined, legacy = [], []
    for trial in range(300):
        true_deg = float(rng.uniform(-60., 60.))
        data = _data('music', geometry, true_deg, snapshots=16, noise=0.05, seed=1000 + trial)
        refined.append(_estimate('music', geometry, data).azimuth_deg - true_deg)
        legacy.append(_estimate('music', geometry, data, refine_peak=False).azimuth_deg - true_deg)
    rmse = lambda errors: float(np.sqrt(np.mean(np.square(errors))))
    quantization_floor = 1.0 / np.sqrt(12.0)
    assert rmse(legacy) == pytest.approx(quantization_floor, rel=0.15)
    assert rmse(refined) < 0.5 * quantization_floor


def test_refinement_keeps_the_grid_angle_when_there_is_no_bracketed_peak():
    geometry = _array()
    grid = np.linspace(-90., 90., 181)
    flat = np.eye(geometry.m, dtype=complex)        # m^H I m is constant: no peak
    assert refine_quadratic_form_peak(geometry.positions_m, FREQUENCY_HZ, 0.0,
                                      flat, grid, 60, maximize=True) == (-30.0, False)
    single = np.array([12.0])
    assert refine_quadratic_form_peak(geometry.positions_m, FREQUENCY_HZ, 0.0,
                                      flat, single, 0, maximize=True) == (12.0, False)


@pytest.mark.parametrize('method', ['phase_beamforming', 'music'])
def test_real_simulation_off_grid_target_and_report_keeps_grid_angle(method, isolated):  # noqa: F811
    """Real ARC + certified BDF + DoA chain with an off-grid target."""
    from rydberg_aoa.simulation.profile_execution import execute_profile
    data = json.loads((ROOT / 'examples/lab006/experimental_near_resonant.json').read_text())
    data['execution'].update(target_azimuth_deg=-30.4, doa_method=method)
    report, output = execute_profile(parse_run_profile(data))
    sensing = report['sensing']
    assert sensing['true_aoa_deg'] == -30.4
    assert sensing['estimated_aoa_deg'] == output['estimated_aoa']
    assert sensing['estimated_aoa_deg'] == pytest.approx(-30.4, abs=1e-3)
    assert sensing['estimated_aoa_grid_deg'] == -30.0      # what was reported before
    assert sensing['doa_peak_refined'] is True
