# analysis.py
"""
Post-processing and analysis functions for AoA estimation.

DOA-ARCH-001: the production beamforming estimator now lives behind the
common DOA estimator interface (rydberg_aoa.doa).  This legacy entry point
adapts its (data, positions, frequency, scan-grid) contract to that
interface and returns identical results to the historical implementation.
"""

from rydberg_aoa.doa.geometry import ArrayGeometry
from rydberg_aoa.doa.estimators import PhaseBeamformingEstimator


def estimate_aoa_beamforming(complex_data, array_pos, rf_freq, scan_angles):
    """
    Estimates AoA using conventional delay-and-sum beamforming.
    
    Args:
        complex_data (np.array): The (N_elements,) complex response vector.
        array_pos (np.array): (N_elements, 2) array of [x, y] positions.
        rf_freq (float): The RF frequency (in Hz).
        scan_angles (np.array): A list of angles (in deg) to scan.
        
    Returns:
        scan_angles (np.array): The input scan_angles.
        power_spectrum (np.array): The calculated power for each scan_angle.
    """
    geometry = ArrayGeometry.from_xy(array_pos)
    estimator = PhaseBeamformingEstimator(scan_azimuth_deg=scan_angles)
    result = estimator.estimate(complex_data, geometry, {"frequency_hz": rf_freq})
    return result.scan_grid, result.spectrum
