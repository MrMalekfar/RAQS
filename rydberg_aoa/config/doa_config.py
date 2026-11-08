"""
DOA post-processing configuration (DOA-INTEGRATE-001).

This module owns ONLY the direction-of-arrival (AoA) post-processing
configuration.  It is deliberately separate from the physical-model
configuration (rydberg_aoa.config.parameters) so selecting a DOA estimator
can never change the simulated physics and vice versa.

Defaults preserve the current production behavior exactly:

    DOA_METHOD = "phase_beamforming"

with the same azimuth scan grid the legacy production estimator used
(np.linspace(-90, 90, 181)).  The spatial steering carrier is NOT taken
from here: it is always the RF signal carrier (parameters.RF_SIGNAL_FREQUENCY);
the heterodyne IF frequency is used only for temporal snapshot extraction.
"""

# ---------------------------------------------------------------------------
# Estimator selection (read LAZILY at call time by rydberg_aoa.doa.registry;
# there is deliberately no import-time frozen copy anywhere).
# ---------------------------------------------------------------------------
DOA_METHOD = "phase_beamforming"

# Multi-source contract (passed explicitly to MUSIC / ESPRIT / Root-MUSIC).
DOA_NUM_SOURCES = 1

# Azimuth scan grid (spectrum/search estimators; degrees).
DOA_AZIMUTH_MIN_DEG = -90.0
DOA_AZIMUTH_MAX_DEG = 90.0
DOA_AZIMUTH_STEP_DEG = 1.0

# Fixed-elevation assumption of the production geometry.
DOA_ELEVATION_DEG = 0.0

# Temporal snapshot extraction: IF cycles per snapshot window
# (consumed by rydberg_aoa.doa.snapshots.extract_if_snapshots).
DOA_SNAPSHOT_CYCLES = 2

# MVDR/Capon diagonal loading (relative to trace(Rxx)/M); a numerical
# regularization parameter, not physical noise.
DOA_MVDR_LOADING_REL = 1.0e-6


def azimuth_scan_grid_deg():
    """Build the configured azimuth scan grid in degrees.

    Returns a 1-D numpy array from DOA_AZIMUTH_MIN_DEG to DOA_AZIMUTH_MAX_DEG
    inclusive with DOA_AZIMUTH_STEP_DEG spacing, generated with
    numpy.linspace so the default configuration reproduces the legacy
    production grid np.linspace(-90, 90, 181) bit-exactly.

    Raises:
        ValueError: when the configured grid is degenerate (non-positive
            step, max <= min, or a non-integer step count).
    """
    import numpy as np

    lo = float(DOA_AZIMUTH_MIN_DEG)
    hi = float(DOA_AZIMUTH_MAX_DEG)
    step = float(DOA_AZIMUTH_STEP_DEG)
    if not step > 0.0:
        raise ValueError(f"DOA_AZIMUTH_STEP_DEG must be > 0; got {step!r}")
    if hi <= lo:
        raise ValueError(
            "DOA_AZIMUTH_MAX_DEG must be > DOA_AZIMUTH_MIN_DEG; "
            f"got [{lo!r}, {hi!r}]"
        )
    n_points = int(round((hi - lo) / step)) + 1
    if n_points < 2:
        raise ValueError(
            "configured azimuth grid has fewer than two points; "
            f"step {step!r} is too coarse for [{lo!r}, {hi!r}]"
        )
    return np.linspace(lo, hi, n_points)
