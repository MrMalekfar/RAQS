"""
AoA spectrum plotting for beamforming results (FIG-002 simplified layout).

Shows exactly three visual elements: the beamforming spectrum curve, the
true-AoA vertical dashed line, and the estimated-peak marker. The AoA
estimate and absolute error are annotated compactly in one box.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from ..exceptions import PhysicsCalculationError


def plot_aoa_spectrum(angles: np.ndarray,
                      power: np.ndarray,
                      true_aoa: float,
                      estimated_aoa: float,
                      colors: Optional[list] = None,
                      *, algorithm: str = 'Beamforming') -> plt.Axes:
    """
    Plot the AoA power spectrum from beamforming (simplified).

    Args:
        angles: Scan angles array (degrees)
        power: Nonnegative spectrum (power or estimator pseudospectrum).
            Displayed as 10*log10(power/max(power)) [dB]; input is unchanged.
            Exact zeros remain -inf dB and are not drawn; no display floor.
        true_aoa: True angle of arrival (degrees)
        estimated_aoa: Estimated angle of arrival (degrees)
        colors: Optional list of colors for styling

    Returns:
        matplotlib Axes object
    """
    if colors is None:
        colors = ['#2ca02c', '#1f77b4', '#d62728']

    values = np.asarray(power)
    if (values.dtype.kind not in 'iuf' or values.ndim != 1 or not values.size
            or not np.isfinite(values).all() or np.any(values < 0) or values.max() <= 0):
        raise PhysicsCalculationError('AoA plot: finite nonnegative spectrum with a positive maximum required.')
    positive = values > 0
    reference_log = np.log10(values.max())
    display_db = np.full(values.shape, -np.inf)
    # Log difference avoids underflow from dividing a tiny value by the peak.
    display_db[positive] = 10 * (np.log10(values[positive]) - reference_log)
    ax = plt.gca()

    # 1) beamforming spectrum curve
    ax.plot(angles, display_db, color=colors[0], linewidth=2.2, alpha=0.9,
            label=f'{algorithm} spectrum')

    # 2) true AoA vertical dashed line
    ax.axvline(true_aoa, color=colors[1], linestyle='--', linewidth=2.0,
               alpha=0.8, label=f'True AoA ({true_aoa:.1f}\u00b0)')

    # 3) estimated peak marker (legend-free; described by the annotation box)
    # Mark the selected estimator's answer; its answer need not be argmax
    # (for example, a multi-peak result). Interpolation is display-only.
    marker_power = np.interp(estimated_aoa, angles, values)
    marker_db = 10 * (np.log10(marker_power) - reference_log) if marker_power > 0 else -np.inf
    ax.plot(estimated_aoa, marker_db, 'o', color=colors[2], markersize=9,
            markerfacecolor='white', markeredgewidth=2)

    # Compact annotation: estimated AoA + absolute error
    error = abs(estimated_aoa - true_aoa)
    ax.text(0.04, 0.92,
            f'Estimated AoA: {estimated_aoa:.2f}\u00b0\n|error| = {error:.2f}\u00b0',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=colors[2], alpha=0.9))

    ax.set_title(f'(c) AoA Power Spectrum ({algorithm})', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel('Scan angle [\u00b0]', fontsize=11)
    ax.set_ylabel('Normalized spectrum [dB]', fontsize=11)
    ax.set_ylim(min(-3., display_db[positive].min() - 3.), 3.)
    if not positive.all():
        ax.text(.98, .03, 'Zero spectrum: −∞ dB (not drawn)', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom')
    # FIG-003: keep the legend off the beamforming main lobe.  Standalone
    # figures get the legend in the reserved band ABOVE the axes (top-right
    # of the figure); the embedded summary panel keeps the in-axes legend at
    # upper right, clear of the centre-left main lobe.
    if len(ax.figure.axes) == 1:
        ax.legend(fontsize=9.5, loc='upper right',
                  bbox_to_anchor=(0.985, 0.985),
                  bbox_transform=ax.figure.transFigure)
        ax.figure.subplots_adjust(top=0.80)
    else:
        ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(True, alpha=0.3)

    if len(angles) > 0:
        ax.set_xlim(angles[0], angles[-1])

    return ax
