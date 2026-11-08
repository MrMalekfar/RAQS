"""
Phase-progression figure (FIG-002).

Plots the REAL extracted production IF phases (runner key
``results['extracted_phases']``, produced by ``extract_phases`` as
``np.unwrap(np.angle(...))`` and converted to degrees here) against the
laser-path index, overlays a linear least-squares fit, and annotates the
phase slope, true AoA, estimated AoA and absolute error.

No mock, ideal or reconstructed phase values are used.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence


def plot_phase_progression(extracted_phases_deg: Sequence[float],
                           true_aoa: float,
                           estimated_aoa: float,
                           ax: Optional[plt.Axes] = None,
                           colors: Optional[list] = None,
                           *, fit_linear: bool = True):
    """
    Plot extracted IF phase vs laser-path index with an LSQ linear fit.

    Args:
        extracted_phases_deg: Real production unwrapped IF phases [deg],
            one value per laser path (len == number of laser paths)
        true_aoa: True angle of arrival [deg]
        estimated_aoa: Production-estimated angle of arrival [deg]
        ax: Optional axes to embed into (standalone figure when None)
        colors: Optional [data color, fit color]
        fit_linear: Whether ordered geometry supports a phase-vs-index fit.

    Returns:
        (fig, ax, slope_deg_per_path) tuple; slope is None without a fit.
    """
    if colors is None:
        colors = ['#1f77b4', '#d62728']

    phases = np.asarray(extracted_phases_deg, dtype=float)
    n_paths = phases.size
    path_idx = np.arange(1, n_paths + 1)  # Laser path index 1..6

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        created_fig = True
    else:
        fig = ax.figure

    # Linear least-squares fit: unwrapped IF phase vs path index
    slope_deg_per_path = None
    if fit_linear:
        slope_deg_per_path, intercept = np.polyfit(path_idx, phases, 1)
        fit = slope_deg_per_path * path_idx + intercept

    ax.plot(path_idx, phases, 'o', color=colors[0], markersize=8,
            label='Extracted IF phase (production)')
    if fit_linear:
        ax.plot(path_idx, fit, '-', color=colors[1], linewidth=2.0,
                label=f'LSQ fit: {slope_deg_per_path:.2f}\u00b0/path')

    ax.set_xlabel('Laser path index', fontsize=11)
    ax.set_ylabel('Extracted IF phase [deg]', fontsize=11)
    ax.set_title('IF Phase Progression Across Laser Paths', fontsize=13,
                 fontweight='bold')
    ax.set_xticks(path_idx)
    ax.grid(True, alpha=0.3)

    error = abs(estimated_aoa - true_aoa)
    ax.text(0.97, 0.97,
            ((f'Phase slope: {slope_deg_per_path:.2f}\u00b0/path\n' if fit_linear
              else 'Nonuniform path geometry: no index fit\n')
             + f'True AoA: {true_aoa:.2f}\u00b0\n'
             f'Estimated AoA: {estimated_aoa:.2f}\u00b0\n'
             f'|AoA error|: {error:.2f}\u00b0'),
            transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#888888', alpha=0.9))
    # FIG-003: legend moved OUTSIDE the data region (below the axes) so it
    # never covers the fitted line.  The upper-right information box above
    # does not cover the data and is kept.
    ax.legend(fontsize=9, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.12))

    if created_fig:
        # Reserve bottom margin for the below-axes legend.
        fig.subplots_adjust(bottom=0.28)

    return fig, ax, None if slope_deg_per_path is None else float(slope_deg_per_path)
