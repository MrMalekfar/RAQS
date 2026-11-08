"""
Estimator-result summary panel for search-free DOA methods (DOA-INTEGRATE-001).

Search-free estimators (ESPRIT, Root-MUSIC) legitimately produce NO angular
spectrum.  This panel shows the estimator result directly — selected
algorithm, estimated AoA, true AoA and the absolute error — WITHOUT
fabricating a Bartlett/MUSIC spectrum and presenting it as the search-free
estimator's output.
"""

import matplotlib.pyplot as plt
from typing import Optional


def plot_doa_summary(doa_result,
                     true_aoa: float,
                     colors: Optional[list] = None) -> plt.Axes:
    """
    Plot a simple estimator-result panel for spectrum-free DOA methods.

    Args:
        doa_result: DOAResult from the production dispatch (its ``spectrum``
            is expected to be None for search-free estimators; a spectrum,
            when present, is ignored here — use plot_aoa_spectrum instead).
        true_aoa: True angle of arrival (degrees).
        colors: Optional list of colors for styling.

    Returns:
        matplotlib Axes object
    """
    if colors is None:
        colors = ['#1f77b4', '#d62728']

    ax = plt.gca()
    ax.axis('off')

    algorithm = getattr(doa_result, 'algorithm', 'unknown')
    estimated_aoa = float(getattr(doa_result, 'azimuth_deg', float('nan')))
    elevation = getattr(doa_result, 'elevation_deg', None)
    error = abs(estimated_aoa - true_aoa)

    lines = [
        f'Algorithm: {algorithm}',
        f'Estimated AoA: {estimated_aoa:.2f}\u00b0',
        f'True AoA: {true_aoa:.2f}\u00b0',
        f'|error| = {error:.2f}\u00b0',
    ]
    if elevation is not None:
        lines.append(f'Elevation (assumed): {float(elevation):.1f}\u00b0')

    ax.text(0.5, 0.55, '\n'.join(lines),
            transform=ax.transAxes, fontsize=12, ha='center',
            va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor=colors[1], alpha=0.9))

    ax.set_title(f'(c) DOA Result \u2014 {algorithm} (search-free, '
                 'no angular spectrum)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax
