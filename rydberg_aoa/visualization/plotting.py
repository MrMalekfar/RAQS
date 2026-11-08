"""
Visualization module for Rydberg AoA simulation results.
Handles all plotting and visualization functionality.

Modernized with:
- Seaborn-inspired styling for publication-quality figures
- Professional color schemes and typography
- Parameter annotations and enhanced layouts
- High-DPI output and multiple formats
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List
import numpy as np
from .style import (
    COLOR_PALETTE_MODERN, ANNOTATION_FONT_SIZE, ANNOTATION_BOX_STYLE_LIGHT,
    FREQUENCY_UNIT_GHZ, FREQUENCY_UNIT_MHZ, ANGLE_UNIT_DEGREES
)

# Import all plotting functions from modular files for backward compatibility
from .plot_sensor_spectrum import plot_sensor_spectrum, plot_analytical_spectrum
from .plot_fft_validation import plot_fft_validation
from .plot_phase_ramp import plot_phase_ramp
from .plot_aoa_spectrum import plot_aoa_spectrum
from .plot_phase_progression import plot_phase_progression
from .plot_doa_summary import plot_doa_summary
from .plot_energy_levels import plot_energy_levels, plot_energy_levels_comparison

# Re-export the functions for external use
__all__ = [
    'plot_sensor_spectrum',
    'plot_analytical_spectrum',
    'plot_fft_validation',
    'plot_phase_ramp',
    'plot_aoa_spectrum',
    'plot_phase_progression',
    'plot_doa_summary',
    'plot_energy_levels',
    'plot_energy_levels_comparison',
    'create_summary_plot',
    'show_plots',
    'save_figure'
]


def _plot_doa_summary_fallback(estimated_aoa, true_aoa, doa_algorithm, colors):
    """Minimal text panel when no DOAResult object is available.

    Used only when a spectrum-free result reaches create_summary_plot
    without the structured DOAResult; shows algorithm / estimated / true
    AoA and the absolute error.  No spectrum is fabricated.
    """
    ax = plt.gca()
    ax.axis('off')
    algorithm = doa_algorithm if doa_algorithm else 'unknown'
    error = abs(float(estimated_aoa) - float(true_aoa))
    ax.text(0.5, 0.55,
            (f'Algorithm: {algorithm}\n'
             f'Estimated AoA: {float(estimated_aoa):.2f}\u00b0\n'
             f'True AoA: {float(true_aoa):.2f}\u00b0\n'
             f'|error| = {error:.2f}\u00b0'),
            transform=ax.transAxes, fontsize=12, ha='center',
            va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor=colors[1] if colors else '#d62728',
                      alpha=0.9))
    ax.set_title(f'(c) DOA Result \u2014 {algorithm} (search-free, '
                 'no angular spectrum)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def create_summary_plot(dp_scan,
                       trans_no_rf,
                       trans_with_rf,
                       omega_rf_base,
                       xf,
                       yf,
                       if_freq,
                       tlist,
                       time_traces,
                       angles,
                       power,
                       true_aoa,
                       estimated_aoa,
                       array_size=None,
                       rf_frequency=None,
                       doa_algorithm=None,
                       doa_result=None) -> plt.Figure:
    """
    Create the compact 3-panel summary plot (numerical results).

    Panel layout (thesis-preferred standalone figures exist separately; this
    composite is retained for presentations only):
        (a) Sensor transmission spectrum (probe detuning scan)
        (b) Time-domain heterodyne signals (AC-coupled, laser paths)
        (c) Beamforming AoA spectrum

    The retired diagnostic/FFT panel is NOT recreated and the energy-level
    diagram is a standalone figure (energy_levels.png/pdf), not a panel.

    Args:
        dp_scan: Probe detuning scan array (MHz)
        trans_no_rf: Transmission without RF (numerical)
        trans_with_rf: Transmission with RF (numerical)
        omega_rf_base: RF Rabi frequency in angular units
        xf: FFT frequency axis (retired; accepted for API compatibility)
        yf: FFT magnitude spectrum (retired; accepted for API compatibility)
        if_freq: Intermediate frequency (MHz)
        tlist: Time axis for heterodyne signals
        time_traces: Complex time-domain traces from heterodyne simulation
        angles: Scan angles array (degrees)
        power: Normalized power spectrum from beamforming
        true_aoa: True angle of arrival (degrees)
        estimated_aoa: Estimated angle of arrival (degrees)
        array_size: Number of array elements (optional)
        rf_frequency: RF frequency (optional)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)

    # Use centralized color palette
    colors = COLOR_PALETTE_MODERN[:5]  # Use first 5 colors

    # Panel (a): Sensor Spectrum
    plt.sca(axes[0])
    plot_sensor_spectrum(dp_scan, trans_no_rf, trans_with_rf, omega_rf_base, colors=colors)

    # Panel (b): Time-Domain Heterodyne Signals (AC-coupled, laser paths)
    plt.sca(axes[1])
    plot_phase_ramp(tlist, time_traces, colors=colors)

    # Panel (c): AoA Spectrum — or, for search-free estimators without an
    # angular spectrum (DOA-INTEGRATE-001), the estimator-result summary
    # panel.  A spectrum is NEVER fabricated for ESPRIT / Root-MUSIC.
    plt.sca(axes[2])
    if power is None or angles is None:
        if doa_result is not None:
            plot_doa_summary(doa_result, true_aoa, colors=colors)
        else:
            _plot_doa_summary_fallback(estimated_aoa, true_aoa,
                                       doa_algorithm, colors)
    else:
        plot_aoa_spectrum(angles, power, true_aoa, estimated_aoa, colors=colors)

    # Add parameter annotation box using centralized styling.
    # Live configuration is the source of truth: only override with the
    # explicit arguments when the caller supplied them (never "N/A").
    try:
        from ..config import parameters as _params
        _n_default = getattr(_params, "N_ELEMENTS", None)
        _rf_hz_default = getattr(_params, "RF_FREQUENCY", None)
    except Exception:  # pragma: no cover - config always available in repo
        _n_default = None
        _rf_hz_default = None
    n_elements = array_size if array_size else _n_default
    rf_freq_hz = rf_frequency if rf_frequency else _rf_hz_default
    array_text = f"{n_elements}" if n_elements else "unknown"
    rf_freq_text = f"{rf_freq_hz/1e9:.3f}" if rf_freq_hz else "unknown"

    param_box = f"""
Simulation Params:
• Array: {array_text} laser paths
• Ω_LO: {omega_rf_base/(2*np.pi):.1f} {FREQUENCY_UNIT_MHZ}
• IF: {if_freq:.3f} {FREQUENCY_UNIT_MHZ}
• AoA Error: {abs(estimated_aoa - true_aoa):.2f}{ANGLE_UNIT_DEGREES}
"""
    fig.text(0.46, 0.98, param_box, transform=fig.transFigure,
             fontsize=ANNOTATION_FONT_SIZE, verticalalignment='top',
             bbox=ANNOTATION_BOX_STYLE_LIGHT)

    return fig


def save_figure(fig, filename, formats=['png', 'pdf'], dpi=300):
    """
    Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
    """
    for fmt in formats:
        if fmt in ['png', 'jpg', 'jpeg']:
            fig.savefig(f'{filename}.{fmt}', dpi=dpi, bbox_inches='tight', format=fmt)
        else:
            fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', format=fmt)


def show_plots(block=True):
    """
    Display all created plots.

    Args:
        block: Whether to block execution until plots are closed
    """
    plt.show(block=block)
