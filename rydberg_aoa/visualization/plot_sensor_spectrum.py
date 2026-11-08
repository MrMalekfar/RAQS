"""
Sensor spectrum plotting (EIT/AT resonances) for Rydberg AoA simulation.

Numerical result figure conventions (thesis):

* The RF-off curve is labelled ``EIT — RF off`` and the RF-on curve
  ``AT — LO on``; the displayed Rabi frequency always uses the
  physically unambiguous angular-frequency notation
  ``Omega_LO/(2*pi)`` because the stored ``omega_rf_base`` is angular.
* The two Autler-Townes transmission maxima are MEASURED directly from
  the plotted RF-on array (``measure_at_peaks``); the figure marks the
  actual detected peak positions and reports the measured splitting.
* A subtle vertical line marks the frozen sensor operating point
  ``Delta_p,sensor`` read from the LIVE configuration
  (``parameters.SENSOR_PROBE_DETUNING_MHZ``).

The analytical variant of this figure is retained ONLY as a
diagnostic/appendix artifact and is explicitly labelled as the reduced
analytical model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .style import (
    TITLE_FONT_SIZE_MAIN, TITLE_FONT_SIZE_SUBPLOT, AXIS_LABEL_FONT_SIZE, LEGEND_FONT_SIZE,
    LEGEND_POSITION_UPPER_RIGHT, GRID_ALPHA, LINE_WIDTH_MAIN, LINE_WIDTH_SECONDARY,
    LINE_ALPHA_MAIN, LINE_ALPHA_SECONDARY, COLOR_PALETTE_MODERN,
    ENERGY_UNIT_MHZ
)


def _sensor_detuning_mhz() -> float:
    """Frozen sensor probe operating detuning [MHz] from the LIVE config."""
    from ..config.parameters import SENSOR_PROBE_DETUNING_MHZ
    return float(SENSOR_PROBE_DETUNING_MHZ)


def measure_at_peaks(
    dp_scan: np.ndarray,
    trans_with_rf: np.ndarray,
    trans_no_rf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Measure the Autler-Townes transmission maxima DIRECTLY from the plotted
    RF-on array (no model formula is used).

    The two most prominent local maxima of the RF-on transmission curve are
    detected with a simple robustness-prominence method
    (``scipy.signal.find_peaks``): a local maximum is kept only when it
    stands above both neighbouring bases by at least 1% of the RF-on
    peak-to-peak span.  No comparison with the RF-off curve is made -- an AT
    transmission maximum is a local property of the RF-on spectrum and is
    not required to exceed the RF-off transmission at the same detuning.

    Args:
        dp_scan: Probe detuning scan array (MHz), monotonically increasing.
        trans_with_rf: RF-on probe transmission on the same grid.
        trans_no_rf: Accepted for API compatibility; no longer used as a
            peak-selection criterion.

    Returns:
        peaks_mhz : sorted array of the up-to-two most prominent peak
            detunings [MHz] (left-to-right)
        splitting_mhz : measured separation between the outermost detected
            peaks [MHz] (0.0 when fewer than two peaks are found)
    """
    from scipy.signal import find_peaks

    d = np.asarray(dp_scan, dtype=float)
    t = np.asarray(trans_with_rf, dtype=float)
    if len(d) < 3:
        return np.array([]), 0.0

    # Robust local maxima: suppress spurious wiggles without imposing any
    # theoretical splitting or forcing the peak positions.
    span = float(np.ptp(t))
    min_prominence = 0.01 * span if span > 0.0 else 1e-12
    idx, props = find_peaks(t, prominence=min_prominence)
    if len(idx) == 0:
        # No resolvable local maximum: fall back to the global maximum.
        return np.array([d[int(np.argmax(t))]]), 0.0

    # Keep only the two most prominent maxima (the AT doublet members).
    order = np.argsort(props["prominences"])[::-1][:2]
    peaks = np.sort(d[idx[order]])
    splitting = float(peaks[-1] - peaks[0]) if len(peaks) >= 2 else 0.0
    return peaks, splitting



def plot_sensor_spectrum(dp_scan: np.ndarray,
                        trans_no_rf: np.ndarray,
                        trans_with_rf: np.ndarray,
                        omega_rf_base: float,
                        colors: list = None) -> plt.Axes:
    """
    Plot the sensor spectrum showing EIT and AT resonances (numerical results).

    Args:
        dp_scan: Probe detuning scan array (MHz)
        trans_no_rf: Transmission without RF (numerical)
        trans_with_rf: Transmission with RF (numerical)
        omega_rf_base: RF Rabi frequency in angular units
        colors: List of colors for modern styling

    Returns:
        matplotlib Axes object
    """
    if colors is None:
        colors = COLOR_PALETTE_MODERN[:2]  # Use first two colors

    ax = plt.gca()
    ax.plot(dp_scan, trans_no_rf, color=colors[0], label='EIT — RF off',
            linewidth=LINE_WIDTH_MAIN, alpha=LINE_ALPHA_MAIN)
    # omega_rf_base is ANGULAR: display Omega_LO/(2*pi) in MHz.
    ax.plot(dp_scan, trans_with_rf, color=colors[1],
            label=(r'AT — LO on, $\Omega_{\mathrm{LO}}/(2\pi)$'
                   + f' = {omega_rf_base/(2*np.pi):.2f} MHz'),
            linewidth=LINE_WIDTH_MAIN, alpha=LINE_ALPHA_MAIN)

    # --- MEASURED AT peaks from the actual plotted array -------------------
    peaks, splitting = measure_at_peaks(dp_scan, trans_with_rf, trans_no_rf)
    if len(peaks) >= 1:
        heights = np.interp(peaks, np.asarray(dp_scan, dtype=float),
                            np.asarray(trans_with_rf, dtype=float))
        ax.plot(peaks, heights, 'v', color=colors[1], markersize=8, zorder=5,
                label='measured AT peaks')
    if len(peaks) >= 2:
        ax.annotate(
            (f"measured AT peaks: {peaks[0]:+.1f} MHz, {peaks[-1]:+.1f} MHz\n"
             f"measured splitting = {splitting:.1f} MHz"),
            xy=(0.02, 0.97), xycoords='axes fraction',
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec=colors[1], alpha=0.85),
        )

    # --- frozen sensor operating point (live config) ------------------------
    d_sensor = _sensor_detuning_mhz()
    ax.axvline(d_sensor, color='#555555', linestyle=(0, (4, 3)),
               linewidth=1.2, alpha=0.9, zorder=1,
               label=f'sensor operating point ({d_sensor:+.1f} MHz)')

    ax.set_title('(a) Sensor Spectrum (EIT/AT)', fontsize=TITLE_FONT_SIZE_MAIN, fontweight='bold')
    ax.set_xlabel(f'Probe Detuning $\\Delta_p$ ({ENERGY_UNIT_MHZ})', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Probe Transmission', fontsize=AXIS_LABEL_FONT_SIZE)

    # FIG-003: keep the legend OFF the EIT/AT curves.  Standalone figures get
    # the legend in the reserved band ABOVE the axes (top-right of the figure,
    # clear of the centered title); the embedded summary panel keeps a compact
    # in-axes legend (the panel's upper right is clear of the centre
    # resonance features).
    if len(ax.figure.axes) == 1:
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right',
                  bbox_to_anchor=(0.985, 0.985),
                  bbox_transform=ax.figure.transFigure)
        ax.figure.subplots_adjust(top=0.80)
    else:
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=LEGEND_POSITION_UPPER_RIGHT)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(dp_scan[0] if len(dp_scan) > 0 else -20, dp_scan[-1] if len(dp_scan) > 0 else 20)

    return ax


def plot_analytical_spectrum(dp_scan: np.ndarray,
                           trans_no_rf_analytic: np.ndarray,
                           trans_with_rf_analytic: np.ndarray,
                           omega_rf_base: float) -> plt.Axes:
    """
    Plot the analytical sensor spectrum showing EIT and AT resonances.

    Args:
        dp_scan: Probe detuning scan array (MHz)
        trans_no_rf_analytic: Analytical transmission without RF
        trans_with_rf_analytic: Analytical transmission with RF
        omega_rf_base: RF Rabi frequency in angular units

    Returns:
        matplotlib Axes object
    """
    ax = plt.gca()
    fig = ax.figure
    ax.plot(dp_scan, trans_no_rf_analytic, 'b-', label='EIT — RF off',
            linewidth=LINE_WIDTH_SECONDARY, alpha=LINE_ALPHA_SECONDARY)
    ax.plot(dp_scan, trans_with_rf_analytic, 'r-',
            label=(r'AT — LO on, $\Omega_{\mathrm{LO}}/(2\pi)$'
                   + f' = {omega_rf_base/(2*np.pi):.2f} MHz'),
            linewidth=LINE_WIDTH_SECONDARY, alpha=LINE_ALPHA_SECONDARY)

    # FIG-003: restored standalone analytical figure.  Title + subtitle state
    # explicitly that this is the REDUCED analytical model; no numerical/
    # analytical agreement is claimed anywhere on the figure.
    fig.suptitle('Reduced Analytical Sensor Spectrum',
                 fontsize=TITLE_FONT_SIZE_MAIN, fontweight='bold', y=0.985)
    ax.set_title('Full Doppler/transit treatment not included',
                 fontsize=TITLE_FONT_SIZE_SUBPLOT, style='italic', pad=16)
    ax.set_xlabel(f'Probe Detuning $\\Delta_p$ ({ENERGY_UNIT_MHZ})', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Probe Transmission $\\propto e^(-Im(\\rho_{21})*L)$', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right',
              bbox_to_anchor=(0.985, 0.985), bbox_transform=fig.transFigure)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.subplots_adjust(top=0.80)
    return ax
