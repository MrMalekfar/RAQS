"""
FFT validation plotting for Rydberg AoA simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def plot_fft_validation(xf: np.ndarray,
                       yf: np.ndarray,
                       if_freq: float,
                       colors: list = None) -> plt.Axes:
    """
    Plot FFT validation showing the intermediate frequency component.

    Args:
        xf: Frequency array from FFT
        yf: FFT amplitude array
        if_freq: Intermediate frequency (MHz)
        colors: List of colors for modern styling

    Returns:
        matplotlib Axes object
    """
    if colors is None:
        colors = ['#1f77b4', '#d62728']  # Default modern colors

    ax = plt.gca()
    # Plot only up to 1/20 of the spectrum to focus on low frequencies
    n_points = len(xf) // 20
    ax.plot(xf[1:n_points], np.abs(yf[1:n_points]), color=colors[0], linewidth=2.5, alpha=0.9)
    ax.axvline(if_freq, color=colors[1], linestyle='--', label=f'IF Freq ({if_freq:.1f} MHz)', linewidth=2.5, alpha=0.8)
    ax.set_title('(b) FFT Validation (Heterodyne IF)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    return ax
