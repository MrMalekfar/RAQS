"""
Time-domain heterodyne signal plotting (FIG-002).

Plotted observable: AC-coupled heterodyne response

    dS_n(t) = S_n(t) - mean(S_n),    S_n = worker observable

over the actual output time grid. The DC mean removed from
each laser path is stored in ``last_dc_offsets`` so the figure caption can
report it explicitly. The underlying simulation data is never modified.
"""

import numpy as np

import matplotlib.pyplot as plt
from typing import List, Optional

# DC offsets removed by the most recent AC-coupling call (one per path)
last_dc_offsets: Optional[List[float]] = None


def plot_phase_ramp(tlist: np.ndarray, time_traces: List[np.ndarray], colors=None) -> plt.Axes:
    """
    Plot AC-coupled worker observables for all declared laser paths.

    Args:
        tlist: Time array (μs)
        time_traces: List of real worker observables (one per laser path)
        colors: Optional list of colors for styling

    Returns:
        matplotlib Axes object
    """
    global last_dc_offsets
    ax = plt.gca()

    t = np.asarray(tlist)

    # Laser-path colors (production uses six laser paths, not "elements")
    if colors is not None and len(time_traces) <= len(colors):
        path_colors = list(colors[:len(time_traces)])
    else:
        base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        reps = len(time_traces) // len(base) + 1
        path_colors = (base * reps)[:len(time_traces)]

    dc_offsets: List[float] = []
    for i, (trace, color) in enumerate(zip(time_traces, path_colors)):
        # The worker already returns Im(<sigma21>); do not rectify it or apply
        # another transmission mapping. AC coupling is display-only.
        s_n = np.asarray(trace)
        dc_offsets.append(float(s_n.mean()))
        s_ac = s_n - s_n.mean()  # AC coupling: dS_n = S_n - <S_n>
        ax.plot(t, s_ac, color=color, linewidth=1.6,
                label=f'Laser path {i + 1}', alpha=0.85)

    last_dc_offsets = dc_offsets

    ax.set_title('(b) Time-Domain Heterodyne Signals (AC-coupled)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (μs)', fontsize=11)
    ax.set_ylabel(r'$\Delta S_n = S_n - \langle S_n \rangle$', fontsize=11)

    # FIG-003: legend moved OUTSIDE the main data region — below the axes,
    # centred, three columns — so the heterodyne oscillations stay visible.
    ax.legend(fontsize=8.5, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.margins(y=0.08)

    # Reserve bottom margin for the below-axes legend (standalone figure
    # only; the summary panel is laid out by constrained_layout).
    if len(ax.figure.axes) == 1:
        ax.figure.subplots_adjust(bottom=0.26)

    return ax
