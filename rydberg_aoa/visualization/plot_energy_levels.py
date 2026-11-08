"""
Energy level diagram plotting for the 4-level Rydberg AoA sensor.

Publication-style schematic with fixed basis identities. Rydberg heights
follow the signed RF energy ordering; |4> is not assumed above |3>.

Vertical distances are SCHEMATIC (not energy-to-scale): the two Rydberg
levels are drawn close together and the large visual gap between 5P and
the Rydberg pair is compressed.  Transition arrows are annotated with the
LIVE configuration values: probe/coupling wavelengths from
``laser_rf_config.LASER_PARAMS`` and the RF LO/signal field frequencies
each from their own config owner (``parameters.RF_LO_FREQUENCY`` /
``parameters.RF_SIGNAL_FREQUENCY``).  No placeholder or "N/A" text is
ever drawn.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from ..config import atomic_config as acfg  # ARCH-001: module access (no stale from-import)


def _wavelength_nm(level_key: str) -> float:
    """Return the live transition wavelength [nm] of the laser leaving the
    level `level_key` ('ground' -> probe laser, 'excited' -> coupling laser).

    Uses the live applied laser configuration only. Missing inputs propagate;
    an energy-derived wavelength must not substitute for an applied field.
    """
    from ..config.laser_rf_config import LASER_PARAMS
    key = "probe" if level_key == "ground" else "coupling"
    return float(LASER_PARAMS[key]["wavelength_nm"])


def _rf_field_frequencies() -> Dict[str, float]:
    """Return the live RF field frequencies, EACH read from its own config
    owner (no inference of one frequency from the other).

    Returns
    -------
    dict with keys:
        'f_lo_hz'    : LO field frequency [Hz]       (``RF_LO_FREQUENCY``)
        'f_signal_hz': signal field frequency [Hz]   (``RF_SIGNAL_FREQUENCY``)
        'f_if_hz'    : heterodyne IF |f_signal - f_LO| [Hz]
    """
    from ..config.parameters import RF_LO_FREQUENCY, RF_SIGNAL_FREQUENCY
    f_lo = float(RF_LO_FREQUENCY)
    f_sig = float(RF_SIGNAL_FREQUENCY)
    return {"f_lo_hz": f_lo, "f_signal_hz": f_sig, "f_if_hz": abs(f_sig - f_lo)}


def _term_label(level: Dict, state_index: int, show_f: bool) -> str:
    """Standard spectroscopy label for one level: ``|n>  nL_{j}[, F=..]``.

    The hyperfine quantum number F is shown only for the low-lying states
    (|1>, |2>); the Rydberg states are labelled in fine structure only.
    """
    n = int(level["n"])
    l = int(level["l"])
    j = float(level["j"])
    letters = 'SPDFGHIKLMNOQRTUVWXYZ'
    letter = letters[l] if 0 <= l < len(letters) else rf'(l={l})'
    two_j = int(round(2 * j))                 # 2j (j is half-integer)
    if two_j % 2:
        j_str = f"{two_j}/2"
    else:
        j_str = f"{two_j // 2}"
    f_str = ""
    if show_f:
        f_val = level.get("f", level.get("F"))
        if f_val is not None:
            f_str = rf", F={int(f_val)}"
    return rf"$|{state_index}\rangle\;\;{n}{letter}_{{{j_str}}}{f_str}$"


def plot_energy_levels(
    energy_unit: str = "MHz",
    show_transitions: bool = True,
    show_quantum_numbers: bool = True,
    ax: Optional[plt.Axes] = None,
    *, report: Optional[Dict] = None,
) -> plt.Figure:
    """
    Publication-style SCHEMATIC four-level ladder with fixed basis identities.
    Rydberg heights follow the signed RF energy ordering. An explicit report
    supplies frozen levels and applied fields without consulting live globals.
    The example labels below are basis roles, not an ascending-energy claim:

        |1> 5S_{1/2}, F=3   (bottom)
        |2> 5P_{3/2}, F=4
        |3> 26D_{5/2}
        |4> 27P_{3/2}       (top)

    Level heights are schematic (NOT energy-to-scale): the two Rydberg
    levels are drawn close together and the large visual gap between 5P
    and the Rydberg pair is compressed.  All annotated values come from
    the LIVE configuration (LASER_PARAMS / parameters.RF_LO_FREQUENCY /
    parameters.RF_SIGNAL_FREQUENCY); "N/A" is never drawn.
    """
    # --- schematic heights (bottom-to-top, arbitrary units) ---------------
    y_levels = {
        "ground": 0.0,
        "excited": 1.0,
        "rydberg1": 2.35,
        "rydberg2": 2.75,
    }
    x_left, x_right = 0.14, 0.62       # horizontal extent of level lines
    # ARCH-001: explicit initialization is required -- never draw stale levels.
    if report is None:
        acfg.require_atomic_physics_initialized()
        levels = acfg.ATOMIC_LEVELS
        signed_frequency = acfg.get_rf_orientation().signed_frequency_hz
    else:
        levels = report['levels']
        signed_frequency = report['rf_orientation']['signed_frequency_hz']
    # Canonical identities |3>, |4> stay fixed while schematic heights
    # respect the signed energy difference, including downward RF pairs.
    if signed_frequency < 0:
        y_levels['rydberg1'], y_levels['rydberg2'] = y_levels['rydberg2'], y_levels['rydberg1']
    order = ("ground", "excited", "rydberg1", "rydberg2")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        created_fig = True
    else:
        fig = ax.figure
    ax.set_facecolor("white")

    # --- horizontal level lines + labels -----------------------------------
    for idx, key in enumerate(order, start=1):
        y = y_levels[key]
        level = levels[key]
        show_f = key in ("ground", "excited")
        ax.hlines(y, x_left, x_right, colors="#1a1a1a", linewidth=2.6)
        ax.text(
            x_left - 0.03, y, "$|" + str(idx) + r"\rangle$",
            ha="right", va="center", fontsize=13, fontweight="bold",
            color="#1a1a1a",
        )
        ax.text(
            x_right + 0.04, y, _term_label(level, idx, show_f),
            ha="left", va="center", fontsize=12.5, color="#1a1a1a",
        )
        magnetic = ('mF', level.get('mf')) if show_f else ('mJ', level.get('mj'))
        if magnetic[1] is not None:
            ax.text(x_right + .04, y - .11, f'{magnetic[0]}={magnetic[1]:g}',
                    fontsize=9, va='top')

    # --- optical transitions -----------------------------------------------
    if show_transitions:
        wl_p = (_wavelength_nm('ground') if report is None else report['applied_fields']['probe']['wavelength_nm'])
        wl_c = (_wavelength_nm('excited') if report is None else report['applied_fields']['coupling']['wavelength_nm'])

        # Probe arrow |1> -> |2>, label in the free band between the levels.
        x_p = 0.24
        ax.annotate(
            "", xy=(x_p, y_levels["excited"] - 0.06),
            xytext=(x_p, y_levels["ground"] + 0.06),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=1.9),
        )
        ax.text(
            x_p + 0.025, (y_levels["ground"] + y_levels["excited"]) / 2,
            "Probe\n" + r"$\lambda_p$ = " + f"{wl_p:.6f} nm",
            ha="left", va="center", fontsize=10.5, color="#c0392b",
        )

        # Coupling arrow |2> -> |3>, label in the free band 5P -> Rydberg.
        x_c = 0.38
        ax.annotate(
            "", xy=(x_c, y_levels["rydberg1"] - 0.06),
            xytext=(x_c, y_levels["excited"] + 0.06),
            arrowprops=dict(arrowstyle="-|>", color="#1273b5", linewidth=1.9),
        )
        ax.text(
            x_c + 0.025, (y_levels["excited"] + y_levels["rydberg1"]) / 2,
            "Coupling\n" + r"$\lambda_c$ = " + f"{wl_c:.6f} nm",
            ha="left", va="center", fontsize=10.5, color="#1273b5",
        )

        # RF transition |3> <-> |4>: two slightly offset double-headed
        # arrows, ONE per field, each labelled from its OWN config owner.
        rf = (_rf_field_frequencies() if report is None else {
            'f_lo_hz': report['applied_fields']['lo']['frequency_hz'],
            'f_signal_hz': report['applied_fields']['signal']['frequency_hz'],
            'f_if_hz': abs(report['signed_if_mhz']) * 1e6})
        f_lo_ghz = rf["f_lo_hz"] / 1e9
        f_sig_ghz = rf["f_signal_hz"] / 1e9
        f_if_mhz = rf["f_if_hz"] / 1e6
        x_lo, x_sig = 0.18, 0.24
        for x_a, color in ((x_lo, "#7d3c98"), (x_sig, "#b03a2e")):
            ax.annotate(
                "", xy=(x_a, max(y_levels['rydberg1'], y_levels['rydberg2']) - .05),
                xytext=(x_a, min(y_levels['rydberg1'], y_levels['rydberg2']) + .05),
                arrowprops=dict(arrowstyle="<|-|>", color=color, linewidth=1.5),
            )
        # RF field label in the free band ABOVE the Rydberg pair (no overlap
        # with any level label; 4-decimal GHz resolves the 0.1 MHz IF).
        ax.text(
            x_sig + 0.06, 3.10,
            "RF fields ($|3\\rangle \\leftrightarrow |4\\rangle$)\n"
            + r"LO: $f_{\mathrm{LO}}$ = " + f"{f_lo_ghz:.4f} GHz\n"
            + r"Signal: $f_{\mathrm{sig}}$ = " + f"{f_sig_ghz:.4f} GHz\n"
            + r"$|f_{\mathrm{IF}}|$ = " + f"{f_if_mhz:.3f} MHz\n"
            + f'(E4-E3)/h = {signed_frequency/1e9:+.6f} GHz',
            ha="left", va="center", fontsize=10, color="#4a235a",
        )

    # --- axes cosmetics ------------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, 3.55)
    ax.axis("off")
    ax.set_title(
        "Four-level Rydberg sensor (schematic, not to scale)"
        + (f" — {report['atom']['species']}" if report is not None and 'atom' in report else ""),
        fontsize=13.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    if not created_fig:
        # Embedded: the parent figure controls layout.
        return fig
    # Standalone figure: manage its own layout.  When embedded, the parent
    # figure (e.g. constrained_layout) controls the geometry, and calling
    # tight_layout on it would fight the parent's layout engine.
    fig.tight_layout()
    return fig


def plot_energy_levels_comparison(energy_unit: str = "MHz") -> plt.Figure:
    """
    Supporting comparison view: normalized energies of several Rydberg n-states
    near the optimized operating point (Rydberg1 / Rydberg2).

    The primary four-level schematic is ``plot_energy_levels()``.
    """
    from ..analytical.energy import RydbergEnergyCalculator
    from ..config import atomic_config as _acfg

    _acfg.require_atomic_physics_initialized()
    if _acfg.ATOM_SPEC.species != 'Rb85':
        from ..exceptions import PhysicsConfigurationError
        raise PhysicsConfigurationError('ATOM-001: analytical energy comparison is Rb85-only; the primary ARC-based schematic remains available.')
    rydberg1_n = int(_acfg.ATOMIC_LEVELS["rydberg1"]["n"])
    rydberg2_n = int(_acfg.ATOMIC_LEVELS["rydberg2"]["n"])
    n_center   = int(round((rydberg1_n + rydberg2_n) / 2))
    n_range    = np.arange(n_center - 2, n_center + 3)

    letter_map = ["S", "P", "D", "F", "G", "H", "I"]
    calc       = RydbergEnergyCalculator()

    energies: List[float] = []
    labels:   List[str]   = []

    for n in n_range:
        try:
            cm = calc.calculate_energy_cm(n, 0, 0.5) - calc.calculate_energy_cm(5, 0, 0.5)
            if energy_unit == "GHz":
                val = cm * 299792458 / 1e9
            elif energy_unit == "MHz":
                val = cm * 299792458 / 1e6
            elif energy_unit == "eV":
                val = cm * 1.9863e-23 / 1.602e-19
            else:
                val = cm
            energies.append(float(val))
            letter = letter_map[0]
            labels.append(
                "{}{}{}".format(
                    n, letter, int(2 * _acfg.ATOMIC_LEVELS["rydberg1"]["j"] + 1) // 2
                )
            )
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if not energies:
        ax.text(0.5, 0.5, "levels unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        fig.tight_layout()
        return fig

    y_pos = np.arange(len(energies))
    scale = max(energies) if max(energies) > 0 else 1.0

    ax.hlines(y_pos, 0, np.array(energies) / scale,
              linewidth=2, colors="#1f77b4", alpha=0.75)

    for i, (e, lab) in enumerate(zip(energies, labels)):
        ax.text(e / scale + 0.01, i, "{:.1f}".format(e),
                ha="left", va="center", fontsize=9)
        ax.text(-0.01, i, lab, ha="right", va="center",
                fontsize=10, fontweight="bold")

    if rydberg1_n in n_range:
        idx1 = int(list(n_range).index(rydberg1_n))
        ax.hlines(y_pos[idx1], 0, energies[idx1] / scale,
                  linewidth=4, colors="#d62728",
                  label="System: {}".format(labels[idx1]))

    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.6, len(energies) + 0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)

    unit_labels = {
        "MHz":   "Frequency (MHz, normalized)",
        "GHz":   "Frequency (GHz, normalized)",
        "cm^-1": "Wavenumber (cm-1, normalized)",
    }
    ax.set_xlabel(unit_labels.get(energy_unit, energy_unit))
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig
