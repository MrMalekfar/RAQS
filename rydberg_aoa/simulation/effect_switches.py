"""PHY-SW-001 — Physical-effect switch registry and analytical parity profile.

Purpose
-------
Make an apples-to-apples numerical-vs-analytical comparison possible by
recording, in ONE production location, which physical effects the CURRENT
analytical model actually implements, and which numerical effect switches
must therefore be ON/OFF for a controlled comparison.

This module changes no physics. The switches themselves live in
`config/parameters.py` (global defaults, all True = accepted production
model) and are threaded as explicit optional arguments through
`simulation.steady_state.simulate_spectrum` and
`physics.lindblad.calculate_dephasing_rates` /
`physics.lindblad.resolve_effective_transit_rate`.

Effect Matrix (audited against the code, PHY-SW-001)
----------------------------------------------------
| Effect                      | Numerical                              | Analytical                                  |
|-----------------------------|----------------------------------------|---------------------------------------------|
| Doppler                     | Maxwell ensemble averaging (DOP-002I)  | ABSENT — zero-velocity single-atom solution |
| Transit reset (DPH-002B)    | reset collapse ops C_j=sqrt(Gt)|1><j|   | ABSENT — no loss/repopulation channel       |
| Laser linewidth dephasing   | gamma_lw in p_dephasing projectors     | PRESENT — inside gamma_2_eff                |
| Collision dephasing         | gamma_col in p_dephasing projectors    | PRESENT — inside gamma_2_eff                |
| Spontaneous decay           | mandatory sqrt(Gamma) jump ops         | mandatory — inside gamma_2_eff (Gamma_21)   |
| RF coupling                 | mandatory, Omega_RF=0 = RF-off         | mandatory — Omega_rf_complex in polynomial  |

SCIENTIFIC PARITY WARNING: even with this profile applied, numerical and
analytical results are NOT expected to agree: the C-003 analytical RF-off
defect remains open and the analytical derivation itself may need
re-derivation. This profile only equalizes the physical-effect CONTENT.
"""

from ..config import parameters as params


# All effects ON — the accepted production numerical model.
FULL_EFFECTS = {
    "doppler_enabled": True,
    "transit_enabled": True,
    "laser_linewidth_dephasing_enabled": True,
    "collision_dephasing_enabled": True,
}

# Analytical parity profile — determined from the ACTUAL analytical model
# (analytical/solutions.py + the analytical branch of simulate_spectrum),
# NOT guessed:
#   * Doppler: OFF — the analytical path is the zero-velocity reference
#     solution and is not Maxwell-averaged (steady_state.py docstring).
#   * Transit: OFF — the analytical ladder polynomial contains no
#     loss/repopulation (reset) channel.
#   * Laser linewidth dephasing: ON — represented with the same physical
#     meaning via gamma_2_eff = Gamma_21 + p_dephasing[0], whose linewidth
#     term is the probe laser linewidth.
#   * Collision dephasing: ON — represented with the same physical meaning
#     via the collision term inside p_dephasing[0] -> gamma_2_eff.
ANALYTICAL_PARITY_EFFECTS = {
    "doppler_enabled": False,
    "transit_enabled": False,
    "laser_linewidth_dephasing_enabled": True,
    "collision_dephasing_enabled": True,
}


def full_effects():
    """Return a copy of the all-ON (production) effect switch set."""
    return dict(FULL_EFFECTS)


def analytical_parity_effects():
    """Return a copy of the analytical-parity effect switch set.

    These switches, passed to `simulate_spectrum(**kwargs)`, make the
    NUMERICAL model solve the same physical problem as the CURRENT
    analytical solution. See the scientific parity warning in the module
    docstring: agreement is NOT expected yet (C-003 open).
    """
    return dict(ANALYTICAL_PARITY_EFFECTS)


def describe_effects(effects=None):
    """Report the exact effective switch state (for diagnostics/tests).

    Args:
        effects: optional dict with any subset of the four switch keys;
            missing keys resolve to the current configuration values.

    Returns:
        dict: the four switch keys with their effective boolean values.
    """
    resolved = {
        "doppler_enabled": params.DOPPLER_ENABLED,
        "transit_enabled": params.TRANSIT_ENABLED,
        "laser_linewidth_dephasing_enabled": params.LASER_LINEWIDTH_DEPHASING_ENABLED,
        "collision_dephasing_enabled": params.COLLISION_DEPHASING_ENABLED,
    }
    if effects:
        for key, value in effects.items():
            if key not in resolved:
                raise ValueError(
                    "unknown effect switch %r; valid keys: %r" % (key, sorted(resolved))
                )
            resolved[key] = bool(value)
    return resolved
