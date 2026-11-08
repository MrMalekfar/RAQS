import numpy as np
from ...config import parameters as params, constants as const
from ...config import atomic_config as acfg
from ...config.laser_rf_config import LASER_PARAMS

def calculate_dephasing_rates(
    temperature_k=None,
    beam_waist_mm=None,
    atom_mass_kg=None,
    n_density_m3=None,
    rydberg_n=None,
    gamma_probe_linewidth=None,
    gamma_coupling_linewidth=None,
    gamma_rf_linewidth=None,
    laser_linewidth_enabled=None,
    collision_dephasing_enabled=None,
):
    """
    Physics-based pure-dephasing budget [γ₂, γ₃, γ₄] in 1/μs.

    DPH-002B: TRANSIT IS NO LONGER PART OF THIS BUDGET. Atomic beam transit
    is a loss/repopulation process and is represented exclusively by reset
    collapse operators built from `calculate_transit_rate_per_us()` and
    applied centrally in `create_sensor_model(transit_rate_per_us=...)`.
    Including a transit contribution here as well would double-count it.

    γ₂ = Γ_probe_lw   + γ_collision
    γ₃ = Γ_coupling_lw + γ_collision
    γ₄ = Γ_rf_lw      + γ_collision

    with γ_collision = n₀·σ(n)·⟨v_rel⟩ (Rydberg collisional broadening,
    σ(n) = 1e-14·(n/50)⁴ cm²). N·σ·⟨v_rel⟩ is an inverse-time event rate
    [s⁻¹]; it receives only the s⁻¹ → μs⁻¹ conversion (no 2π — the former
    transit/collision 2π was a unit-convention error, removed in DPH-002A/B).

    PHY-SW-001: the two dephasing COMPONENTS can be independently gated via
    `laser_linewidth_enabled` / `collision_dephasing_enabled`. A disabled
    component contributes exactly 0.0 to γ₂/γ₃/γ₄; formulas, units and the
    spontaneous decay rates are untouched. Default (None) resolves each gate
    from `params.LASER_LINEWIDTH_DEPHASING_ENABLED` /
    `params.COLLISION_DEPHASING_ENABLED` (both True in production), which
    reproduces the legacy arithmetic exactly.

    RF-002 injection contract: every implicit config input can be supplied
    explicitly; omitted arguments fall back lazily to the identical
    configuration source.

    Args:
        temperature_k: vapour-cell temperature [K] (default const.VAPOR_CELL_TEMPERATURE_K)
        atom_mass_kg: atom mass [kg] (default acfg.ATOM.mass)
        n_density_m3: number density [m⁻³] (default params.CALCULATED_N0)
        rydberg_n: principal quantum number of Rydberg state |3⟩ (default ATOMIC_LEVELS['rydberg1']['n'])
        gamma_probe_linewidth: probe laser linewidth [rad/μs] (default params.GAMMA_PROBE_LINEWIDTH)
        gamma_coupling_linewidth: coupling laser linewidth [rad/μs] (default params.GAMMA_COUPLING_LINEWIDTH)
        gamma_rf_linewidth: RF/LO linewidth [rad/μs] (default params.GAMMA_RF_LINEWIDTH)
        laser_linewidth_enabled: PHY-SW-001 gate for the linewidth component
            (default None → params.LASER_LINEWIDTH_DEPHASING_ENABLED)
        collision_dephasing_enabled: PHY-SW-001 gate for the collision component
            (default None → params.COLLISION_DEPHASING_ENABLED)

    Returns:
        list: [gamma_2, gamma_3, gamma_4] pure-dephasing rates [1/μs]
              (enabled components only — transit excluded, see above)
    """
    # Lazy fallbacks: omitted arguments resolve to the identical legacy
    # source, so the default path executes the identical arithmetic.
    if temperature_k is None:
        temperature_k = const.VAPOR_CELL_TEMPERATURE_K
    if atom_mass_kg is None:
        atom_mass_kg = acfg.ATOM.mass
    if n_density_m3 is None:
        n_density_m3 = params.CALCULATED_N0
    if rydberg_n is None:
        rydberg_n = acfg.ATOMIC_LEVELS['rydberg1']['n']
    if gamma_probe_linewidth is None:
        gamma_probe_linewidth = params.GAMMA_PROBE_LINEWIDTH
    if gamma_coupling_linewidth is None:
        gamma_coupling_linewidth = params.GAMMA_COUPLING_LINEWIDTH
    if gamma_rf_linewidth is None:
        gamma_rf_linewidth = params.GAMMA_RF_LINEWIDTH
    # PHY-SW-001 component gates: None → production configuration flags.
    if laser_linewidth_enabled is None:
        laser_linewidth_enabled = params.LASER_LINEWIDTH_DEPHASING_ENABLED
    if collision_dephasing_enabled is None:
        collision_dephasing_enabled = params.COLLISION_DEPHASING_ENABLED
    if collision_dephasing_enabled and acfg.ATOM_SPEC.species != 'Rb85':
        from ...exceptions import PhysicsConfigurationError
        raise PhysicsConfigurationError('ATOM-001: collision approximation is validated only for the existing Rb85 model; use explicit rates for other isotopes.')

    # 1. COLLISIONAL DEPHASING
    # σ(n) = 1e-14·(n/50)⁴ cm² — Rydberg collision cross-section scaling
    sigma_cm2 = 1e-14 * (rydberg_n / 50)**4
    sigma_m2 = sigma_cm2 * 1e-4
    # ⟨v_rel⟩ = √(8·k_B·T / π·m) — mean relative thermal speed
    v_relative = np.sqrt(8 * const.k_B * temperature_k / (np.pi * atom_mass_kg))
    gamma_collision_hz = n_density_m3 * sigma_m2 * v_relative  # [s⁻¹] inverse-time event rate
    GAMMA_COLLISION = gamma_collision_hz / const.US_TO_S_FACTOR  # [1/μs] — no 2π

    # 2. TOTAL RATES (enabled components ONLY — transit handled by reset ops)
    # PHY-SW-001: with both gates True this is bit-identical to the legacy
    # `linewidth + collision` addition (same operands, same order).
    gamma_2 = (gamma_probe_linewidth if laser_linewidth_enabled else 0.0) \
        + (GAMMA_COLLISION if collision_dephasing_enabled else 0.0)
    gamma_3 = (gamma_coupling_linewidth if laser_linewidth_enabled else 0.0) \
        + (GAMMA_COLLISION if collision_dephasing_enabled else 0.0)
    gamma_4 = (gamma_rf_linewidth if laser_linewidth_enabled else 0.0) \
        + (GAMMA_COLLISION if collision_dephasing_enabled else 0.0)

    # Return the numbers (not Qobjs!)
    return [gamma_2, gamma_3, gamma_4]
