"""Connect explicit profiles to existing thermal budgets and spectrum solvers."""

from dataclasses import dataclass, asdict
from types import SimpleNamespace
import numpy as np

from ..config.laboratory import _real
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError


@dataclass(frozen=True)
class ProfileMedium:
    temperature_k: float
    atom_mass_kg: float
    number_density_m3: float
    cell_length_m: float


def resolve_profile_medium(profile, ac):
    """Use the actual atom at the declared temperature; never substitute density/mass."""
    env = profile.environment
    if env is None:
        return None, profile.run_config
    from ..config.atom_species import arc_density, effective_atom_spec, require_profile_atom
    require_profile_atom(profile, ac)
    try:
        mass = _real(ac.ATOM.mass, 'ARC atom mass [kg]', positive=True)
    except (AttributeError, PhysicsConfigurationError) as exc:
        raise PhysicsCalculationError('LAB-007: actual ARC atom mass is unavailable or invalid; no mass substitute.') from exc
    try:
        density = arc_density(ac.ATOM, effective_atom_spec(profile), env.temperature_k)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f'LAB-007: ARC getNumberDensity at T={env.temperature_k} K failed.') from exc
    try:
        density = _real(density, 'ARC number density [m^-3]', nonnegative=True)
    except PhysicsConfigurationError as exc:
        raise PhysicsCalculationError(f'LAB-007: invalid ARC density at T={env.temperature_k} K: {density!r}.') from exc
    medium = ProfileMedium(env.temperature_k, mass, density, env.cell_length_m)
    if env.rate_source == 'explicit':
        return medium, profile.run_config
    from ..physics.lindblad.calculate_dephasing_rates import calculate_dephasing_rates
    from ..physics.lindblad.calculate_transit_rate import calculate_transit_rate_per_us
    dephasing = calculate_dephasing_rates(
        temperature_k=medium.temperature_k, atom_mass_kg=mass, n_density_m3=density,
        rydberg_n=ac.ATOMIC_LEVELS['rydberg1']['n'],
        gamma_probe_linewidth=env.linewidth_rates_per_us[0],
        gamma_coupling_linewidth=env.linewidth_rates_per_us[1],
        gamma_rf_linewidth=env.linewidth_rates_per_us[2],
        laser_linewidth_enabled=env.linewidth_enabled, collision_dephasing_enabled=env.collision_enabled)
    transit = (calculate_transit_rate_per_us(
        temperature_k=medium.temperature_k, beam_waist_mm=profile.probe.beam_waist_mm,
        atom_mass_kg=mass) if env.transit_enabled else 0.)
    return medium, profile.run_config.resolve(dephasing, transit)


def run_profile_spectrum(profile, config, medium, ac):
    """RF-off/on numerical spectra share the temporal model's rates and base detunings.

    The signed IF signal is absent from this stationary LO-only scan. Doppler
    averaging acts on coherence before a single Beer-Lambert propagation.
    The optional analytical curve is the existing limited zero-velocity formula.
    """
    from ..config.magnetic_core import require_four_state_profile
    require_four_state_profile(profile)
    from ..config.noise import require_noise_execution_supported
    require_noise_execution_supported(profile.noise)
    from .steady_state import simulate_spectrum
    from ..physics.susceptibility.absorption_const import calculate_absorption_constant
    spec = profile.spectrum
    if spec is None:
        return None
    # Confirm complete runtime before a cache reader could trigger legacy repair.
    from .runner import SimulationRunner
    request = profile.request if profile.mode == 'experimental' else profile
    before = SimulationRunner._current_explicit_runtime(request, profile.mode)
    from ..config.profile_radiative import active_radiative_rates, radiative_report, named_internal_rates
    rates = active_radiative_rates(config, before, ac, profile)
    dipole = ac.get_cached_dipole_matrix()['probe']['dipole_si']
    prefactors = calculate_absorption_constant(
        mu_21_c_m=dipole, omega_p_rad_per_s=ac.OMEGA_P * 1e6,
        n0_m3=medium.number_density_m3, cell_length_m=medium.cell_length_m,
        lambda_p_m=profile.probe.wavelength_nm * 1e-9)
    if not np.all(np.isfinite(prefactors)):
        raise PhysicsCalculationError('LAB-007: nonfinite transmission prefactors.')
    settings = SimpleNamespace(**{key: value for key, value in asdict(spec).items()
                                 if key not in ('probe_offsets_mhz', 'doppler_enabled', 'analytical_enabled')},
                               temperature_k=medium.temperature_k, atom_mass_kg=medium.atom_mass_kg)
    output = {'probe_offsets_mhz': np.asarray(spec.probe_offsets_mhz),
              'base_detunings_rad_per_us': before.detunings,
              'dephasing_rates_per_us': config.dephasing_rates_per_us,
              'transit_rate_per_us': config.transit_rate_per_us,
              'transmission_prefactors': prefactors,
              'analytical_contract': 'existing zero-velocity formula; excludes Doppler, transit and full upper-state dephasing'}
    if config.radiative is not None:
        output['radiative_model'] = radiative_report(config.radiative)
        output['decay_rates_per_us'] = named_internal_rates(config.radiative)
        output['analytical_contract'] = 'not computed: reduced analytical formula does not implement prepared radiative graph'
    backend_options = {}
    if spec.steady_state_backend is not None:
        backend_options['steady_state_backend'] = spec.steady_state_backend
        output['steady_state_backend'] = spec.steady_state_backend
    for name, omega in (('rf_off', 0.), ('rf_on', ac.OMEGA_RF_BASE)):
        scan, numerical, analytical = simulate_spectrum(
            omega, spec.probe_offsets_mhz, (ac.OMEGA_P, ac.OMEGA_C, ac.OMEGA_RF_BASE),
            before.detunings, rates,
            config.dephasing_rates_per_us, None, transit_rate_per_us=config.transit_rate_per_us,
            transit_enabled=True, doppler_enabled=spec.doppler_enabled, rf_orientation=ac.get_rf_orientation(),
            doppler_settings=settings, transmission_prefactors=prefactors,
            analytical_enabled=spec.analytical_enabled, **backend_options)
        if not np.array_equal(scan, output['probe_offsets_mhz']) or not np.all(np.isfinite(numerical)):
            raise PhysicsCalculationError('LAB-007: invalid spectrum output; no substituted scan or curve.')
        output[name] = numerical
        output[name + '_analytical'] = analytical
    if SimulationRunner._current_explicit_runtime(request, profile.mode) != before:
        raise PhysicsCalculationError('LAB-007: atomic runtime changed during spectrum calculation.')
    active_radiative_rates(config, before, ac, profile)
    return output
