"""PIN detector with independent boxcar charge samples and explicit SI units.

Optical propagation is the declared lumped, quasistatic Beer-Lambert model,
not a Maxwell-Bloch propagation solver. Each certified point represents the
constant optical intensity held over its centered integration bin. There is
no undocumented analog anti-alias filter, gain sign, or saturation repair.
"""

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.constants import elementary_charge as Q_E, Boltzmann as K_B, Planck as H_SI, speed_of_light as C_SI

from ..config.laboratory import _real, _integer
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError
from .noise_random import NoiseRandomPlan


@dataclass(frozen=True)
class DetectorContext:
    incident_probe_power_w: float
    od_factor: float
    susceptibility_factor: float
    probe_wavelength_m: float

    def __post_init__(self):
        for name in ('incident_probe_power_w', 'od_factor', 'susceptibility_factor', 'probe_wavelength_m'):
            object.__setattr__(self, name, _real(getattr(self, name), 'NOISE-004 '+name,
                positive=name in ('incident_probe_power_w', 'probe_wavelength_m'), nonnegative=name == 'od_factor'))


def resolve_detector_context(profile, atomic, medium):
    if profile.noise is None or not profile.noise.physical_readout_enabled:
        return None
    if medium is None:
        raise PhysicsConfigurationError('NOISE-004: declared medium required; no density/length defaults.')
    from ..physics.susceptibility.absorption_const import calculate_absorption_constant
    dipole = atomic.get_cached_dipole_matrix()['probe']['dipole_si']
    od, chi = calculate_absorption_constant(mu_21_c_m=dipole,
        omega_p_rad_per_s=atomic.OMEGA_P*1e6, n0_m3=medium.number_density_m3,
        cell_length_m=medium.cell_length_m, lambda_p_m=profile.probe.wavelength_nm*1e-9)
    return DetectorContext(profile.probe.power_mw/1000, od, chi, profile.probe.wavelength_nm*1e-9)


def require_detector_context(noise, run_config, context):
    from ..config.detector import validate_detector_sampling
    validate_detector_sampling(noise, run_config)
    if noise is not None and noise.physical_readout_enabled and not isinstance(context, DetectorContext):
        raise PhysicsConfigurationError('NOISE-004: explicit resolved DetectorContext required; no optical calibration fallback.')
    if noise is not None and noise.physical_readout_enabled:
        validate_pin_efficiency(noise.detector, context)


def validate_pin_efficiency(detector, context):
    return pin_efficiency(detector.responsivity_a_per_w, context.probe_wavelength_m)


def pin_efficiency(responsivity_a_per_w, wavelength_m):
    try:
        eta = responsivity_a_per_w*(H_SI*C_SI/Q_E)/wavelength_m
    except (OverflowError, ZeroDivisionError) as exc:
        raise PhysicsConfigurationError('NOISE-004: PIN quantum efficiency cannot be represented.') from exc
    if not math.isfinite(eta) or not 0 < eta <= 1:
        raise PhysicsConfigurationError('NOISE-004: responsivity implies PIN quantum efficiency outside (0,1]; no APD gain assumed.')
    return eta


def detector_forward(atomic_traces, detector, context, *, probe_intensity=None):
    """Return received W, noiseless A, selected A/V samples, transmission.

Use the canonical transmission function with purely imaginary <sigma21>:
the existing susceptibility factor is real, so Re coherence cannot affect
the absorption in this declared approximation. No sign flip or clipping.
NOISE-007: an optional shared N-sample positive intensity factor changes both
incident power P0*r and susceptibility K0/sqrt(r), using the same realization.
"""
    from ..config.detector import PINDetector
    if not isinstance(detector, PINDetector) or not isinstance(context, DetectorContext):
        raise PhysicsConfigurationError('NOISE-004: validated DetectorContext required.')
    validate_pin_efficiency(detector, context)
    try:
        y = np.asarray(atomic_traces)
        if y.ndim != 2 or not all(y.shape) or y.dtype.kind not in 'iuf' or not np.isfinite(y).all():
            raise PhysicsCalculationError('NOISE-004: finite real atomic M-by-N trace required.')
        from ..physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
        if probe_intensity is None:
            transmission = np.array([calculate_transmission_from_rho21(1j*float(v),
                absorption_prefactors=(context.od_factor, context.susceptibility_factor))
                for v in y.flat]).reshape(y.shape)
        else:
            if isinstance(probe_intensity, (list, tuple)):
                if any(isinstance(v, (bool, np.bool_)) or
                       not isinstance(v, (int, float, np.integer, np.floating)) for v in probe_intensity):
                    raise PhysicsCalculationError('NOISE-007: intensity vector requires real numeric scalars; no coercion.')
            elif not isinstance(probe_intensity, np.ndarray) or np.ma.isMaskedArray(probe_intensity):
                raise PhysicsCalculationError('NOISE-007: an unmasked numeric intensity vector is required.')
            r = np.asarray(probe_intensity)
            if (r.ndim != 1 or len(r) != y.shape[1] or r.dtype.kind not in 'iuf' or
                    not np.isfinite(r).all() or np.any(r <= 0)):
                raise PhysicsCalculationError('NOISE-007: shared positive finite N-sample probe intensity required; no broadcast/repair.')
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                chi = context.susceptibility_factor/np.sqrt(r)
            if not np.isfinite(chi).all():
                raise PhysicsCalculationError('NOISE-007: dynamic susceptibility prefactor is nonfinite.')
            transmission = np.array([[calculate_transmission_from_rho21(1j*float(v),
                absorption_prefactors=(context.od_factor, float(k))) for v, k in zip(row, chi)] for row in y])
        with np.errstate(over='raise', invalid='raise'):
            if probe_intensity is None:
                power = context.incident_probe_power_w*detector.collection_efficiency*transmission
            else:
                incident_power = context.incident_probe_power_w*r
                if not np.isfinite(incident_power).all():
                    raise PhysicsCalculationError('NOISE-007: incident probe power is nonfinite; no substitute.')
                power = incident_power[None, :]*detector.collection_efficiency*transmission
            current = detector.responsivity_a_per_w*power+detector.dark_current_a
            samples = current*detector.transimpedance_v_per_a if detector.output_units == 'V' else current.copy()
        if any(not np.isfinite(a).all() or np.any(a < 0) for a in (power, current, samples)):
            raise PhysicsCalculationError('NOISE-004: invalid forward readout; no clipping or substitute.')
        return power, current, samples, transmission
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-004: optical/current forward model failed.') from exc


def apply_pin_readout(atomic_traces, config, context, *, trial=0, probe_intensity=None, poisson_quantiles=False):
    """Exact Poisson charge counts; Gaussian white electronics/shunt current.

One-sided current ASD gives variance ASD^2/(2*tau_s). Poisson variance of
averaged current is q*I/tau_s, equivalent to PSD 2*q*I integrated over boxcar
ENBW 1/(2*tau_s). Samples/paths are independent; counts are never replaced
by a Gaussian approximation and negative electronic samples are not clipped.
"""
    trial = _integer(trial, 'NOISE-004 trial')
    if type(poisson_quantiles) is not bool:
        raise PhysicsConfigurationError('NOISE-016: Poisson diagnostic flag must be boolean.')
    from ..config.noise import require_noise_execution_supported
    require_noise_execution_supported(config)
    if config is None or config.schema_version != 2 or not config.physical_readout_enabled or trial >= config.trials:
        raise PhysicsConfigurationError('NOISE-004: physical readout must be enabled and trial in range.')
    detector = config.detector
    forward_kwargs = {} if probe_intensity is None else {'probe_intensity': probe_intensity}
    power, current, clean, transmission = detector_forward(atomic_traces, detector, context, **forward_kwargs)
    tau = detector.integration_time_us*1e-6
    if tau <= 0 or not math.isfinite(tau):
        raise PhysicsCalculationError('NOISE-004: integration time not representable in seconds.')
    plan = NoiseRandomPlan(config)
    total = np.zeros(current.shape)
    payload = {'clean_atomic_time_traces': np.asarray(atomic_traces), 'clean_optical_power_w': power,
               'noiseless_current_a': current, 'clean_detector_time_traces': clean,
               'clean_detector_transmission': transmission}
    if probe_intensity is not None:
        payload.update(incident_probe_power_w=context.incident_probe_power_w*np.asarray(probe_intensity),
                       probe_rabi_multiplier=np.sqrt(probe_intensity))
    variances = {}
    applied = False
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for source in config.sources:
                if source.name not in config.active_sources:
                    continue
                component = np.zeros_like(current)
                if source.name in ('photon_shot', 'dark_shot'):
                    nominal = (detector.responsivity_a_per_w*power if source.name == 'photon_shot'
                               else np.full(current.shape, detector.dark_current_a))
                    expected_counts = nominal*(tau/Q_E)
                    variance = nominal*(Q_E/tau)
                    counts = np.zeros(current.shape, dtype=np.int64)
                    if np.any(expected_counts > 0):
                        applied = True
                        for path in range(len(current)):
                            generator=plan.generator(source.name, trial=trial, path=path)
                            if poisson_quantiles:
                                counts[path]=poisson_quantile_counts(generator.random(current.shape[1]),expected_counts[path])
                            else:
                                counts[path]=generator.poisson(expected_counts[path])
                        component = counts*(Q_E/tau)-nominal
                    payload[source.name+'_electron_counts'] = counts
                    variances[source.name] = {'min_a2': float(variance.min()), 'max_a2': float(variance.max())}
                else:
                    spec = source.specification
                    psd = (spec.asd_a_per_sqrt_hz**2 if source.name == 'electronic_current'
                           else 4*K_B*spec.temperature_k/spec.shunt_resistance_ohm)
                    variance = psd/(2*tau)
                    variances[source.name] = {'a2': float(variance)}
                    if variance > 0:
                        applied = True
                        for path in range(len(current)):
                            component[path] = math.sqrt(variance)*plan.generator(source.name, trial=trial, path=path).standard_normal(current.shape[1])
                payload[source.name+'_current_noise_a'] = component
                total += component
            measured_current = current+total
            gain = detector.transimpedance_v_per_a if detector.output_units == 'V' else 1.
            additive = total*gain
            measured = clean+additive
            measured_voltage = measured_current*detector.transimpedance_v_per_a
        if any(not np.isfinite(a).all() for a in (total, measured, measured_current, measured_voltage)):
            raise PhysicsCalculationError('NOISE-004: nonfinite detector samples; no repair.')
        payload.update(measured_current_a=measured_current, measured_voltage_v=measured_voltage)
        metadata = {'model': 'pin_boxcar_sample_hold', 'units': detector.output_units,
                    'noise_applied': applied, 'trial_index': trial, 'trials_executed': 1,
                    'detector': asdict(detector), 'forward_context': asdict(context),
                    'pin_quantum_efficiency': validate_pin_efficiency(detector, context),
                    'integration_time_s': tau, 'boxcar_enbw_hz': .5/tau,
                    'anti_alias_model': 'boxcar charge integration only; no ideal brickwall or extra analog filter',
                    'temporal_optical_model': 'lumped quasistatic Beer-Lambert, per-bin held intensity; no propagation; temporal ensemble status in run receipt',
                    'variance_current_a2': variances, 'random_plan': plan.metadata(),
                    'spectrum_readout_noise_applied': False}
        metadata['probe_intensity_modulated'] = probe_intensity is not None
        if poisson_quantiles:metadata['diagnostic_poisson_coupling']='shared uniforms via inverse CDF; production Poisson RNG not altered'
        return measured, additive, metadata, payload
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-004: physical detector generation failed; no Gaussian/clean fallback.') from exc


def poisson_quantile_counts(uniforms,means):
    """Diagnostic-only Poisson coupling, without Gaussian approximation/clip."""
    from scipy.stats import poisson
    if np.ma.isMaskedArray(uniforms) or np.ma.isMaskedArray(means):
        raise PhysicsCalculationError('NOISE-016: masked quantile evidence is not a complete realization.')
    try:u,m=np.asarray(uniforms),np.asarray(means)
    except Exception as exc:raise PhysicsCalculationError('NOISE-016: invalid quantile array structure.') from exc
    if (u.shape!=m.shape or u.ndim!=1 or not len(u) or any(a.dtype.kind not in 'iuf' for a in (u,m)) or
            not np.isfinite(u).all() or not np.isfinite(m).all() or np.any(u<0) or np.any(u>=1) or np.any(m<0)):
        raise PhysicsCalculationError('NOISE-016: finite compatible uniforms in [0,1) and nonnegative means required.')
    # Generalized inverse at u=0 is the support minimum 0. scipy's -1
    # endpoint convention is not a physical electron count.
    counts=np.zeros(u.shape)
    active=(u>0)&(m>0)
    counts[active]=poisson.ppf(u[active],m[active])
    if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts>2**53-1) or np.any(counts!=np.floor(counts)):
        raise PhysicsCalculationError('NOISE-016: Poisson quantile not exactly representable; no substitute counts.')
    return counts.astype(np.int64)
