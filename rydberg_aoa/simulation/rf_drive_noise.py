"""Precomputed RF envelopes and residual laser frequency; no RNG in solver RHS."""

from bisect import bisect_right
from dataclasses import dataclass
import math

import numpy as np

from ..config.laboratory import _real, _integer
from ..config.noise import require_noise_execution_supported
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError
from .noise_random import NoiseRandomPlan


@dataclass(frozen=True)
class RFDriveRealization:
    times_us: tuple
    lo_amplitude: tuple
    signal_amplitude: tuple
    relative_phase_rad: tuple
    probe_frequency_hz: tuple = None
    coupling_frequency_hz: tuple = None
    probe_intensity: tuple = None
    coupling_intensity: tuple = None

    def __post_init__(self):
        if not isinstance(self.times_us, tuple):
            raise PhysicsConfigurationError('NOISE-006: immutable time grid required.')
        for name in ('probe_frequency_hz', 'coupling_frequency_hz'):
            if getattr(self, name) is None:
                object.__setattr__(self, name, (0.,)*len(self.times_us))
        for name in ('probe_intensity', 'coupling_intensity'):
            if getattr(self, name) is None:
                object.__setattr__(self, name, (1.,)*len(self.times_us))
        arrays = []
        for name in ('times_us', 'lo_amplitude', 'signal_amplitude', 'relative_phase_rad',
                     'probe_frequency_hz', 'coupling_frequency_hz', 'probe_intensity', 'coupling_intensity'):
            raw = getattr(self, name)
            if not isinstance(raw, tuple) or any(isinstance(v, (bool, np.bool_)) or
                    not isinstance(v, (int, float, np.integer, np.floating)) for v in raw):
                raise PhysicsConfigurationError('NOISE-005: RF samples must be real numeric scalars; no coercion of bool/arrays.')
            a = np.asarray(raw)
            if (not isinstance(raw, tuple) or a.ndim != 1 or a.dtype.kind not in 'iuf' or
                    len(a) < 2 or not np.isfinite(a).all()):
                raise PhysicsConfigurationError('NOISE-005: immutable finite 1-D RF realization required.')
            arrays.append(a)
        if (len({len(a) for a in arrays}) != 1 or arrays[0][0] != 0 or
                not np.all(np.diff(arrays[0]) > 0) or np.any(arrays[1] <= 0) or np.any(arrays[2] <= 0)):
            raise PhysicsConfigurationError('NOISE-005: invalid RF grid or positive envelope; no repair.')
        if any(np.any(a <= 0) for a in arrays[-2:]):
            raise PhysicsConfigurationError('NOISE-007: intensity factors must be positive; no clipping.')

    def at(self, t):
        """Linear interpolation of positive envelopes and unwrapped phase."""
        return self._interpolate(t, (self.lo_amplitude, self.signal_amplitude, self.relative_phase_rad))

    def _interpolate(self, t, arrays):
        t = _real(t, 'NOISE-005 coefficient time')
        if t < self.times_us[0] or t > self.times_us[-1]:
            raise PhysicsCalculationError(f'NOISE-005: coefficient time {t:g}us outside declared RF horizon; no clamp/extrapolation.')
        index = min(bisect_right(self.times_us, t)-1, len(self.times_us)-2)
        weight = (t-self.times_us[index])/(self.times_us[index+1]-self.times_us[index])
        return tuple((1-weight)*a[index]+weight*a[index+1]
                     for a in arrays)

    def laser_at(self, t):
        """Residual instantaneous laser frequency offsets [Hz], not FWHM."""
        return self._interpolate(t, (self.probe_frequency_hz, self.coupling_frequency_hz))

    def laser_intensity_at(self, t):
        """Positive shared intensity factors, linearly interpolated before sqrt."""
        return self._interpolate(t, (self.probe_intensity, self.coupling_intensity))

    def laser_phase_at(self, t):
        """Exact integral of the linearly interpolated frequency; phase(0)=0.

        Diagnostic only; H uses frequency directly in the laser-following frame.
        No finite difference of phase, no second independent phase trajectory.
        """
        self.laser_at(t)  # same horizon/type validation
        t = float(t)
        index = min(bisect_right(self.times_us, t)-1, len(self.times_us)-2)
        dt = t-self.times_us[index]
        width = self.times_us[index+1]-self.times_us[index]
        phases = []
        for a in (self.probe_frequency_hz, self.coupling_frequency_hz):
            area = sum((a[k]+a[k+1])*.5*(self.times_us[k+1]-self.times_us[k]) for k in range(index))
            area += a[index]*dt + (a[index+1]-a[index])*dt*dt/(2*width)
            phase = (area/1e6)*(2*np.pi)
            if not np.isfinite(phase):
                raise PhysicsCalculationError('NOISE-006: integrated phase is nonfinite; no substitute.')
            phases.append(phase)
        return tuple(phases)

    def laser_phase_nodes(self):
        """Diagnostic phases on all nodes, shape (K,2), in linear time."""
        try:
            with np.errstate(over='raise', invalid='raise'):
                frequency = np.asarray((self.probe_frequency_hz, self.coupling_frequency_hz)).T
                increments = (frequency[:-1]+frequency[1:])*.5*np.diff(self.times_us)[:, None]
                area = np.vstack((np.zeros((1, 2)), np.cumsum(increments, axis=0)))
                phase = (area/1e6)*(2*np.pi)
            if not np.isfinite(phase).all():
                raise PhysicsCalculationError('NOISE-006: phase nodes are nonfinite; no substitute.')
            return phase
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError('NOISE-006: phase integration failed; no substitute.') from exc


class RFDriveEvaluator:
    """One-worker, last-exact-time interpolation cache for a frozen drive.

    Every query validates its real finite time and horizon before any hit.
    A new float bit pattern (including +0/-0) discards all three lazy groups;
    backward/repeated solver times are legal and never rounded. The bracket,
    weight and arithmetic order match RFDriveRealization._interpolate.
    The immutable realization remains the process-transport/public reference;
    this mutable evaluator is constructed only inside a worker, never shared.
    """
    __slots__ = ('_drive', '_last_time_key', '_index', '_weight',
                 '_rf', '_laser', '_intensity')

    def __init__(self, realization):
        if type(realization) is not RFDriveRealization:
            raise PhysicsConfigurationError(
                'NUM-WARM-PERF-002: evaluator requires a concrete frozen RFDriveRealization.')
        self._drive = realization
        self._last_time_key = None
        self._index = None
        self._weight = None
        self._rf = self._laser = self._intensity = None

    def _sample(self, t, arrays, group):
        t = _real(t, 'NOISE-005 coefficient time')
        times = self._drive.times_us
        if t < times[0] or t > times[-1]:
            raise PhysicsCalculationError(
                f'NOISE-005: coefficient time {t:g}us outside declared RF horizon; no clamp/extrapolation.')
        key = t.hex()
        if key != self._last_time_key:
            index = min(bisect_right(times, t)-1, len(times)-2)
            weight = (t-times[index])/(times[index+1]-times[index])
            self._index, self._weight = index, weight
            self._rf = self._laser = self._intensity = None
            self._last_time_key = key
        value = getattr(self, group)
        if value is None:
            index, weight = self._index, self._weight
            value = tuple((1-weight)*a[index]+weight*a[index+1] for a in arrays)
            setattr(self, group, value)
        return value

    def at(self, t):
        """RF (LO amplitude, signal amplitude, phase [rad]), canonical order."""
        r = self._drive
        return self._sample(t, (r.lo_amplitude, r.signal_amplitude, r.relative_phase_rad), '_rf')

    def laser_at(self, t):
        """Residual (probe, coupling) frequency [Hz]; no phase integration."""
        r = self._drive
        return self._sample(t, (r.probe_frequency_hz, r.coupling_frequency_hz), '_laser')

    def laser_intensity_at(self, t):
        """Positive (probe, coupling) intensity factors, before Rabi sqrt."""
        r = self._drive
        return self._sample(t, (r.probe_intensity, r.coupling_intensity), '_intensity')


def build_rf_realization(config, *, trial):
    """Exact stationary OU samples at nodes, explicit linear interpolation.

Amplitude node factors exp(x-sigma^2/2) have ensemble mean one, never clip or
renormalize a trajectory. Linear interpolation of factors preserves positivity
and ensemble mean one. Phase is bounded-variance colored envelope phase, not
Wiener diffusion/linewidth or white-noise frequency jitter. Laser samples are
stationary frequency offsets [Hz], whose integral is phase (not another source).
"""
    require_noise_execution_supported(config)
    trial = _integer(trial, 'NOISE-005 trial')
    if config is None or config.schema_version != 3 or trial >= config.trials:
        raise PhysicsConfigurationError('NOISE-005: validated schema 3 and trial index required.')
    if not config.stochastic_drive:
        return None
    try:
        settings = config.drive
        count = round(settings.horizon_us/settings.noise_step_us)+1
        times = np.linspace(0., settings.horizon_us, count)
        plan = NoiseRandomPlan(config)
        values = {'rf_lo_amplitude': np.ones(count), 'rf_signal_amplitude': np.ones(count),
                  'rf_relative_phase': np.zeros(count), 'laser_probe_frequency': np.zeros(count),
                  'laser_coupling_frequency': np.zeros(count), 'laser_probe_intensity': np.ones(count),
                  'laser_coupling_intensity': np.ones(count)}
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            for source in config.sources:
                spec = source.specification
                if not spec.enabled or spec.stationary_std == 0:
                    continue
                sigma = spec.stationary_std
                a = math.exp(-settings.noise_step_us/spec.correlation_time_us)
                innovation_std = sigma*math.sqrt(-math.expm1(-2*settings.noise_step_us/spec.correlation_time_us))
                draws = plan.generator(source.name, trial=trial, path=None).standard_normal(count)
                x = np.empty(count)
                x[0] = sigma*draws[0]
                for k in range(1, count):
                    x[k] = a*x[k-1]+innovation_std*draws[k]
                values[source.name] = np.exp(x-.5*sigma**2) if spec.units in ('log_amplitude', 'relative_rms') else x
                if spec.units == 'relative_rms' and np.all(values[source.name] == 1.):
                    raise PhysicsCalculationError('NOISE-007: positive intensity noise is unresolved in floating-point factors; no silent zero replacement.')
        return RFDriveRealization(tuple(float(v) for v in times),
            tuple(float(v) for v in values['rf_lo_amplitude']),
            tuple(float(v) for v in values['rf_signal_amplitude']),
            tuple(float(v) for v in values['rf_relative_phase']),
            tuple(float(v) for v in values['laser_probe_frequency']),
            tuple(float(v) for v in values['laser_coupling_frequency']),
            tuple(float(v) for v in values['laser_probe_intensity']),
            tuple(float(v) for v in values['laser_coupling_intensity']))
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-005: RF realization construction failed; no replacement trajectory.') from exc


def drive_signal_coefficient(t, nominal_coefficient, realization):
    _, amplitude, phase = realization.at(t)
    value = nominal_coefficient*amplitude*np.exp(1j*phase)
    if not np.isfinite(value):
        raise PhysicsCalculationError('NOISE-005: nonfinite RF coefficient; no substitute.')
    return value
