"""LAB-005 explicit, picklable single-channel sensing inputs.

No level selection or atomic runtime publication occurs in this module.
The child adapter supplies the legacy worker's transient-cut global locally;
all model parameters and signed IF arrive in the task, including transit.
"""

from dataclasses import dataclass
import math
import numpy as np

from ..config.laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError


def _validate_sensing_grid(self):
    if self.temporal_ode is not None:
        from ..config.run_numerics import TemporalODE
        if not isinstance(self.temporal_ode,TemporalODE):
            raise PhysicsConfigurationError('NUM-ODE-002: typed temporal ODE controls required.')
    if self.temporal_doppler is not None:
        from ..config.temporal_doppler import ResolvedTemporalDoppler
        if not isinstance(self.temporal_doppler, ResolvedTemporalDoppler):
            raise PhysicsConfigurationError('DOP-TD-001: resolved typed temporal ensemble required.')
    if not isinstance(self.measurement_mode,str) or self.measurement_mode not in ('aoa','if_only'):
        raise PhysicsConfigurationError('NOISE-013: valid explicit execution measurement mode required.')
    for name in ('tlist_us', 'positions_m'):
        if not isinstance(getattr(self, name), tuple):
            raise PhysicsConfigurationError(f'LAB-005: {name} must be an immutable tuple.')
    times = tuple(_real(t, 'LAB-005 time [us]', nonnegative=True) for t in self.tlist_us)
    steps = np.diff(times)
    if (len(times) < 10 or times[0] != 0 or not np.all(np.isfinite(steps)) or
            np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-7, atol=0)):
        raise PhysicsConfigurationError('LAB-005: require >=10 uniform increasing output times starting at zero [us].')
    positions = []
    for row in self.positions_m:
        if not isinstance(row, tuple) or len(row) not in (2, 3):
            raise PhysicsConfigurationError('LAB-005: each position must be a real (x,y) or (x,y,z) tuple [m].')
        values = tuple(_real(v, 'LAB-005 position [m]') for v in row)
        positions.append(values if len(values) == 3 else (*values, 0.))
    if self.measurement_mode == 'if_only' and (not positions or len(set(positions)) != len(positions)):
        raise PhysicsConfigurationError('NOISE-013: IF-only requires >=1 distinct declared path.')
    if self.measurement_mode == 'aoa' and (len(positions) < 2 or len(set(positions)) != len(positions) or len(set(p[:2] for p in positions)) < 2):
        raise PhysicsConfigurationError('LAB-005: azimuth sensing requires >=2 distinct paths with an xy baseline.')
    angle = _real(self.target_azimuth_deg, 'LAB-005 azimuth [deg]')
    if not -90 <= angle <= 90:
        raise PhysicsConfigurationError('LAB-005: azimuth must lie in the visible [-90,90] degree interval.')
    if not isinstance(self.doa_method, str) or not self.doa_method.strip():
        raise PhysicsConfigurationError('LAB-005: declare a nonempty DoA method name.')
    if self.doa_options is not None:
        from ..config.doa_options import ProfileDOAOptions
        if not isinstance(self.doa_options, ProfileDOAOptions):
            raise PhysicsConfigurationError('DOA-PROFILE-001: resolved ProfileDOAOptions required.')
        self.doa_options.validate(self.doa_method.strip().lower())
    for name, value in (('tlist_us', times), ('positions_m', tuple(positions)),
                        ('target_azimuth_deg', angle),
                        ('transient_delay_us', _real(self.transient_delay_us, 'LAB-005 delay [us]', nonnegative=True)),
                        ('sensor_probe_offset_mhz', _real(self.sensor_probe_offset_mhz, 'LAB-005 sensing offset [MHz]')),
                        ('processes', _integer(self.processes, 'LAB-005 process count', minimum=1)),
                        ('cycles_per_snapshot', _integer(self.cycles_per_snapshot, 'LAB-005 snapshot cycles', minimum=1))):
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LaboratoryRunConfig:
    tlist_us: tuple
    positions_m: tuple
    target_azimuth_deg: float
    transient_delay_us: float
    dephasing_rates_per_us: tuple
    transit_rate_per_us: float
    sensor_probe_offset_mhz: float = 0.0
    processes: int = 1
    doa_method: str = 'phase_beamforming'
    cycles_per_snapshot: int = 2
    doa_options: object = None
    measurement_mode: str = 'aoa'
    temporal_doppler: object = None
    temporal_ode: object = None
    radiative: object = None

    def __post_init__(self):
        _validate_sensing_grid(self)
        if self.radiative is not None:
            from ..config.profile_radiative import ResolvedProfileRadiative
            if not isinstance(self.radiative, ResolvedProfileRadiative):
                raise PhysicsConfigurationError('RELAX-002: resolved radiative execution receipt required.')
        if not isinstance(self.dephasing_rates_per_us, tuple) or len(self.dephasing_rates_per_us) != 3:
            raise PhysicsConfigurationError('LAB-005: declare exactly three pure-dephasing rates as a tuple.')
        object.__setattr__(self, 'dephasing_rates_per_us', tuple(
            _real(v, 'LAB-005 pure dephasing [1/us]', nonnegative=True) for v in self.dephasing_rates_per_us))
        object.__setattr__(self, 'transit_rate_per_us',
                           _real(self.transit_rate_per_us, 'LAB-005 transit [1/us]', nonnegative=True))

    def validate_sampling(self, signed_if_mhz):
        frequency = _real(signed_if_mhz, 'LAB-005 signed IF [MHz]')
        dt = self.tlist_us[1] - self.tlist_us[0]
        if frequency == 0 or abs(frequency) >= .5 / dt:
            raise PhysicsConfigurationError('LAB-005: beat-note IF must be nonzero and strictly inside Nyquist; no DC substitute.')
        selected = np.count_nonzero(np.asarray(self.tlist_us) >= self.transient_delay_us)
        if self.measurement_mode == 'if_only':
            if selected < 10:
                raise PhysicsConfigurationError('NOISE-013: >=10 retained IF samples required; no earlier-sample substitute.')
            return
        window = self.cycles_per_snapshot * (1 / abs(frequency)) / dt
        if not math.isfinite(window) or window < 8 or selected < 10 or selected < math.floor(window + .5):
            raise PhysicsConfigurationError('LAB-005: transient cut leaves insufficient data for IF fit/full snapshot window; no earlier-sample substitute.')


@dataclass(frozen=True)
class ThermalRunPlan:
    """Validated sampling/geometry; physical rates remain unresolved until ARC initialization."""
    tlist_us: tuple
    positions_m: tuple
    target_azimuth_deg: float
    transient_delay_us: float
    sensor_probe_offset_mhz: float
    processes: int
    doa_method: str
    cycles_per_snapshot: int
    doa_options: object = None
    measurement_mode: str = 'aoa'
    temporal_doppler: object = None
    temporal_ode: object = None

    def __post_init__(self):
        _validate_sensing_grid(self)

    validate_sampling = LaboratoryRunConfig.validate_sampling

    def resolve(self, dephasing, transit):
        from dataclasses import fields
        return LaboratoryRunConfig(**{f.name: getattr(self, f.name) for f in fields(self)}, dephasing_rates_per_us=tuple(dephasing),
                                   transit_rate_per_us=transit)


@dataclass(frozen=True)
class LaboratoryWorkerTask:
    element_index: int
    omega_signal_complex: complex
    tlist_us: tuple
    signed_if_mhz: float
    omegas_rad_per_us: tuple
    detunings_rad_per_us: tuple
    decay_rates_per_us: object  # legacy tuple or immutable prepared radiative graph
    dephasing_rates_per_us: tuple
    transit_rate_per_us: float
    rf_orientation: object
    transient_delay_us: float
    rf_drive: object = None
    solver_diagnostic: object = None
    temporal_doppler: object = None
    atom: object = None
    temporal_ode: object = None
    measurement_mode: str = 'aoa'

    def __post_init__(self):
        if not isinstance(self.measurement_mode, str) or self.measurement_mode not in ('aoa', 'if_only'):
            raise PhysicsConfigurationError('DOP-H0-001: worker requires explicit aoa/if_only measurement mode.')


def run_laboratory_worker(task):
    """Spawn-safe adapter; restore the one legacy global even on failure.

    Pool workers execute one task at a time. This adapter is not a threaded
    override API; no atomic initialization or parent globals are required.
    Optional RF realization contains frozen numeric tuples, never Qobjs/RNGs.
    """
    if task.atom is not None:
        from ..config.atom_species import ResolvedAtom
        if not isinstance(task.atom, ResolvedAtom):
            raise PhysicsConfigurationError('ATOM-001: invalid worker atom receipt.')
        if task.temporal_doppler is not None and task.temporal_doppler.atom_mass_kg != task.atom.mass_kg:
            raise PhysicsConfigurationError('ATOM-001: temporal Doppler mass differs from worker atom.')
    from ..config import parameters as params
    if task.temporal_ode is not None:
        from ..config.run_numerics import TemporalODE
        if not isinstance(task.temporal_ode,TemporalODE):
            raise PhysicsConfigurationError('NUM-ODE-002: invalid worker ODE declaration.')
    ode_options = {'temporal_ode': task.temporal_ode} if task.temporal_ode is not None else {}
    from .time_domain import run_single_heterodyne_worker
    previous = params.ATOMIC_TRANSIENT_DELAY_US
    try:
        params.ATOMIC_TRANSIENT_DELAY_US = task.transient_delay_us
        def invoke(times):
            if task.temporal_doppler is not None:
                from .temporal_doppler import run_temporal_ensemble
                def solve(dp_shift, dc_shift):
                    detunings = (task.detunings_rad_per_us[0]+dp_shift,
                                 task.detunings_rad_per_us[1]+dc_shift,
                                 task.detunings_rad_per_us[2])
                    return run_single_heterodyne_worker(
                        task.element_index, task.omega_signal_complex, times,
                        task.signed_if_mhz, *task.omegas_rad_per_us, *detunings,
                        task.decay_rates_per_us, task.dephasing_rates_per_us,
                        noise_model=None, transit_rate_per_us=task.transit_rate_per_us,
                        rf_orientation=task.rf_orientation, _coherence_only=True,
                        **ode_options,
                        **({'rf_drive': task.rf_drive} if task.rf_drive is not None else {}))
                return run_temporal_ensemble(task, times, solve)
            return run_single_heterodyne_worker(
                task.element_index, task.omega_signal_complex, times,
                task.signed_if_mhz, *task.omegas_rad_per_us, *task.detunings_rad_per_us,
                task.decay_rates_per_us, task.dephasing_rates_per_us,
                noise_model=None, transit_rate_per_us=task.transit_rate_per_us,
                rf_orientation=task.rf_orientation,
                **ode_options,
                **({'rf_drive': task.rf_drive} if task.rf_drive is not None else {}))
        if task.solver_diagnostic is not None:
            from .temporal_diagnostic import run_diagnostic_worker
            return run_diagnostic_worker(task,invoke)
        return invoke(task.tlist_us)
    finally:
        params.ATOMIC_TRANSIENT_DELAY_US = previous
