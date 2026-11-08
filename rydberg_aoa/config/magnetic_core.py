"""Explicit internal magnetic-core preflight and opt-in transient diagnostics."""
from dataclasses import dataclass
from .laboratory import _real
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class CorePeriodicControls:
    """Fixed Fourier cutoff and IF peak-transmission error budgets."""
    harmonic_order: int
    if_max_abs_error: float
    if_max_rel_error: float

    def __post_init__(self):
        from .laboratory import _integer
        object.__setattr__(self,'harmonic_order',_integer(self.harmonic_order,'BBR-017 harmonic order',minimum=2))
        for name in ('if_max_abs_error','if_max_rel_error'):
            object.__setattr__(self,name,_real(getattr(self,name),'BBR-017 '+name,nonnegative=True))
        if self.if_max_rel_error>=1:
            raise PhysicsConfigurationError('BBR-017: relative IF error budget must be below one.')


@dataclass(frozen=True)
class CoreTemporalDiagnostic:
    duration_us: float
    sample_count: int
    signal_phase_rad: float
    backend: str = "qutip"
    initial_state: str = "prepared"
    extract_if: bool = False
    transient_delay_us: float = 0.
    periodic: CorePeriodicControls | None = None

    def __post_init__(self):
        from .laboratory import _integer
        object.__setattr__(self,'duration_us',_real(self.duration_us,'BBR-014 duration [us]',positive=True))
        object.__setattr__(self,'sample_count',_integer(self.sample_count,'BBR-014 samples',minimum=2))
        object.__setattr__(self,'signal_phase_rad',_real(self.signal_phase_rad,'BBR-014 signal phase [rad]'))
        object.__setattr__(self,'transient_delay_us',_real(self.transient_delay_us,'BBR-015 delay [us]',nonnegative=True))
        if (type(self.backend) is not str or type(self.initial_state) is not str or
                self.backend not in ('qutip','charge_block_bdf','charge_block_periodic') or self.initial_state not in ('prepared','lo_steady')):
            raise PhysicsConfigurationError('BBR-015: unsupported backend or initial-state declaration.')
        if type(self.extract_if) is not bool:
            raise PhysicsConfigurationError('BBR-015: extract_if must be boolean.')
        if self.backend == 'qutip' and (self.initial_state != 'prepared' or self.extract_if or self.transient_delay_us):
            raise PhysicsConfigurationError('BBR-015: LO steady state/IF requires explicit charge_block_bdf backend.')
        if self.transient_delay_us >= self.duration_us:
            raise PhysicsConfigurationError('BBR-015: delay must precede final time.')
        if self.extract_if and self.initial_state != 'lo_steady':
            raise PhysicsConfigurationError('BBR-015: IF diagnostic requires certified LO steady initial state.')
        if self.backend=='charge_block_periodic':
            if not isinstance(self.periodic,CorePeriodicControls) or not self.extract_if or self.initial_state!='lo_steady':
                raise PhysicsConfigurationError('BBR-017: periodic IF requires explicit controls, extract_if and LO steady initial reference.')
        elif self.periodic is not None:
            raise PhysicsConfigurationError('BBR-017: periodic controls require charge_block_periodic; settings cannot be ignored.')




@dataclass(frozen=True)
class MagneticCoreOptions:
    model: str
    bbr_enabled: bool
    radiation_temperature_k: float
    temporal_diagnostic: CoreTemporalDiagnostic | None = None

    def __post_init__(self):
        if self.model != 'hfs_optical_rydberg_fs_internal_e1':
            raise PhysicsConfigurationError('BBR-003: unknown magnetic core model.')
        if type(self.bbr_enabled) is not bool:
            raise PhysicsConfigurationError('BBR-003: bbr_enabled must be boolean.')
        temperature = _real(self.radiation_temperature_k, 'BBR-003 radiation temperature [K]', nonnegative=True)
        object.__setattr__(self, 'radiation_temperature_k', temperature)
        if self.temporal_diagnostic is not None and not isinstance(self.temporal_diagnostic,CoreTemporalDiagnostic):
            raise PhysicsConfigurationError('BBR-014: typed core temporal diagnostic required.')

    @property
    def effective_temperature_k(self):
        return self.radiation_temperature_k if self.bbr_enabled else 0.


def validate_magnetic_core(profile):
    if profile.magnetic_core is None:
        return
    if not isinstance(profile.magnetic_core, MagneticCoreOptions):
        raise PhysicsConfigurationError('BBR-003: typed magnetic-core declaration required.')
    if profile.radiative is None or profile.radiative.model != 'selected_internal_spontaneous_e1':
        raise PhysicsConfigurationError('BBR-003: declare the prepared nuclear reference via radiative options.')
    if profile.study is not None:
        raise PhysicsConfigurationError('BBR-003: multilevel studies are not connected; no four-state study substitute.')
    if profile.magnetic_core.temporal_diagnostic is not None:
        if profile.probe.power_mw<=0:
            raise PhysicsConfigurationError('BBR-014: nonzero probe required for explicit polarization calibration.')
        if profile.numerics is None and profile.magnetic_core.temporal_diagnostic.backend!='charge_block_periodic':
            raise PhysicsConfigurationError('BBR-014: explicit numerics.temporal_ode required.')
        if profile.signal.polarization != profile.lo.polarization:
            raise PhysicsConfigurationError('BBR-014: signal and LO must declare the same q for this runtime.')
        spec=profile.magnetic_core.temporal_diagnostic
        block=spec.backend in ("charge_block_bdf","charge_block_periodic")
        env=profile.environment
        if env is not None and (env.linewidth_enabled or env.collision_enabled or (env.transit_enabled and not block)):
            raise PhysicsConfigurationError('BBR-014: internal-only temporal core does not implement linewidth/collision/transit.')
        config=profile.run_config
        if config is not None and (any(getattr(config,'dephasing_rates_per_us',())) or (getattr(config,'transit_rate_per_us',0.) and not block)):
            raise PhysicsConfigurationError('BBR-014: nonzero dephasing/transit cannot be ignored.')
        if block:
            from .run_numerics import BDFOrderControls
            if spec.backend=="charge_block_bdf" and isinstance(profile.numerics.temporal_ode,BDFOrderControls):
                raise PhysicsConfigurationError('BBR-015: charge-block SciPy BDF cannot enforce max_order.')
            if spec.initial_state=='lo_steady':
                thermal_transit=env is not None and env.rate_source=='thermal_budget' and env.transit_enabled
                explicit_transit=config is not None and getattr(config,'transit_rate_per_us',0.)>0
                if not (thermal_transit or explicit_transit):
                    raise PhysicsConfigurationError('BBR-015: LO stationary diagnostic requires positive prepared transit.')
        if profile.temporal_doppler is not None and profile.temporal_doppler.enabled:
            raise PhysicsConfigurationError('BBR-014: temporal Doppler is not connected to this diagnostic.')
        if profile.noise is not None and (profile.noise.readout_enabled or profile.noise.drive_selected or profile.noise.physical_readout_enabled):
            raise PhysicsConfigurationError('BBR-014: active noise/detector is not connected to this diagnostic.')


def require_four_state_profile(profile):
    from .lo_phase_reference import require_uniform_lo_execution
    require_uniform_lo_execution(profile)
    if getattr(profile, 'magnetic_core', None) is not None:
        raise PhysicsConfigurationError('BBR-003: magnetic core is construction/validation only; legacy four-state consumer refused.')
