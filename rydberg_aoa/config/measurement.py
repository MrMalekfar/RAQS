"""Explicit receiver mode and RF signal switch; channel declarations stay valid."""
from dataclasses import dataclass
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class MeasurementOptions:
    mode: str
    signal_enabled: bool

    def __post_init__(self):
        if not isinstance(self.mode,str) or self.mode not in ('aoa','if_only') or type(self.signal_enabled) is not bool:
            raise PhysicsConfigurationError('NOISE-013: measurement mode aoa/if_only and boolean signal_enabled required.')
        if self.mode == 'aoa' and not self.signal_enabled:
            raise PhysicsConfigurationError('NOISE-013: a disabled signal cannot define an AoA; choose if_only.')


def resolved_measurement(options):
    if options is None:
        return MeasurementOptions('aoa',True)
    if not isinstance(options,MeasurementOptions):
        raise PhysicsConfigurationError('NOISE-013: parsed MeasurementOptions required.')
    return options


def validate_measurement_profile(profile):
    options=resolved_measurement(profile.measurement)
    if profile.run_config is None:
        if profile.measurement is not None:
            raise PhysicsConfigurationError('NOISE-013: receiver measurement requires a temporal execution profile.')
        return
    if profile.run_config.measurement_mode != options.mode:
        raise PhysicsConfigurationError('NOISE-013: receiver mode differs from execution-grid mode.')
    from .detection_study import DetectionStudy
    from .operating_study import OperatingStudy
    if options.mode == 'if_only' and profile.study is not None and not isinstance(profile.study,(DetectionStudy,OperatingStudy)):
        raise PhysicsConfigurationError('NOISE-013: current AoA-based studies cannot consume IF-only records; use an explicit trial campaign.')
    if profile.signal.electric_field_vm <= 0:
        # This is a source setpoint used to validate/calibrate the declared
        # channel, NOT the effective field when the explicit switch is OFF.
        raise PhysicsConfigurationError('NOISE-013: positive configured signal setpoint required; switch it off with signal_enabled=false, no substitute setpoint.')
    validate_signal_noise(options,profile.noise)


def validate_signal_noise(options,noise):
    if not options.signal_enabled and noise is not None:
        invalid={'rf_signal_amplitude','rf_relative_phase'} & set(noise.active_sources)
        if invalid:
            raise PhysicsConfigurationError(f'NOISE-013: noise on disabled signal is inapplicable: {sorted(invalid)}; disable those sources explicitly.')
