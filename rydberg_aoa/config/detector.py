"""Explicit PIN readout declarations; no device defaults or ARC imports."""

from dataclasses import dataclass
import numpy as np

from .laboratory import _real
from ..exceptions import PhysicsConfigurationError


def flag(value):
    if type(value) is not bool:
        raise PhysicsConfigurationError('NOISE-004: enabled must be boolean.')


@dataclass(frozen=True)
class ShotNoise:
    enabled: bool

    def __post_init__(self):
        flag(self.enabled)


@dataclass(frozen=True)
class ElectronicCurrentNoise:
    enabled: bool
    asd_a_per_sqrt_hz: float

    def __post_init__(self):
        flag(self.enabled)
        object.__setattr__(self, 'asd_a_per_sqrt_hz', _real(
            self.asd_a_per_sqrt_hz, 'NOISE-004 electronic current ASD', nonnegative=True))


@dataclass(frozen=True)
class JohnsonCurrentNoise:
    enabled: bool
    temperature_k: float
    shunt_resistance_ohm: float

    def __post_init__(self):
        flag(self.enabled)
        for name, positive in [('temperature_k', False), ('shunt_resistance_ohm', True)]:
            object.__setattr__(self, name, _real(getattr(self, name), 'NOISE-004 '+name,
                positive=positive, nonnegative=not positive))


@dataclass(frozen=True)
class PINDetector:
    enabled: bool
    collection_efficiency: float
    responsivity_a_per_w: float
    dark_current_a: float
    integration_time_us: float
    transimpedance_v_per_a: float
    output_units: str
    aperture_model: str

    def __post_init__(self):
        flag(self.enabled)
        for name in ('collection_efficiency', 'responsivity_a_per_w', 'integration_time_us',
                     'transimpedance_v_per_a', 'dark_current_a'):
            object.__setattr__(self, name, _real(getattr(self, name), 'NOISE-004 '+name,
                positive=name != 'dark_current_a', nonnegative=name == 'dark_current_a'))
        if self.collection_efficiency > 1:
            raise PhysicsConfigurationError('NOISE-004: collection efficiency must be in (0,1].')
        if (not isinstance(self.output_units, str) or self.output_units not in ('A', 'V') or
                not isinstance(self.aperture_model, str) or self.aperture_model != 'boxcar_sample_hold'):
            raise PhysicsConfigurationError('NOISE-004: output_units A/V and boxcar_sample_hold aperture required.')
        if self.integration_time_us*1e-6 == 0:
            raise PhysicsConfigurationError('NOISE-004: integration time underflows SI seconds.')


PHYSICAL_SOURCE_TYPES = {'photon_shot': ShotNoise, 'dark_shot': ShotNoise,
                         'electronic_current': ElectronicCurrentNoise, 'johnson_current': JohnsonCurrentNoise}


def validate_detector_sampling(config, run_config):
    """Independent nonoverlapping counting bins; no implicit bandwidth choice."""
    if config is None or not config.physical_readout_enabled:
        return
    if run_config is None:
        raise PhysicsConfigurationError('NOISE-004: explicit temporal sampling required.')
    dt = run_config.tlist_us[1]-run_config.tlist_us[0]
    if not np.isclose(config.detector.integration_time_us, dt, rtol=1e-7, atol=0):
        raise PhysicsConfigurationError('NOISE-004: boxcar integration_time_us must equal sample step; no implicit resampling.')
