"""Immutable declarations for RF/laser drive, readout noise and explicit trials."""

from dataclasses import dataclass, fields
from .detector import PINDetector, PHYSICAL_SOURCE_TYPES
from .rf_noise import RFOrnsteinUhlenbeck, RFDriveSettings, RF_SOURCE_UNITS
from .laser_noise import LaserFrequencyNoise, LASER_SOURCE_UNITS, LaserIntensityNoise, LASER_INTENSITY_UNITS

DRIVE_SOURCE_UNITS = {**RF_SOURCE_UNITS, **LASER_SOURCE_UNITS, **LASER_INTENSITY_UNITS}

from .laboratory import _integer, _real
from ..exceptions import PhysicsConfigurationError


def _flag(value, name):
    if type(value) is not bool:
        raise PhysicsConfigurationError(f'NOISE-001: {name} must be boolean.')
    return value


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PhysicsConfigurationError(f'NOISE-001: {name} requires exactly {sorted(keys)}.')
    return value


@dataclass(frozen=True)
class GaussianReadout:
    """Effective additive readout sigma in units of Im<sigma21>, not A/V.

    A disabled source retains validated parameters for controlled on/off runs.
    This declaration does not generate samples or apply noise to the trace.
    """
    enabled: bool
    sigma_observable: float

    def __post_init__(self):
        _flag(self.enabled, 'readout_gaussian.enabled')
        object.__setattr__(self, 'sigma_observable', _real(
            self.sigma_observable, 'NOISE-001 sigma_observable', nonnegative=True))


@dataclass(frozen=True)
class NoiseSource:
    name: str
    specification: object

    def __post_init__(self):
        kinds = {'readout_gaussian': GaussianReadout, **PHYSICAL_SOURCE_TYPES,
                 **{name: RFOrnsteinUhlenbeck for name in RF_SOURCE_UNITS},
                 **{name: LaserFrequencyNoise for name in LASER_SOURCE_UNITS},
                 **{name: LaserIntensityNoise for name in LASER_INTENSITY_UNITS}}
        if (not isinstance(self.name, str) or self.name not in kinds or
                not isinstance(self.specification, kinds[self.name])):
            raise PhysicsConfigurationError('NOISE-001: unsupported noise source/specification.')
        if self.name in RF_SOURCE_UNITS and self.specification.units != RF_SOURCE_UNITS[self.name]:
            raise PhysicsConfigurationError('NOISE-005: RF source units do not match its physical role.')


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool
    seed: int
    trials: int
    sources: tuple
    schema_version: int = 1
    detector: object = None
    drive: object = None
    readout: object = None

    def __post_init__(self):
        _flag(self.enabled, 'enabled')
        if type(self.schema_version) is not int or self.schema_version not in (1, 2, 3):
            raise PhysicsConfigurationError('NOISE-005: noise schema_version must be integer 1/2/3.')
        object.__setattr__(self, 'seed', _integer(self.seed, 'NOISE-001 seed'))
        object.__setattr__(self, 'trials', _integer(self.trials, 'NOISE-001 trials', minimum=1))
        if (not isinstance(self.sources, tuple) or not self.sources or
                any(not isinstance(source, NoiseSource) for source in self.sources)):
            raise PhysicsConfigurationError('NOISE-001: nonempty immutable noise sources required.')
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise PhysicsConfigurationError('NOISE-001: duplicate noise source.')
        if self.schema_version == 1 and (self.detector is not None or set(names) != {'readout_gaussian'}):
            raise PhysicsConfigurationError('NOISE-004: schema 1 is effective Gaussian readout only.')
        if self.schema_version == 2 and (not isinstance(self.detector, PINDetector) or
                                        not set(names).issubset(PHYSICAL_SOURCE_TYPES)):
            raise PhysicsConfigurationError('NOISE-004: schema 2 requires explicit PIN detector and physical current sources.')
        if self.schema_version < 3 and (self.drive is not None or self.readout is not None):
            raise PhysicsConfigurationError('NOISE-005: drive/readout nesting requires schema 3.')
        if self.schema_version == 3:
            if not isinstance(self.drive, RFDriveSettings) or not set(names).issubset(DRIVE_SOURCE_UNITS):
                raise PhysicsConfigurationError('NOISE-005/006: explicit drive settings and RF/laser OU sources required.')
            if self.readout is not None and (not isinstance(self.readout, NoiseConfig) or
                    self.readout.schema_version not in (1, 2) or self.readout.seed != self.seed or self.readout.trials != self.trials):
                raise PhysicsConfigurationError('NOISE-005: nested readout must be schema 1/2 with identical seed and trials.')
            object.__setattr__(self, 'detector', self.readout.detector if self.readout is not None else None)

    @property
    def drive_selected(self):
        return self.schema_version == 3 and self.enabled and any(s.specification.enabled for s in self.sources)

    @property
    def stochastic_drive(self):
        return self.drive_selected and any(s.specification.enabled and s.specification.stationary_std > 0 for s in self.sources)

    @property
    def effective_readout(self):
        return self.readout if self.schema_version == 3 and self.enabled else None

    @property
    def physical_readout_enabled(self):
        if self.schema_version == 3:
            return self.effective_readout is not None and self.effective_readout.physical_readout_enabled
        return self.schema_version == 2 and self.enabled and self.detector.enabled

    @property
    def readout_enabled(self):
        if self.schema_version == 3:
            return self.effective_readout is not None and self.effective_readout.readout_enabled
        return self.physical_readout_enabled or bool(self.active_sources)

    @property
    def active_sources(self):
        if self.schema_version == 3:
            own = tuple(s.name for s in self.sources if self.enabled and s.specification.enabled)
            return own + (self.effective_readout.active_sources if self.effective_readout is not None else ())
        return tuple(source.name for source in self.sources
                     if self.enabled and source.specification.enabled and
                     (self.schema_version == 1 or self.detector.enabled))


def parse_noise_config(document):
    """Strict versioned block; unknown sources never become ignored options."""
    version = document.get('schema_version') if isinstance(document, dict) else None
    if type(version) is int and version == 4:
        return _parse_unified_noise(document)
    keys = ('schema_version', 'enabled', 'seed', 'trials', 'sources')
    extra = (('detector',) if type(version) is int and version == 2 else
             ('drive', 'readout') if type(version) is int and version == 3 else ())
    data = _object(document, keys + extra, 'noise')
    raw = data['sources']
    if not isinstance(raw, dict) or not raw:
        raise PhysicsConfigurationError('NOISE-001: sources must be a nonempty object.')
    if type(version) is int and version == 3:
        settings = RFDriveSettings(**_object(data['drive'], (f.name for f in fields(RFDriveSettings)), 'drive'))
        sources = []
        if any(not isinstance(name, str) for name in raw) or not set(raw).issubset(DRIVE_SOURCE_UNITS):
            raise PhysicsConfigurationError('NOISE-005/006: unsupported drive source.')
        for name in sorted(raw):
            kind = (LaserFrequencyNoise if name in LASER_SOURCE_UNITS else
                    LaserIntensityNoise if name in LASER_INTENSITY_UNITS else RFOrnsteinUhlenbeck)
            spec = kind(**_object(raw[name], (f.name for f in fields(kind)), name))
            sources.append(NoiseSource(name, spec))
        nested = data['readout']
        if nested is not None and (not isinstance(nested, dict) or type(nested.get('schema_version')) is not int or nested['schema_version'] not in (1, 2)):
            raise PhysicsConfigurationError('NOISE-005: readout is null or a complete schema 1/2 declaration.')
        return NoiseConfig(data['enabled'], data['seed'], data['trials'], tuple(sources), 3,
            drive=settings, readout=parse_noise_config(nested) if nested is not None else None)
    if type(version) is int and version == 2:
        if any(not isinstance(name, str) for name in raw):
            raise PhysicsConfigurationError('NOISE-004: noise source names must be strings.')
        detector = PINDetector(**_object(data['detector'], (f.name for f in fields(PINDetector)), 'detector'))
        sources = []
        for name in sorted(raw):
            if name not in PHYSICAL_SOURCE_TYPES:
                raise PhysicsConfigurationError(f'NOISE-004: unsupported physical source {name!r}.')
            kind = PHYSICAL_SOURCE_TYPES[name]
            sources.append(NoiseSource(name, kind(**_object(raw[name], (f.name for f in fields(kind)), name))))
        return NoiseConfig(data['enabled'], data['seed'], data['trials'], tuple(sources), version, detector)
    if set(raw) != {'readout_gaussian'}:
        raise PhysicsConfigurationError('NOISE-001: only readout_gaussian is declared in this version; unknown sources are rejected.')
    spec = _object(raw['readout_gaussian'], ('enabled', 'sigma_observable'), 'readout_gaussian')
    return NoiseConfig(data['enabled'], data['seed'], data['trials'],
                       (NoiseSource('readout_gaussian', GaussianReadout(**spec)),), data['schema_version'])


def _parse_unified_noise(document):
    """Map the wire format onto the existing, tested drive/readout models."""
    data = _object(document, ('schema_version', 'enabled', 'seed', 'trials', 'drive', 'readout'), 'noise')
    _flag(data['enabled'], 'enabled')
    campaign = {name: data[name] for name in ('seed', 'trials')}
    drive, readout = data['drive'], data['readout']
    if drive is None and readout is None:
        raise PhysicsConfigurationError('NOISE-001: declare a drive or readout model, or omit noise.')
    if readout is not None:
        model = readout.get('model') if isinstance(readout, dict) else None
        if model not in ('gaussian', 'pin'):
            raise PhysicsConfigurationError('NOISE-001: readout model must be gaussian or pin.')
        keys = ('model', 'sources') + (('detector',) if model == 'pin' else ())
        if drive is not None:
            keys += ('enabled',)
        readout = dict(_object(readout, keys, 'noise.readout'))
        readout.pop('model')
        if drive is None:
            readout['enabled'] = data['enabled']
        readout.update(campaign, schema_version=1 if model == 'gaussian' else 2)
    if drive is None:
        return parse_noise_config(readout)
    drive = _object(drive, ('settings', 'sources'), 'noise.drive')
    return parse_noise_config({**campaign, 'schema_version': 3, 'enabled': data['enabled'],
                               'drive': drive['settings'], 'sources': drive['sources'], 'readout': readout})


def require_noise_execution_supported(config):
    """Validate the supported readout declaration before any atomic writes."""
    if config is None:
        return
    if not isinstance(config, NoiseConfig):
        raise PhysicsConfigurationError('NOISE-001: a validated NoiseConfig is required.')
