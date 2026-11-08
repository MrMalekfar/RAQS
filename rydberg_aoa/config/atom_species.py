"""ATOM-001: explicit isotope selection; lazy ARC construction and SI data.

Parsing/capability discovery never imports ARC. No unknown-species fallback.
The legacy declaration means pure Rb85 (fraction=1), preserving old density.
"""
from dataclasses import dataclass, asdict
from importlib import import_module

from .laboratory import _real, _integer
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError, PhysicsError

_CLASSES = {'Rb85': 'Rubidium85', 'Rb87': 'Rubidium87', 'Cs133': 'Caesium'}
_ATOMS = {}


def _arc_version():
    from importlib.metadata import version
    try:
        return version('ARC-Alkali-Rydberg-Calculator')
    except Exception as exc:
        raise PhysicsCalculationError('ATOM-001: installed ARC version metadata unavailable; no invented version.') from exc


def _arc_real(value, name, **kwargs):
    try:
        return _real(value, name, **kwargs)
    except PhysicsConfigurationError as exc:
        raise PhysicsCalculationError(f'ATOM-001: invalid {name} from ARC: {value!r}.') from exc


@dataclass(frozen=True)
class AtomSpec:
    species: str
    isotope_fraction: float

    def __post_init__(self):
        if not isinstance(self.species, str) or self.species not in _CLASSES:
            raise PhysicsConfigurationError('ATOM-001: species must be Rb85, Rb87 or Cs133; no substitute.')
        fraction = _real(self.isotope_fraction, 'isotope_fraction', positive=True)
        if fraction > 1:
            raise PhysicsConfigurationError('ATOM-001: isotope_fraction must be in (0,1].')
        if self.species == 'Cs133' and fraction != 1:
            raise PhysicsConfigurationError('ATOM-001: Cs133 is the supported Cs isotope; fraction must be 1.')
        object.__setattr__(self, 'isotope_fraction', fraction)


LEGACY_ATOM = AtomSpec('Rb85', 1.)


@dataclass(frozen=True)
class ResolvedAtom:
    """Spawn-safe isotope receipt; no ARC object/SQLite connection."""
    specification: AtomSpec
    mass_kg: float
    nuclear_spin: float
    arc_version: str

    def __post_init__(self):
        if not isinstance(self.specification, AtomSpec) or not isinstance(self.arc_version, str) or not self.arc_version:
            raise PhysicsConfigurationError('ATOM-001: invalid resolved atom receipt.')
        _real(self.mass_kg, 'resolved mass [kg]', positive=True)
        spin = _real(self.nuclear_spin, 'resolved nuclear spin', nonnegative=True)
        if not (2*spin).is_integer():
            raise PhysicsConfigurationError('ATOM-001: invalid resolved nuclear spin.')


def resolved_atom(atom, spec):
    return ResolvedAtom(spec, _arc_real(atom.mass, 'ARC mass [kg]', positive=True),
                        _arc_real(atom.I, 'ARC I', nonnegative=True),
                        _arc_version())


def species_capabilities():
    """JSON-compatible public choices for CLI/GUI; no scientific initialization."""
    return {'schema_version': 1, 'species': dict(_CLASSES),
            'isotope_fraction': {'min_exclusive': 0., 'max_inclusive': 1.},
            'species_constraints': {'Cs133': {'isotope_fraction': [1.]}},
            'mixture_dynamics_supported': False,
            'collision_model_supported_species': ['Rb85']}


def get_arc_atom(spec=LEGACY_ATOM):
    if not isinstance(spec, AtomSpec):
        raise PhysicsConfigurationError('ATOM-001: AtomSpec required.')
    if spec.species not in _ATOMS:
        try:
            cls = getattr(import_module('arc'), _CLASSES[spec.species])
            atom = cls()
            _arc_real(atom.mass, 'ARC mass [kg]', positive=True)
            spin = _arc_real(atom.I, 'ARC nuclear spin I', nonnegative=True)
            try:
                _integer(atom.groundStateN, 'ARC groundStateN', minimum=1)
            except PhysicsConfigurationError as exc:
                raise PhysicsCalculationError('ATOM-001: invalid ARC groundStateN.') from exc
            if not (2*spin).is_integer():
                raise PhysicsCalculationError('ATOM-001: invalid ARC nuclear spin.')
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError(f'ATOM-001: ARC construction failed for {spec.species}.') from exc
        _ATOMS[spec.species] = atom
    return _ATOMS[spec.species]


def arc_density(atom, spec, temperature_k):
    """Selected isotope density [m^-3] = ARC element density * declared fraction."""
    if not isinstance(spec, AtomSpec):
        raise PhysicsConfigurationError('ATOM-001: AtomSpec required for density.')
    temperature_k = _real(temperature_k, 'ARC density temperature [K]', positive=True)
    try:
        element = _arc_real(atom.getNumberDensity(temperature_k), 'ARC density [m^-3]', nonnegative=True)
        return _arc_real(element * spec.isotope_fraction, 'isotope density [m^-3]', nonnegative=True)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f'ATOM-001: ARC density failed for {spec.species} at T={temperature_k} K. F05: no density substitute.') from exc


def atom_report(atom, spec, explicit):
    return {**asdict(spec), 'selection_source': 'profile' if explicit else 'legacy_Rb85_contract',
            'arc_class': type(atom).__module__ + '.' + type(atom).__qualname__,
            'arc_version': _arc_version(),
            'arc_options': {'preferQuantumDefects': atom.preferQuantumDefects, 'cpp_numerov': atom.cpp_numerov},
            'mass_kg': _arc_real(atom.mass, 'ARC mass [kg]', positive=True),
            'nuclear_spin': _arc_real(atom.I, 'ARC I', nonnegative=True),
            'ground_state_n': _integer(atom.groundStateN, 'ARC groundStateN', minimum=1),
            'natural_abundance': _arc_real(atom.abundance, 'ARC natural abundance', nonnegative=True),
            'density_contract': 'ARC element vapour density times declared isotope_fraction',
            'mixed_isotope_dynamics_included': False}


def effective_atom_spec(profile):
    return LEGACY_ATOM if profile.atom is None else profile.atom


def require_profile_atom(profile, ac):
    spec = effective_atom_spec(profile)
    if ac.ATOM_SPEC != spec or type(ac.ATOM).__name__ != _CLASSES[spec.species]:
        raise PhysicsConfigurationError('ATOM-001: profile atom differs from selected runtime atom; no reinterpretation.')


def validate_profile_atom(profile, atom):
    """Validate HFS/projections against the actual isotope; no level selection."""
    spec = effective_atom_spec(profile)
    env = profile.environment
    if env is not None and env.rate_source == 'thermal_budget' and env.collision_enabled and spec.species != 'Rb85':
        raise PhysicsConfigurationError('ATOM-001: current collision model is unvalidated for this isotope; disable it or declare explicit total rates.')
    if profile.mode == 'experimental':
        profile.request.validate_for_atom(atom)
        return
    validate_declared_ladder(atom, profile.request.level_mappings(), profile.coupling.polarization)


def validate_declared_ladder(atom, levels, coupling_q):
    """Same isotope boundary for profile and programmatic prescribed callers."""
    if levels['ground']['n'] != atom.groundStateN or levels['ground']['l'] != 0 or levels['ground']['j'] != .5:
        raise PhysicsConfigurationError('ATOM-001: declared ground state is incompatible with selected ARC atom.')
    for role in ('ground', 'excited'):
        level = levels[role]
        j, f, mf = level['j'], level['f'], level['mf']
        if not abs(atom.I-j) <= f <= atom.I+j or not (atom.I+j-f).is_integer() or abs(mf) > f or not (f-mf).is_integer():
            raise PhysicsConfigurationError(f'ATOM-001: {role} F/mF incompatible with selected isotope I/j.')
    mi = levels['excited']['mf'] + coupling_q - levels['rydberg1']['mj']
    if abs(mi) > atom.I or not (atom.I-mi).is_integer():
        raise PhysicsConfigurationError('ATOM-001: coupling channel has no allowed nuclear projection mI.')
