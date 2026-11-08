"""LAB-004 local runtime construction and exact laboratory input/cache guards.

atomic_config owns transactional publication. This module never tunes or
reconciles applied fields and never calls the legacy level/dipole search.
"""

from dataclasses import asdict, dataclass
import copy
import math

from .laboratory import AppliedLaser
from .rf_configuration import read_configured_rf_fields
from ..arc._scan_validation import arc_real_scalar
from ..arc.dipole import EA0_TO_CM
from ..arc.decay_rates import calculate_decay_rates, resolve_rydberg_orientation
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError, PhysicsError
from ..level_finder.prepared_channel import PreparedLabSelection, select_prepared_laboratory_channel


def laboratory_identity(atom, request):
    """Full unrounded request plus actual atom object/isotope identity."""
    request.validate_for_atom(atom)
    import sys
    ac = sys.modules.get(__package__ + '.atomic_config')
    # Standalone ARC callers have no profile composition. Selected runtime
    # callers include the declared fraction in every cache/consumer identity.
    spec = ac.ATOM_SPEC if ac is not None and atom is ac.ATOM else None
    return ('LAB-004', request, type(atom).__module__, type(atom).__qualname__,
            id(atom), float(atom.I), int(atom.groundStateN), spec)


def require_matching_applied_fields(acfg, request):
    """Consumers still use global applied fields; reject disagreement."""
    lo, signal = read_configured_rf_fields()
    try:
        probe = AppliedLaser(**acfg.LASER_PARAMS['probe'])
        coupling = AppliedLaser(**acfg.LASER_PARAMS['coupling'])
    except (KeyError, TypeError) as exc:
        raise PhysicsConfigurationError('LAB-004: malformed configured laser fields; no substitution.') from exc
    for name, declared, current in (('probe', request.probe, probe), ('coupling', request.coupling, coupling),
                                    ('LO', request.lo, lo), ('signal', request.signal, signal)):
        if declared != current:
            raise PhysicsConfigurationError(f'LAB-004: explicit {name} input differs from configured applied field; '
                                            'configure inputs explicitly. No automatic synchronization.')


@dataclass(frozen=True)
class LaboratoryRuntime:
    identity: tuple
    selection: PreparedLabSelection
    rates: tuple
    detunings: tuple

    @property
    def request(self):
        return self.selection.request

    def levels(self):
        objects = self.selection.level_objects()
        for level in objects.values():
            level.label = f"{level.n}{('S', 'P', 'D', 'F')[level.l]}{level.j}"
        return {role: asdict(level) for role, level in objects.items()}

    def dipoles(self):
        levels, winner = self.selection.level_objects(), self.selection.winner
        output = {}
        for name, src, dst, signed in (('probe', 'ground', 'excited', self.selection.probe_ea0),
                                       ('coupling', 'excited', 'rydberg1', winner.coupling_ea0),
                                       ('rf', 'rydberg1', 'rydberg2', winner.rf_ea0)):
            output[name] = {'dipole_si': abs(signed) * EA0_TO_CM, 'dipole_ea0': signed,
                            'level1': levels[src], 'level2': levels[dst], 'transition': f'{src} -> {dst}'}
        return output

    def detuning_entry(self):
        return dict(zip(('delta_p_rad_us', 'delta_c_rad_us', 'delta_rf_rad_us'), self.detunings))


def build_laboratory_runtime(atom, request):
    """Calculate all quantities locally; no shared cache/global writes."""
    identity = laboratory_identity(atom, request)
    selection = select_prepared_laboratory_channel(atom, request)
    levels = selection.level_objects()
    rates = calculate_decay_rates(atom, levels['ground'], levels['excited'], levels['rydberg1'], levels['rydberg2'])
    try:
        orientation = resolve_rydberg_orientation(atom, levels['rydberg1'], levels['rydberg2'])
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('LAB-004: final ARC RF orientation check failed for '
                                      f"{levels['rydberg1']} -> {levels['rydberg2']}.") from exc
    if orientation != selection.rf_orientation:
        raise PhysicsCalculationError('LAB-004: RF orientation/frequency changed during decay calculation; no stale publication.')
    rf_detuning = arc_real_scalar(2 * math.pi * (request.lo.frequency_hz - selection.winner.manifold.carrier_hz) / 1e6,
                                  'LAB-004 RF detuning [rad/µs]')
    return LaboratoryRuntime(identity, selection, tuple(rates),
                             (selection.optical_selection.probe.detuning_rad_per_us,
                              selection.winner.manifold.coupling.detuning_rad_per_us, rf_detuning))


def populate_laboratory_caches(acfg, runtime):
    """Whole-entry writes preserve ARCH-001TX's shallow cache rollback."""
    key = runtime.identity
    acfg._ATOMIC_LEVELS_CACHE[key] = runtime.levels()
    acfg._DIPOLE_MATRIX_CACHE[key] = runtime.dipoles()
    acfg._DETUNING_CACHE[key] = runtime.detuning_entry()
    acfg._OPTIMIZATION_RESULTS_CACHE[key] = {'lab_runtime': runtime, 'dipole_rf': runtime.selection.winner.rf_ea0}


def require_current_laboratory_runtime(acfg):
    """Fail on stale inputs, missing/corrupt cache or changed derived views."""
    request = acfg._INITIALIZED_REQUEST
    require_matching_applied_fields(acfg, request)
    key = laboratory_identity(acfg.ATOM, request)
    if key != acfg._INITIALIZED_RESOLUTION_IDENTITY:
        raise PhysicsConfigurationError('LAB-004: atom/request identity changed; explicitly reinitialize before consuming physics.')
    caches = (acfg._ATOMIC_LEVELS_CACHE, acfg._DIPOLE_MATRIX_CACHE, acfg._DETUNING_CACHE, acfg._OPTIMIZATION_RESULTS_CACHE)
    if any(key not in cache for cache in caches):
        raise PhysicsCalculationError('LAB-004: published laboratory runtime has a missing cache; no self-heal or zero substitution.')
    metadata = acfg._OPTIMIZATION_RESULTS_CACHE[key]
    if not isinstance(metadata, dict):
        raise PhysicsCalculationError('LAB-004: malformed laboratory runtime metadata cache.')
    runtime = metadata.get('lab_runtime')
    if not isinstance(runtime, LaboratoryRuntime) or runtime.identity != key:
        raise PhysicsCalculationError('LAB-004: missing or inconsistent laboratory runtime metadata.')
    rabi = runtime.selection.rabi
    expected = (runtime.selection.winner.rf_ea0, *runtime.rates, rabi.probe, rabi.coupling,
                rabi.lo, rabi.signal, *runtime.detunings)
    actual = tuple(arc_real_scalar(getattr(acfg, name), f'LAB-004 published {name}') for name in
                   ('ARC_RF_DIPOLE_EA0', 'GAMMA_21', 'GAMMA_32', 'GAMMA_43', 'OMEGA_P', 'OMEGA_C',
                    'OMEGA_RF_BASE', 'OMEGA_SIGNAL_RF_BASE', 'DELTA_P', 'DELTA_C', 'DELTA_RF'))
    try:
        changed = (actual != expected or acfg.ATOMIC_LEVELS != runtime.levels() or
                   acfg._ATOMIC_LEVELS_CACHE[key] != runtime.levels() or
                   acfg._DIPOLE_MATRIX_CACHE[key] != runtime.dipoles() or
                   acfg._DETUNING_CACHE[key] != runtime.detuning_entry() or
                   acfg._OPTIMIZATION_RESULTS_CACHE[key].get('dipole_rf') != runtime.selection.winner.rf_ea0 or
                   acfg._RF_ORIENTATION != runtime.selection.rf_orientation)
    except (TypeError, ValueError) as exc:
        raise PhysicsCalculationError('LAB-004: malformed laboratory cache contents.') from exc
    if changed:
        raise PhysicsCalculationError('LAB-004: laboratory quantities/cache differ from the computed runtime metadata.')
    derived = tuple(arc_real_scalar(value, f'LAB-004 derived {name}') for name, value in
                    (('OMEGA_S_BASE', acfg.params.OMEGA_S_BASE), ('OMEGA_LO_BASE', acfg.params.OMEGA_LO_BASE),
                     ('LAMBDA_P', acfg.params.SI_Constants.LAMBDA_P)))
    if derived != (rabi.signal, rabi.lo, request.probe.wavelength_nm * 1e-9):
        raise PhysicsConfigurationError('LAB-004: derived heterodyne/probe-wavelength view changed; explicitly reinitialize.')
    return runtime


def read_laboratory_dipoles(acfg):
    runtime = require_current_laboratory_runtime(acfg)
    return copy.deepcopy(acfg._DIPOLE_MATRIX_CACHE[runtime.identity])
