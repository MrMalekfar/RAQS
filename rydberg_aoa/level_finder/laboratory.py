"""LAB-003: optically accessible, RF-first lab manifold selection.

Consumes LAB-002 inputs explicitly. No global configuration/caches are read
or written; no mJ/q/DME scan, channel resolution, tuning or initialization.
Rydberg outputs remain FS-only until the separate channel-resolution stage.
"""

from dataclasses import dataclass, asdict
import math

from ..arc.types import AtomicLevel
from ..arc._scan_validation import arc_real_scalar
from ..config.constants import C_LIGHT
from ..config.laboratory import ExperimentalLabRequest, OpticalEnergyConvention
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError, PhysicsError
from .arc_level_finder import scan_fs_rf_candidates, resolve_rf_first_manifold, rf_first_ranking_key


def _arc_call(operation, context):
    try:
        return operation()
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f'LAB-003: {context} failed; no substitute calculation.') from exc


def _frequency(atom, lower, upper):
    context = f'ARC frequency ({lower.n},{lower.l},{lower.j})->({upper.n},{upper.l},{upper.j}) [Hz]'
    value = _arc_call(lambda: atom.getTransitionFrequency(lower.n, lower.l, lower.j,
                                                        upper.n, upper.l, upper.j), context)
    return arc_real_scalar(value, context, nonzero=True)


def _hfs_shift(atom, state):
    context = f'ARC HFS n={state.n},l={state.l},j={state.j},F={state.f} [Hz]'
    coefficients = _arc_call(lambda: atom.getHFSCoefficients(state.n, state.l, state.j), context)
    if not isinstance(coefficients, (tuple, list)) or len(coefficients) != 2:
        raise PhysicsCalculationError(f'LAB-003: invalid A,B coefficients for {context}; no zero-HFS substitute.')
    a, b = (arc_real_scalar(value, context) for value in coefficients)
    value = _arc_call(lambda: atom.getHFSEnergyShift(state.j, state.f, a, b), context)
    return arc_real_scalar(value, context)


@dataclass(frozen=True)
class OpticalAccess:
    """All frequencies cyclic Hz, wavelengths nm, angular detuning rad/µs.

    Applied frequency = c / applied wavelength + detuning_mhz * 1e6.
    Proposed carrier wavelength = c / (resonance - that fixed offset).
    A proposal is diagnostic output only; no field is modified.
    """
    applied_wavelength_nm: float
    offset_mhz: float
    applied_frequency_hz: float
    resonant_frequency_hz: float
    resonant_wavelength_nm: float
    proposed_wavelength_nm: float
    detuning_hz: float
    detuning_rad_per_us: float


def _optical_access(laser, resonance_hz, context):
    resonance_hz = arc_real_scalar(resonance_hz, context, positive=True)
    offset_hz = laser.detuning_mhz * 1e6
    wavelength_m = laser.wavelength_nm * 1e-9
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise PhysicsConfigurationError(f'LAB-003: {context} wavelength cannot be represented in metres.')
    applied_hz = C_LIGHT / wavelength_m + offset_hz
    proposed_carrier_hz = resonance_hz - offset_hz
    if not all(math.isfinite(v) and v > 0 for v in (applied_hz, proposed_carrier_hz)):
        raise PhysicsConfigurationError(f'LAB-003: {context} has an invalid applied frequency/fixed offset; no tuning repair.')
    delta = arc_real_scalar(applied_hz - resonance_hz, f'{context} detuning [Hz]')
    values = (C_LIGHT / resonance_hz * 1e9, C_LIGHT / proposed_carrier_hz * 1e9,
              2 * math.pi * delta / 1e6)
    for value in values:
        arc_real_scalar(value, f'{context} derived wavelength/detuning')
    return OpticalAccess(laser.wavelength_nm, laser.detuning_mhz, applied_hz,
                         resonance_hz, values[0], values[1], delta, values[2])


@dataclass(frozen=True)
class FSManifold:
    """FS output only. A declared request projection is not yet resolved."""
    n: int
    l: int
    j: float

    @property
    def mj(self):
        return None


@dataclass(frozen=True)
class LabCandidate:
    rydberg1: FSManifold
    rydberg2: FSManifold
    signed_frequency_hz: float
    carrier_hz: float
    mismatch_hz: float
    coupling: OpticalAccess


@dataclass(frozen=True)
class LabSelection:
    """A feasible FS selection, not a published four-state sensor runtime."""
    request: ExperimentalLabRequest
    winner: LabCandidate
    candidates: tuple
    probe: OpticalAccess
    rf_orientation: object
    fs_pairs_checked: int
    rf_candidates_count: int
    projection_exclusions: int
    optical_exclusions: int
    ground_hfs_shift_hz: float
    excited_hfs_shift_hz: float

    def level_objects(self):
        """Detached level objects with R1/R2.mj=None; no channel inference."""
        return {'ground': AtomicLevel(**asdict(self.request.preparation.ground)),
                'excited': AtomicLevel(**asdict(self.request.preparation.excited)),
                'rydberg1': AtomicLevel(**asdict(self.winner.rydberg1)),
                'rydberg2': AtomicLevel(**asdict(self.winner.rydberg2))}


def select_laboratory_manifolds(atom, request):
    """Filter hardware/projection feasibility BEFORE deterministic RF ranking.

    Structural mJ support is checked against each j at the fixed requested
    channel. No magnetic values/polarizations are enumerated, no dipole enters
    ranking, and angular-coefficient zeros remain a later channel gate.
    Negative optical transition energies are valid domain exclusions: an
    absorption ladder cannot use them. Invalid ARC data/errors abort the scan.
    """
    if not isinstance(request, ExperimentalLabRequest):
        raise PhysicsConfigurationError('LAB-003: selection requires an explicit ExperimentalLabRequest.')
    request.validate_for_atom(atom)
    preparation, bounds = request.preparation, request.bounds
    g, e = preparation.ground, preparation.excited
    ground_shift, excited_shift = 0., 0.
    if request.optical_energy_convention is OpticalEnergyConvention.LOW_LEVEL_HFS:
        ground_shift, excited_shift = _hfs_shift(atom, g), _hfs_shift(atom, e)
    # Zero shifts above implement the DECLARED FS-centroid approximation,
    # never a rescue for missing HFS data.
    probe_hz = arc_real_scalar(_frequency(atom, g, e) + excited_shift - ground_shift,
                               'probe resonance [Hz]')
    if probe_hz <= 0:
        raise PhysicsCalculationError('LAB-003: prepared probe is not an energy-upward absorption transition.')
    probe = _optical_access(request.probe, probe_hz, 'probe')
    if not bounds.probe_window_nm[0] <= probe.proposed_wavelength_nm <= bounds.probe_window_nm[1]:
        raise PhysicsCalculationError(
            f'LAB-003: prepared probe resonance/setpoint {probe.proposed_wavelength_nm:g} nm '
            f'is inaccessible in {bounds.probe_window_nm}; no alternative excited state or tuning.')
    excited = AtomicLevel(**asdict(e))
    scanned, checked = scan_fs_rf_candidates(atom, (bounds.n_min, bounds.n_max + 1),
                                            excited, request.selection_carrier_hz,
                                            bounds.rf_tolerance_hz, max_l=bounds.max_l)
    eligible, optical_cache = [], {}
    projection_exclusions = optical_exclusions = 0
    for candidate in scanned:
        r1, r2 = candidate['from_level'], candidate['to_level']
        if any(abs(mj) > state.j or not (state.j - mj).is_integer()
               for state, mj in ((r1, preparation.rydberg1_mj), (r2, preparation.rydberg2_mj))):
            projection_exclusions += 1
            continue
        identity = (r1.n, r1.l, r1.j)
        if identity not in optical_cache:
            coupling_hz = arc_real_scalar(_frequency(atom, e, r1) - excited_shift,
                                          f'coupling resonance {identity} [Hz]')
            optical_cache[identity] = (None if coupling_hz <= 0 else
                                       _optical_access(request.coupling, coupling_hz, f'coupling {identity}'))
        optical = optical_cache[identity]
        if optical is None or not bounds.coupling_window_nm[0] <= optical.proposed_wavelength_nm <= bounds.coupling_window_nm[1]:
            optical_exclusions += 1
            continue
        eligible.append({**candidate, 'lab_coupling': optical})
    if not eligible:
        raise PhysicsCalculationError(
            f'LAB-003: No feasible RF-first manifold in n={bounds.n_min}..{bounds.n_max}, '
            f'l<= {bounds.max_l}, RF={request.selection_carrier_hz:g} Hz, '
            f'tolerance={bounds.rf_tolerance_hz:g} Hz, coupling window={bounds.coupling_window_nm}; '
            f'RF candidates={len(scanned)}, projection exclusions={projection_exclusions}, '
            f'optical exclusions={optical_exclusions}. No retry, widened domain or substituted channel.')
    eligible.sort(key=rf_first_ranking_key)
    selected = resolve_rf_first_manifold(eligible, atom, request.selection_carrier_hz)
    if selected['signed_transition_freq_hz'] != eligible[0]['signed_transition_freq_hz']:
        raise PhysicsCalculationError('LAB-003: ARC RF frequency changed during selection; ranking is stale.')

    def report(candidate):
        r1, r2 = candidate['from_level'], candidate['to_level']
        signed = candidate['signed_transition_freq_hz']
        carrier = abs(signed)
        return LabCandidate(FSManifold(r1.n, r1.l, r1.j), FSManifold(r2.n, r2.l, r2.j),
                            signed, carrier, abs(carrier - request.selection_carrier_hz),
                            candidate['lab_coupling'])

    return LabSelection(request, report(selected), tuple(report(c) for c in eligible),
                        probe, selected['rf_orientation'], checked, len(scanned),
                        projection_exclusions, optical_exclusions, ground_shift, excited_shift)
