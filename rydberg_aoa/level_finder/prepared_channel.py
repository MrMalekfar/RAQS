"""AUTO-001B: RF-first selection in one explicitly prepared magnetic channel.

This is an explicit selection API, not runtime initialization. No magnetic
scan, alternate polarization, DME ranking, global publication or tuning.
"""

from dataclasses import asdict, dataclass
from ..arc._scan_validation import arc_real_scalar
from ..arc.prepared_channel import _angular_zero, _dipole, _prepared_rabi, PreparedRabiFrequencies
from ..arc.types import AtomicLevel, validate_magnetic_quantum_number
from ..arc.rf_orientation import resolve_rf_orientation, RFTransitionOrientation
from ..config.laboratory import ExperimentalLabRequest
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from .laboratory import LabSelection, LabCandidate, _arc_call, select_laboratory_manifolds


@dataclass(frozen=True)
class PreparedFSState:
    n: int
    l: int
    j: float
    mj: float


@dataclass(frozen=True)
class PreparedCandidate:
    manifold: LabCandidate
    rydberg1: PreparedFSState
    rydberg2: PreparedFSState
    coupling_ea0: float
    rf_ea0: float


@dataclass(frozen=True)
class PreparedLabSelection:
    """Prepared-channel result; no solver, caches or runtime are published.

    optical_selection is the provisional LAB-003 FS domain. Its winner may
    be dark in this channel. winner below is the RF-first NONDARK result.
    """
    optical_selection: LabSelection
    candidates: tuple
    probe_ea0: float
    dark_exclusions: tuple
    implied_excited_mj: float
    nuclear_projection: float
    rf_orientation: RFTransitionOrientation
    rabi: PreparedRabiFrequencies

    @property
    def request(self):
        return self.optical_selection.request

    @property
    def winner(self):
        return self.candidates[0]

    def level_objects(self):
        """Detached canonical G,E,R1,R2; mF and mJ are never conflated."""
        return {'ground': AtomicLevel(**asdict(self.request.preparation.ground)),
                'excited': AtomicLevel(**asdict(self.request.preparation.excited)),
                'rydberg1': AtomicLevel(**asdict(self.winner.rydberg1)),
                'rydberg2': AtomicLevel(**asdict(self.winner.rydberg2))}


def select_prepared_laboratory_channel(atom, request):
    """Keep fixed mJ/q, filter exact angular/ARC zeros, then RF-first rank.

    A valid dark candidate is outside the driven-channel domain. Any invalid
    ARC value/exception aborts the calculation; it is never a dark exclusion.
    Zero input power/field switches that drive off, not its intrinsic DME.
    """
    if not isinstance(request, ExperimentalLabRequest):
        raise PhysicsConfigurationError('AUTO-001B requires an explicit ExperimentalLabRequest.')
    request.validate_for_atom(atom)
    p = request.preparation
    g, e = AtomicLevel(**asdict(p.ground)), AtomicLevel(**asdict(p.excited))
    mj_source = p.rydberg1_mj - request.coupling.polarization
    validate_magnetic_quantum_number(mj_source, e.j, 'implied excited mJ', 'excited j')
    mi = e.mf - mj_source  # Nuclear projection carried by both Rydberg states.
    if _angular_zero('probe', g, e, request.probe.polarization, atom.I):
        raise PhysicsConfigurationError('AUTO-001B: prepared probe has an exact zero angular coefficient; no alternate mF/q.')
    probe = _dipole(atom, 'getDipoleMatrixElementHFS',
                    (g.n, g.l, g.j, g.f, g.mf, e.n, e.l, e.j, e.f, e.mf,
                     request.probe.polarization), 'probe HFS→HFS')
    if probe == 0:
        raise PhysicsConfigurationError('AUTO-001B: prepared probe has zero ARC dipole; no alternate mF/q.')
    selection = select_laboratory_manifolds(atom, request)
    candidates, dark, coupling_cache = [], [], {}
    for candidate in selection.candidates:  # Already deterministically RF-ranked.
        r1 = PreparedFSState(candidate.rydberg1.n, candidate.rydberg1.l,
                             candidate.rydberg1.j, p.rydberg1_mj)
        r2 = PreparedFSState(candidate.rydberg2.n, candidate.rydberg2.l,
                             candidate.rydberg2.j, p.rydberg2_mj)
        if r1 not in coupling_cache:
            coupling_cache[r1] = (0. if _angular_zero('coupling', e, r1, request.coupling.polarization, atom.I)
                                  else _dipole(atom, 'getDipoleMatrixElementHFStoFS',
                                              (e.n, e.l, e.j, e.f, e.mf, r1.n, r1.l, r1.j,
                                               r1.mj, request.coupling.polarization), f'coupling HFS→FS {r1}'))
        coupling = coupling_cache[r1]
        if coupling == 0:
            dark.append((candidate, 'coupling'))
            continue
        rf = (0. if _angular_zero('rf', r1, r2, request.lo.polarization, atom.I)
              else _dipole(atom, 'getDipoleMatrixElement',
                           (r1.n, r1.l, r1.j, r1.mj, r2.n, r2.l, r2.j, r2.mj,
                            request.lo.polarization), f'RF FS→FS {r1}→{r2}'))
        if rf == 0:
            dark.append((candidate, 'rf'))
            continue
        candidates.append(PreparedCandidate(candidate, r1, r2, coupling, rf))
    if not candidates:
        raise PhysicsConfigurationError(f'AUTO-001B: no driven candidate in the declared channel; '
                                        f'dark exclusions={len(dark)}. No alternate mJ/q, retry or floor.')
    winner = candidates[0]
    r1, r2 = winner.rydberg1, winner.rydberg2
    fresh = _arc_call(lambda: atom.getTransitionFrequency(r1.n, r1.l, r1.j, r2.n, r2.l, r2.j),
                      f'AUTO-001B winner RF frequency {r1}→{r2}')
    fresh = arc_real_scalar(fresh, 'AUTO-001B winner signed RF [Hz]', nonzero=True)
    if fresh != winner.manifold.signed_frequency_hz:
        raise PhysicsCalculationError('AUTO-001B: ARC frequency changed during channel calculation; ranking is stale.')
    orientation = resolve_rf_orientation(fresh)
    rabi = _prepared_rabi(probe, winner.coupling_ea0, winner.rf_ea0, request)
    return PreparedLabSelection(selection, tuple(candidates), probe, tuple(dark),
                                mj_source, mi, orientation, rabi)
