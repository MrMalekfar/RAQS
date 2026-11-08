"""Selected internal spontaneous E1 channels, not a closed atomic lifetime model.

FS-centroid energies and an isotropic zero-temperature radiation bath are used.
The nuclear projection of each Rydberg state must be declared explicitly. No
population outside the four states is returned, normalized or assigned a rate.
"""

from dataclasses import asdict, dataclass

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

from ._scan_validation import integer_scalar, real_scalar, arc_real_scalar
from .decay_rates import _validated_rate_per_us
from .rf_orientation import resolve_rf_orientation
from .types import AtomicLevel
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError


ROLES = ('ground', 'excited', 'rydberg1', 'rydberg2')
_QN = ('n', 'l', 'j', 'f', 'mf', 'mj')


def _half(value, name, *, minimum=None):
    value = real_scalar(value, 'RELAX-001A ' + name)
    if not (2 * value).is_integer() or (minimum is not None and value < minimum):
        raise PhysicsConfigurationError(f'RELAX-001A: invalid {name}={value}.')
    return Rational(value)


def _projection(value, total, name):
    value = _half(value, name)
    if abs(value) > total or (total - value).q != 1:
        raise PhysicsConfigurationError(f'RELAX-001A: invalid {name}={value} for angular momentum {total}.')
    return value


def _states(levels, nuclear_spin):
    if not isinstance(levels, dict) or set(levels) != set(ROLES):
        raise PhysicsConfigurationError('RELAX-001A: exactly four named prepared states are required.')
    result = []
    for role in ROLES:
        item = levels[role]
        if isinstance(item, AtomicLevel):
            item = asdict(item)
        if not isinstance(item, dict):
            raise PhysicsConfigurationError(f'RELAX-001A: invalid {role} declaration.')
        try:
            n = integer_scalar(item['n'], role + '.n', minimum=1)
            l = integer_scalar(item['l'], role + '.l')
            j = _half(item['j'], role + '.j', minimum=.5)
            if l >= n or j not in (Rational(l) - Rational(1, 2), Rational(l) + Rational(1, 2)):
                raise PhysicsConfigurationError(f'RELAX-001A: invalid alkali FS state {role}.')
            f, mf, mj = item.get('f'), item.get('mf'), item.get('mj')
            if role in ROLES[:2]:
                f = _half(f, role + '.F', minimum=0)
                if not abs(nuclear_spin - j) <= f <= nuclear_spin + j or (nuclear_spin + j - f).q != 1:
                    raise PhysicsConfigurationError(f'RELAX-001A: invalid {role}.F for I,j.')
                mf = _projection(mf, f, role + '.mF')
                if mj is not None:
                    raise PhysicsConfigurationError(f'RELAX-001A: {role} requires HFS, not simultaneous mJ.')
            else:
                if f is not None or mf is not None:
                    raise PhysicsConfigurationError(f'RELAX-001A: {role} requires an FS mJ state.')
                mj = _projection(mj, j, role + '.mJ')
            result.append((n, l, float(j), None if f is None else float(f),
                           None if mf is None else float(mf), None if mj is None else float(mj)))
        except KeyError as exc:
            raise PhysicsConfigurationError(f'RELAX-001A: incomplete {role} declaration.') from exc
    if len(set(result)) != 4:
        raise PhysicsConfigurationError('RELAX-001A: the four states must be distinct.')
    return tuple(result)


def _electronic_branch(js, ms, jd, md):
    if abs(ms) > js or abs(md) > jd or abs(ms - md) > 1:
        return Rational(0)
    return clebsch_gordan(jd, 1, js, md, ms - md, ms) ** 2


def _branch(source, target, nuclear_spin, nuclear_projection):
    js, jd = Rational(source[2]), Rational(target[2])
    fs, fd = source[3], target[3]
    if fs is not None and fd is not None:
        fs, fd, ms, md = map(Rational, (fs, fd, source[4], target[4]))
        if abs(ms - md) > 1:
            return Rational(0)
        return ((2*js+1)*(2*fs+1)*(2*fd+1)*
                wigner_6j(jd, fd, nuclear_spin, fs, js, 1)**2 *
                wigner_3j(fd, 1, fs, -md, md-ms, ms)**2)
    mi = nuclear_projection
    if fs is not None:
        ms = Rational(source[4]) - mi
        if abs(ms) > js:
            return Rational(0)
        cg = clebsch_gordan(js, nuclear_spin, Rational(fs), ms, mi, Rational(source[4]))**2
        return cg * _electronic_branch(js, ms, jd, Rational(target[5]))
    if fd is not None:
        md = Rational(target[4]) - mi
        if abs(md) > jd:
            return Rational(0)
        cg = clebsch_gordan(jd, nuclear_spin, Rational(fd), md, mi, Rational(target[4]))**2
        return cg * _electronic_branch(js, Rational(source[5]), jd, md)
    return _electronic_branch(js, Rational(source[5]), jd, Rational(target[5]))


@dataclass(frozen=True)
class PreparedDecayEdge:
    """One energy-downward FS line and its prepared-state probability; µs⁻¹."""
    source: int  # canonical zero-based state index
    target: int
    fs_rate_per_us: float
    branch: float

    def __post_init__(self):
        for name in ('source', 'target'):
            if integer_scalar(getattr(self, name), 'RELAX-001A ' + name) > 3:
                raise PhysicsConfigurationError('RELAX-001A: state index outside the four-state basis.')
        if self.source == self.target:
            raise PhysicsConfigurationError('RELAX-001A: radiative self-edge is invalid.')
        rate = real_scalar(self.fs_rate_per_us, 'RELAX-001A FS rate [µs⁻¹]')
        branch = real_scalar(self.branch, 'RELAX-001A branch')
        if rate < 0 or not 0 <= branch <= 1:
            raise PhysicsConfigurationError('RELAX-001A: invalid rate or branch probability.')

    @property
    def rate_per_us(self):
        value = self.fs_rate_per_us * self.branch
        if self.fs_rate_per_us > 0 and self.branch > 0 and value == 0:
            raise PhysicsCalculationError('RELAX-001A: partial-rate underflow; no zero substitution.')
        return value


@dataclass(frozen=True)
class PreparedRadiativeGraph:
    """Picklable internal-channel approximation; outside-space rates UNKNOWN.

    This object deliberately is not a three-number gamma sequence. Passing it
    through the existing solver's ``gammas`` argument retains its named edges.
    """
    states: tuple
    nuclear_spin: float
    nuclear_projection: float
    signed_rf_frequency_hz: float
    edges: tuple[PreparedDecayEdge, ...]

    def __post_init__(self):
        spin = _half(self.nuclear_spin, 'I', minimum=0)
        _projection(self.nuclear_projection, spin, 'mI')
        if not isinstance(self.states, tuple) or len(self.states) != 4:
            raise PhysicsConfigurationError('RELAX-001A: immutable four-state identity required.')
        if any(not isinstance(s, tuple) or len(s) != len(_QN) for s in self.states):
            raise PhysicsConfigurationError('RELAX-001A: malformed state identity.')
        if _states({r: dict(zip(_QN, s)) for r, s in zip(ROLES, self.states)}, spin) != self.states:
            raise PhysicsConfigurationError('RELAX-001A: inconsistent state identity.')
        if not isinstance(self.edges, tuple) or any(not isinstance(e, PreparedDecayEdge) for e in self.edges):
            raise PhysicsConfigurationError('RELAX-001A: immutable prepared edges required.')
        pairs = [(e.source, e.target) for e in self.edges]
        if len(set(pairs)) != len(pairs) or any((b, a) in pairs for a, b in pairs):
            raise PhysicsConfigurationError('RELAX-001A: duplicate/reverse spontaneous edges are invalid.')
        direction = self.rf_orientation
        for e in self.edges:
            if {e.source, e.target} == {2, 3} and (e.source, e.target) != (
                    direction.upper_ladder_index-1, direction.lower_ladder_index-1):
                raise PhysicsConfigurationError('RELAX-001A: RF edge contradicts its energy orientation.')

    @property
    def rf_orientation(self):
        value = real_scalar(self.signed_rf_frequency_hz, 'RELAX-001A signed RF frequency [Hz]')
        if value == 0:
            raise PhysicsConfigurationError('RELAX-001A: degenerate RF states are unsupported.')
        return resolve_rf_orientation(value)

    def require_orientation(self, orientation):
        expected = self.rf_orientation
        if orientation is not None and orientation != expected:
            raise PhysicsConfigurationError('RELAX-001A: supplied RF orientation differs from the radiative graph.')
        return expected

    @property
    def outgoing_rates_per_us(self):
        return tuple(arc_real_scalar(sum(e.rate_per_us for e in self.edges if e.source == i),
                                    f'RELAX-001A internal outgoing rate of {ROLES[i]} [µs⁻¹]')
                     for i in range(4))

    def report(self):
        return dict(model='selected_internal_spontaneous_e1', temperature_k=0.,
                    energy_convention='fs_centroid', closed_physical_space=False,
                    outside_space_rates_per_us=None, outside_return_model=None,
                    states={r: dict(zip(_QN, s)) for r, s in zip(ROLES, self.states)},
                    nuclear_spin=self.nuclear_spin, nuclear_projection=self.nuclear_projection,
                    signed_rf_frequency_hz=self.signed_rf_frequency_hz,
                    edges=[dict(source=ROLES[e.source], target=ROLES[e.target],
                                fs_rate_per_us=e.fs_rate_per_us, branch=e.branch,
                                rate_per_us=e.rate_per_us) for e in self.edges])


def calculate_prepared_radiative_graph(atom, levels, *, nuclear_projection):
    """Calculate only internal spontaneous E1 rates for explicit prepared states.

    ``levels`` contains HFS ground/excited and FS Rydberg states. ``mI`` is
    explicit, never inferred here. No ARC lifetime sum or nuclear average is
    substituted. Changing applied field amplitudes cannot change these rates.
    """
    spin = _half(atom.I, 'I', minimum=0)
    mi = _projection(nuclear_projection, spin, 'mI')
    states = _states(levels, spin)

    def call(method, a, b, **kwargs):
        try:
            return getattr(atom, method)(*a[:3], *b[:3], **kwargs)
        except PhysicsError:
            raise
        except Exception as exc:
            raise PhysicsCalculationError(f'RELAX-001A: ARC {method} failed for {a} -> {b}.') from exc

    signed_rf = arc_real_scalar(call('getTransitionFrequency', states[2], states[3]),
                                'RELAX-001A signed RF frequency [Hz]', nonzero=True)
    edges = []
    for a in range(4):
        for b in range(a+1, 4):
            sa, sb = states[a], states[b]
            if abs(sa[1]-sb[1]) != 1 or abs(sa[2]-sb[2]) > 1:
                continue
            signed = signed_rf if (a, b) == (2, 3) else arc_real_scalar(
                call('getTransitionFrequency', sa, sb), 'RELAX-001A E1 frequency [Hz]', nonzero=True)
            source, target = (b, a) if signed > 0 else (a, b)
            probability = _branch(states[source], states[target], spin, mi)
            if not 0 <= probability <= 1:
                raise PhysicsCalculationError('RELAX-001A: angular probability outside [0,1].')
            raw_rate = call('getTransitionRate', states[source], states[target], temperature=0.)
            rate = _validated_rate_per_us(raw_rate, f'RELAX-001A {ROLES[source]} -> {ROLES[target]}')
            if raw_rate > 0 and rate == 0:
                raise PhysicsCalculationError('RELAX-001A: event-rate unit conversion underflow; no zero substitution.')
            edges.append(PreparedDecayEdge(source, target, rate, float(probability)))
    return PreparedRadiativeGraph(states, float(spin), float(mi), signed_rf, tuple(edges))
