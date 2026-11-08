"""Explicit selection and immutable runtime receipt for prepared radiative rates.

Parsing has no ARC imports or side effects. Resolved graphs stay local to one
profile execution; canonical initialization caches and compatibility GAMMAs
remain the legacy FS data, never the active selected-graph rate contract.
"""

from dataclasses import dataclass

from .laboratory import _half
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError


@dataclass(frozen=True)
class RadiativeOptions:
    model: str
    nuclear_projection: float | None

    def __post_init__(self):
        if not isinstance(self.model, str) or self.model not in (
                'legacy_fs_three_edge', 'selected_internal_spontaneous_e1'):
            raise PhysicsConfigurationError('RELAX-002: unknown explicit radiative model; no substitution.')
        if self.model == 'legacy_fs_three_edge':
            if self.nuclear_projection is not None:
                raise PhysicsConfigurationError('RELAX-002: legacy FS rates have no prepared nuclear projection.')
        else:
            object.__setattr__(self, 'nuclear_projection', _half(
                self.nuclear_projection, 'RELAX-002 explicit nuclear projection mI'))


def validate_profile_radiative(profile):
    options = profile.radiative
    if options is None:
        return
    if not isinstance(options, RadiativeOptions):
        raise PhysicsConfigurationError('RELAX-002: typed radiative declaration required.')
    if options.model == 'legacy_fs_three_edge':
        return
    if profile.spectrum is not None and profile.spectrum.analytical_enabled:
        raise PhysicsConfigurationError('RELAX-002: selected internal radiative graph requires '
                                        'spectrum.analytical_enabled=false; no analytical substitute.')
    if profile.mode == 'experimental':
        prep = profile.request.preparation
        mf, mj = prep.excited.mf, prep.rydberg1_mj
    else:
        states = profile.request.level_mappings()
        mf, mj = states['excited']['mf'], states['rydberg1']['mj']
    # This checks the declared channel, never chooses or infers an mI.
    if options.nuclear_projection != mf + profile.coupling.polarization - mj:
        raise PhysicsConfigurationError('RELAX-002: explicit mI disagrees with the declared coupling channel.')


def _atom_identity(ac):
    atom = ac.ATOM
    return (id(atom), type(atom).__module__, type(atom).__qualname__, float(atom.I))


def _runtime_identity(runtime):
    from .prescribed_runtime import _freeze
    return _freeze(runtime)


@dataclass(frozen=True)
class ResolvedProfileRadiative:
    declaration: RadiativeOptions
    atom_identity: tuple
    runtime_identity: tuple
    graph: object

    def __post_init__(self):
        from ..arc.prepared_decay import PreparedRadiativeGraph
        if (not isinstance(self.declaration, RadiativeOptions) or
                self.declaration.model != 'selected_internal_spontaneous_e1' or
                not isinstance(self.graph, PreparedRadiativeGraph) or
                self.graph.nuclear_projection != self.declaration.nuclear_projection or
                not isinstance(self.atom_identity, tuple) or not isinstance(self.runtime_identity, tuple)):
            raise PhysicsConfigurationError('RELAX-002: inconsistent resolved radiative receipt.')

    def require_current(self, ac, runtime):
        if _atom_identity(ac) != self.atom_identity or _runtime_identity(runtime) != self.runtime_identity:
            raise PhysicsConfigurationError('RELAX-002: stale atom/runtime for prepared radiative graph; re-execute profile.')
        roles = ('ground', 'excited', 'rydberg1', 'rydberg2')
        keys = ('n', 'l', 'j', 'f', 'mf', 'mj')
        states = tuple(tuple(ac.ATOMIC_LEVELS[role].get(key) for key in keys) for role in roles)
        if states != self.graph.states or float(ac.ATOM.I) != self.graph.nuclear_spin:
            raise PhysicsCalculationError('RELAX-002: prepared graph differs from the actual atomic states.')
        orientation = (runtime.selection.rf_orientation if hasattr(runtime, 'selection')
                       else runtime.rf_orientation)
        self.graph.require_orientation(orientation)


def resolve_profile_radiative(profile, ac):
    """Build locally AFTER canonical initialization, with no cache/publication writes."""
    validate_profile_radiative(profile)
    if profile.radiative is None or profile.radiative.model == 'legacy_fs_three_edge':
        return None
    from ..arc.prepared_decay import calculate_prepared_radiative_graph
    from ..config.prescribed_runtime import capture_prescribed_runtime

    def current():
        return (ac.get_laboratory_runtime() if profile.mode == 'experimental'
                else capture_prescribed_runtime(ac, profile))

    ac.require_atomic_physics_initialized()
    before = current()
    atom_identity = _atom_identity(ac)
    graph = calculate_prepared_radiative_graph(ac.ATOM, ac.ATOMIC_LEVELS,
                         nuclear_projection=profile.radiative.nuclear_projection)
    after = current()
    if before != after:
        raise PhysicsCalculationError('RELAX-002: atomic runtime changed during prepared rate calculation.')
    receipt = ResolvedProfileRadiative(profile.radiative, atom_identity, _runtime_identity(before), graph)
    receipt.require_current(ac, after)
    return receipt


def require_radiative_selection(options, resolved):
    """An explicit selected model cannot qualify without its matching graph."""
    if options is not None and not isinstance(options, RadiativeOptions):
        raise PhysicsConfigurationError('RELAX-002: typed radiative declaration required.')
    if resolved is not None and not isinstance(resolved, ResolvedProfileRadiative):
        raise PhysicsConfigurationError('RELAX-002: resolved radiative execution receipt required.')
    selected = options is not None and options.model == 'selected_internal_spontaneous_e1'
    if selected != (resolved is not None) or (selected and options != resolved.declaration):
        raise PhysicsConfigurationError('RELAX-002: execution graph does not match the explicit profile model.')


def active_radiative_rates(config, runtime, ac, profile=None):
    """No rebuilding, cache repair or model fallback at consumer boundaries."""
    resolved = config.radiative
    if resolved is not None and not isinstance(resolved, ResolvedProfileRadiative):
        raise PhysicsConfigurationError('RELAX-002: resolved radiative execution receipt required.')
    options = getattr(profile, 'radiative', None)
    if profile is not None and hasattr(profile, 'radiative'):
        require_radiative_selection(options, resolved)
    if resolved is None:
        return runtime.rates
    resolved.require_current(ac, runtime)
    return resolved.graph


def radiative_report(resolved):
    if resolved is None:
        return dict(model='legacy_fs_three_edge', temperature_k=0.,
                    magnetic_branching_resolved=False, closed_physical_space=False,
                    outside_space_rates_per_us=None, outside_return_model=None)
    return resolved.graph.report()


def named_internal_rates(resolved):
    return {f'{edge["source"]}->{edge["target"]}': edge['rate_per_us']
            for edge in resolved.graph.report()['edges']}
