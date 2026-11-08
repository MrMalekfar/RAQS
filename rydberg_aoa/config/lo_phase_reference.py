"""Explicit spatial LO phase offsets, referenced to the first sensor."""
from dataclasses import dataclass
from .laboratory import _real
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class SpatialLOReference:
    model: str
    phase_offsets_rad: tuple | None = None

    def __post_init__(self):
        if type(self.model) is not str or self.model not in ('uniform','per_element'):
            raise PhysicsConfigurationError('BBR-019: LO reference model must be uniform or per_element.')
        if self.model=='uniform':
            if self.phase_offsets_rad is not None:
                raise PhysicsConfigurationError('BBR-019: uniform LO has no phase map; unrelated values cannot be ignored.')
        else:
            if not isinstance(self.phase_offsets_rad,tuple) or not self.phase_offsets_rad:
                raise PhysicsConfigurationError('BBR-019: explicit nonempty immutable LO phase-offset tuple required.')
            values=tuple(_real(x,'BBR-019 LO phase offset [rad]') for x in self.phase_offsets_rad)
            if values[0]!=0.:
                raise PhysicsConfigurationError('BBR-019: first LO phase offset must be zero; input is relative to sensor 0.')
            object.__setattr__(self,'phase_offsets_rad',values)


def parse_lo_phase_reference(raw):
    if not isinstance(raw,dict):
        raise PhysicsConfigurationError('BBR-019: explicit LO reference object required.')
    model=raw.get('model')
    keys={'model','phase_offsets_rad'} if model=='per_element' else {'model'}
    if set(raw)!=keys:
        raise PhysicsConfigurationError(f'BBR-019: LO reference requires exactly {sorted(keys)}.')
    values=raw.get('phase_offsets_rad')
    if model=='per_element' and not isinstance(values,list):
        raise PhysicsConfigurationError('BBR-019: LO phase map must be a JSON array.')
    return SpatialLOReference(model,None if values is None else tuple(values))


def validate_lo_phase_reference(profile):
    reference=getattr(profile,'lo_phase_reference',None)
    if reference is None:return
    config=getattr(profile,'run_config',None)
    if not isinstance(reference,SpatialLOReference) or config is None:
        raise PhysicsConfigurationError('BBR-019: typed LO reference and explicit sensor geometry required.')
    if reference.model=='per_element' and len(reference.phase_offsets_rad)!=len(config.positions_m):
        raise PhysicsConfigurationError('BBR-019: LO phase map count must match the ordered sensor positions; no padding or truncation.')


def require_uniform_lo_execution(profile):
    validate_lo_phase_reference(profile)
    reference=getattr(profile,'lo_phase_reference',None)
    if reference is not None and reference.model=='per_element' and any(value!=0. for value in reference.phase_offsets_rad):
        raise PhysicsConfigurationError('BBR-019: nonuniform LO calibration is not connected to legacy sensing/studies; no ignored phase map. Use explicit diagnostics.')


def lo_phase_document(reference):
    result={'model':reference.model}
    if reference.model=='per_element':result['phase_offsets_rad']=list(reference.phase_offsets_rad)
    return result


def lo_phase_receipt(profile,diagnostics_only):
    """Report the spatial contract without claiming an array was simulated."""
    validate_lo_phase_reference(profile)
    reference=getattr(profile,'lo_phase_reference',None)
    count=None if profile.run_config is None else len(profile.run_config.positions_m)
    values=None if count is None else ([0.]*count if reference is None or reference.model=='uniform' else list(reference.phase_offsets_rad))
    temporal=getattr(getattr(profile,'magnetic_core',None),'temporal_diagnostic',None)
    scope=('reference element 0 only; other sensor phases not executed' if temporal is not None else 'atomic diagnostics only; spatial phases not executed') if diagnostics_only else 'uniform LO in existing sensing pipeline'
    return dict(declaration=None if reference is None else lo_phase_document(reference),
        source='legacy_uniform_assumption' if reference is None else 'profile',
        reference_element_index=0 if count is not None else None,phase_offsets_rad=values,
        scope=scope,
        hardware_validation='not established by simulation',
        relative_phase_definition='signal phase minus LO phase, with spatial offsets referenced to sensor 0')
