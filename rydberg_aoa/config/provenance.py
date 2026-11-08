"""Input declarations are not evidence of calibrated experimental hardware."""
from dataclasses import dataclass, asdict
from ..exceptions import PhysicsConfigurationError


@dataclass(frozen=True)
class InputProvenance:
    classification: str
    note: str

    def __post_init__(self):
        if not isinstance(self.classification,str) or self.classification not in (
                'simulation_assumptions','unverified_inputs'):
            raise PhysicsConfigurationError('NOISE-ALIGN-001: input classification must be simulation_assumptions or unverified_inputs; calibration cannot be certified by a label.')
        if not isinstance(self.note,str) or not self.note.strip():
            raise PhysicsConfigurationError('NOISE-ALIGN-001: nonempty provenance note required.')


def provenance_report(provenance):
    if provenance is not None and not isinstance(provenance,InputProvenance):
        raise PhysicsConfigurationError('NOISE-ALIGN-001: validated InputProvenance required before initialization.')
    data={'classification':'not_declared','note':None} if provenance is None else asdict(provenance)
    return {**data,'experimental_calibration_verified':False,
            'experimental_mode_meaning':'RF-first atomic resolution, not evidence of hardware calibration'}
