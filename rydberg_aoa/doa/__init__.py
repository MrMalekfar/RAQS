"""Direction-of-arrival (AoA) estimation architecture (DOA-ARCH-001).

Geometry (ArrayGeometry), steering (steering_vector / steering_matrix),
estimators (DOAEstimator / DOAResult) and the complex IF snapshot layer
(IFSnapshotData / extract_if_snapshots / sample_covariance) are decoupled
so estimator choice, array geometry and the narrowband data contract are
independent.  Selectable estimators are phase_beamforming, bartlett,
mvdr/capon, music, esprit and root_music (DOA-ALG-001/002).
"""

from .geometry import ArrayGeometry, ShiftInvariantPair
from .steering import (
    arrival_phases,
    direction_unit_vector,
    steering_matrix,
    steering_vector,
    wave_number,
)
from .covariance_estimators import (
    BartlettEstimator,
    MUSICEstimator,
    MVDREstimator,
    SnapshotDOAEstimator,
)
from .structured_estimators import ESPRITEstimator, RootMUSICEstimator
from .estimators import (
    DOAError,
    DOAEstimator,
    DOAResult,
    IncompatibleGeometryError,
    InvalidContextError,
    PhaseBeamformingEstimator,
)
from .registry import get_doa_estimator
from .snapshots import IFSnapshotData, extract_if_snapshots, sample_covariance

__all__ = [
    "ArrayGeometry",
    "ShiftInvariantPair",
    "arrival_phases",
    "direction_unit_vector",
    "steering_matrix",
    "steering_vector",
    "wave_number",
    "DOAError",
    "DOAEstimator",
    "DOAResult",
    "IncompatibleGeometryError",
    "InvalidContextError",
    "PhaseBeamformingEstimator",
    "SnapshotDOAEstimator",
    "BartlettEstimator",
    "MVDREstimator",
    "MUSICEstimator",
    "ESPRITEstimator",
    "RootMUSICEstimator",
    "get_doa_estimator",
    "IFSnapshotData",
    "extract_if_snapshots",
    "sample_covariance",
]