"""DOA estimator interface and the current production phase-beamforming estimator.

The phase-beamforming estimator wraps the legacy delay-and-sum beamformer
(rydberg_aoa.tools.analysis.estimate_aoa_beamforming) behind the common
interface.  The scan grid and the mathematics are the same as the legacy
implementation, so results are numerically unchanged (for the frozen
six-path ULA: true AoA -30 deg, estimated AoA -30 deg, phase progression
-22.5 deg per path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .geometry import ArrayGeometry
from .steering import steering_vector

# Same scan grid as the legacy production estimator: one degree steps
# over the visible half-plane.
DEFAULT_AZIMUTH_SCAN_DEG = np.linspace(-90.0, 90.0, 181)


class DOAError(Exception):
    """Base class for direction-of-arrival estimation errors."""


class IncompatibleGeometryError(DOAError):
    """Algorithm/geometry mismatch detected up front (never a silent fallback)."""

    def __init__(self, algorithm, required, missing):
        self.algorithm = algorithm
        self.required = frozenset(required)
        self.missing = frozenset(missing)
        super().__init__(
            f"DOA estimator {algorithm!r} requires geometry capabilities "
            f"{sorted(self.required)}; geometry is missing {sorted(self.missing)}. "
            "Refusing to estimate on an incompatible geometry."
        )


class InvalidContextError(DOAError):
    """The estimation context is missing a required key."""


@dataclass
class DOAResult:
    """Structured estimator output (algorithm-agnostic)."""

    algorithm: str
    azimuth_deg: float
    elevation_deg: Optional[float] = None
    spectrum: Optional[np.ndarray] = None
    scan_grid: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.spectrum is not None:
            self.spectrum = np.asarray(self.spectrum)
        if self.scan_grid is not None:
            self.scan_grid = np.asarray(self.scan_grid)


class DOAEstimator(ABC):
    """Common interface for all DOA estimators.

    Supports:
        supports(geometry, data) -> bool
        estimate(data, geometry, context) -> DOAResult
    """

    algorithm: str = "base"
    required_capabilities: frozenset = frozenset()

    def supports(self, geometry, data):
        """Validate the algorithm/geometry/data combination up front."""
        if not isinstance(geometry, ArrayGeometry):
            return False
        arr = np.asarray(data)
        if arr.ndim != 1 or arr.shape[0] != geometry.m:
            return False
        if not np.all(np.isfinite(arr)):
            return False
        caps = geometry.capabilities
        return all(caps.get(key, False) for key in self.required_capabilities)

    def _require_supported(self, geometry, data):
        """Structural gate: raise a descriptive error when unsupported."""
        if self.supports(geometry, data):
            return
        if not isinstance(geometry, ArrayGeometry):
            raise DOAError(
                f"estimator {self.algorithm!r} requires an ArrayGeometry; "
                f"got {type(geometry).__name__}"
            )
        arr = np.asarray(data)
        if arr.ndim != 1 or arr.shape[0] != geometry.m:
            raise DOAError(
                f"estimator {self.algorithm!r} requires 1-D element responses "
                f"with {geometry.m} entries (geometry.m); got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise DOAError(f"data contains non-finite values")
        missing = [
            key for key in self.required_capabilities
            if not geometry.capabilities.get(key, False)
        ]
        raise IncompatibleGeometryError(self.algorithm, self.required_capabilities, missing)

    def estimate(self, data, geometry, context=None):
        """Common entry point: capability gate, then the algorithm core.

        Subclasses implement _estimate_impl; the gate is enforced here so no
        estimator can silently run on an incompatible geometry.
        """
        self._require_supported(geometry, data)
        return self._estimate_impl(data, geometry, context)

    @abstractmethod
    def _estimate_impl(self, data, geometry, context=None):
        """Algorithm-specific estimation core (capabilities already validated).

        Args:
            data: complex element/path responses, shape (M,).
            geometry: ArrayGeometry of the receiver array.
            context: mapping that must provide "frequency_hz" — the RF
                signal carrier in Hz used for spatial steering.
        """


class PhaseBeamformingEstimator(DOAEstimator):
    """Current production estimator: conventional delay-and-sum beamforming.

    The mathematics is the legacy implementation:
        phi(az) = k_vec . r          (k = 2*pi*f_RF/c, see steering)
        s(az)   = exp(-j phi(az))
        P(az)   = |s(az)^T x|^2

    An exactly zero response cannot supply an arrival angle. No amplitude
    threshold is imposed on nonzero responses; unrepresentable scan power
    fails explicitly instead of selecting an arbitrary scan-grid entry.
    """

    algorithm = "phase_beamforming"
    required_capabilities = frozenset({"arbitrary"})

    def __init__(self, scan_azimuth_deg=None, elevation_deg=0.0):
        if scan_azimuth_deg is None:
            self.scan_azimuth_deg = DEFAULT_AZIMUTH_SCAN_DEG
        else:
            grid = np.asarray(scan_azimuth_deg, dtype=float)
            if grid.ndim != 1 or grid.shape[0] < 1:
                raise ValueError("scan_azimuth_deg must be a non-empty 1-D grid")
            self.scan_azimuth_deg = grid
        self.elevation_deg = float(elevation_deg)

    def _estimate_impl(self, data, geometry, context=None):
        ctx = dict(context) if context else {}
        frequency_hz = ctx.get("frequency_hz")
        if frequency_hz is None:
            raise InvalidContextError(
                "context must provide 'frequency_hz' (RF signal carrier, Hz). "
                "The heterodyne IF frequency is not a spatial steering frequency."
            )
        arr = np.asarray(data)
        if not np.any(arr != 0):
            raise DOAError(
                "phase_beamforming cannot estimate an angle from an exactly "
                "zero element response; no arrival angle is available."
            )
        grid = self.scan_azimuth_deg
        power = np.empty(grid.shape[0], dtype=float)
        positions = geometry.positions_m
        try:
            with np.errstate(over="raise", invalid="raise"):
                for index in range(grid.shape[0]):
                    s = steering_vector(
                        positions, frequency_hz, float(grid[index]), self.elevation_deg)
                    power[index] = abs(np.dot(s, arr)) ** 2
        except FloatingPointError as exc:
            raise DOAError(
                "phase_beamforming scan power cannot be represented safely; "
                "no arrival angle was selected."
            ) from exc
        if not np.all(np.isfinite(power)) or not np.any(power > 0):
            raise DOAError(
                "phase_beamforming requires finite scan power with at least "
                "one positive value; no arrival angle was selected."
            )
        peak_index = int(np.argmax(power))
        return DOAResult(
            algorithm=self.algorithm,
            azimuth_deg=float(grid[peak_index]),
            elevation_deg=self.elevation_deg,
            spectrum=power,
            scan_grid=grid,
            diagnostics={
                "frequency_hz": frequency_hz,
                "peak_index": peak_index,
                "n_elements": geometry.m,
                "capabilities": dict(geometry.capabilities),
            },
        )
