"""
Data Classes and Types for Level Optimization.

Contains data structures used throughout the level finding process.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..arc.types import validate_magnetic_quantum_number
from ..exceptions import PhysicsConfigurationError


@dataclass
class OptimizationCriteria:
    """Criteria for selecting optimal atomic levels."""

    max_dipole_strength: float = 1e-25  # Minimum dipole for strong transitions
    max_detuning_tolerance: float = 5.0  # MHz, how far from resonance we tolerate
    prioritize_electric_dipole: bool = True  # Prefer Δl=±1 transitions
    minimize_linewidth: bool = True  # Prefer narrow transitions
    prefer_low_n: bool = True  # Prefer lower principal quantum numbers


@dataclass
class NarrowedRanges:
    """Narrowed search ranges from analytical predictions."""

    probe_n_range: Tuple[int, int] = (35, 45)
    coupling_n1_range: Tuple[int, int] = (45, 55)
    coupling_n2_range: Tuple[int, int] = (46, 56)
    rf_n_range: Tuple[int, int] = (45, 65)
    analytical_predictions: Optional[Dict[str, Any]] = None


@dataclass
class LevelTransitionCandidate:
    """Represents a candidate atomic level transition."""

    wavelength_nm: float
    wavelength_diff_nm: float
    frequency_hz: float = 0.0
    frequency_diff_hz: float = 0.0
    dipole_moment: float = 0.0
    fig_of_merit: float = 0.0
    n_from: int = 0
    l_from: int = 0
    j_from: float = 0.0
    n_to: int = 0
    l_to: int = 0
    j_to: float = 0.0

    @property
    def label(self) -> str:
        """Generate human-readable label for the transition."""
        states = ['S', 'P', 'D', 'F']
        return f"{self.n_from}{states[self.l_from]}{int(2*self.j_from+1)//2} -> {self.n_to}{states[self.l_to]}{int(2*self.j_to+1)//2}"


@dataclass
class AtomicLevel:
    """Represents an atomic energy level."""

    n: int = 0
    l: int = 0
    j: float = 0.0
    f: Optional[float] = None
    mj: Optional[float] = None  # Add mj field
    mf: Optional[int] = None
    energy: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        """Set the default label and VALIDATE the quantum numbers.

        QN-001A (removed defect): the historical implementation forced
        ``mj = mf`` (else ``0.0``). These level objects are serialised into the
        atomic-level cache (``atomic_config._ATOMIC_LEVELS_CACHE``) and
        re-hydrated as :class:`rydberg_aoa.arc.types.AtomicLevel`, where a
        fabricated ``mj`` is rejected for every half-integer ``j`` (e.g. the
        26D5/2 Rydberg level, j = 5/2: mJ = 0 is not a sublevel). ``mj`` and
        ``mf`` are therefore left ``None`` unless explicitly configured, and
        validated when present — never manufactured from one another.
        """
        if not self.label and self.n > 0:
            states = ['S', 'P', 'D', 'F']
            self.label = f"{self.n}{states[self.l % 4]}{int(2*self.j+1)//2}"
        if self.mj is not None:
            validate_magnetic_quantum_number(self.mj, self.j, "mj", "j")
        if self.mf is not None:
            if self.f is None:
                raise PhysicsConfigurationError(
                    f"mf={self.mf!r} was supplied for level "
                    f"{self.label or f'{self.n}{self.l}{self.j}'} without f: "
                    "mf is the projection of F and cannot be validated "
                    "without it. QN-001A: no implicit f, no implicit mj."
                )
            validate_magnetic_quantum_number(self.mf, self.f, "mf", "f")

@dataclass
class OptimizationResult:
    """Result of level optimization."""

    levels: Dict[str, Any]
    transitions: Dict[str, Any]
    optimization_metrics: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    optimization_info: Optional[Dict[str, Any]] = None
