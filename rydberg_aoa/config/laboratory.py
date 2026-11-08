"""LAB-002: explicit inputs for a prepared-channel RF-first experiment.

These immutable declarations describe applied fields and available hardware.
They perform no ARC queries, tuning, magnetic scan or global publication.
LAB-003/AUTO-001B will consume the optical windows/preparation in selection;
constructing a request does not certify a resolved ladder or an EIT point.
"""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from ..exceptions import PhysicsConfigurationError


class OpticalEnergyConvention(str, Enum):
    """Explicit optical-energy model; neither policy includes Zeeman.

    FS_CENTROID retains the existing project's n,l,j detuning approximation.
    LOW_LEVEL_HFS uses required ARC HFS data for G/E, with unresolved-HFS
    Rydberg FS centroids. Missing data never switches to FS_CENTROID.
    """
    FS_CENTROID = 'fs_centroid'
    LOW_LEVEL_HFS = 'low_level_hfs'


def _real(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating))):
        raise PhysicsConfigurationError(f"LAB-002: {name} requires a real scalar, got {value!r}.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(f"LAB-002: {name} cannot be represented as a finite float.") from exc
    if (not math.isfinite(result) or (positive and result <= 0) or
            (nonnegative and result < 0)):
        raise PhysicsConfigurationError(f"LAB-002: invalid {name}={value!r}; no substitution.")
    return result


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PhysicsConfigurationError(f"LAB-002: {name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise PhysicsConfigurationError(f"LAB-002: {name} must be >= {minimum}.")
    return value


def _q(value, name):
    value = _integer(value, name, minimum=-1)
    if value not in (-1, 0, 1):
        raise PhysicsConfigurationError(f"LAB-002: {name} must be -1, 0 or +1.")
    return value


def _half(value, name, *, nonnegative=False):
    value = _real(value, name, nonnegative=nonnegative)
    if not math.isfinite(2 * value) or not (2 * value).is_integer():
        raise PhysicsConfigurationError(f"LAB-002: {name} must be an integer or half-integer.")
    return value


def _window(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise PhysicsConfigurationError(f"LAB-002: {name} requires an immutable (minimum, maximum) tuple.")
    lo, hi = (_real(v, name, positive=True) for v in value)
    if hi < lo:
        raise PhysicsConfigurationError(f"LAB-002: reversed {name} bounds.")
    return lo, hi


@dataclass(frozen=True)
class AppliedRF:
    """Positive carrier [Hz], nonnegative field amplitude [V/m], fixed q."""
    frequency_hz: float
    electric_field_vm: float
    polarization: int

    def __post_init__(self):
        object.__setattr__(self, 'frequency_hz', _real(self.frequency_hz, 'RF frequency [Hz]', positive=True))
        object.__setattr__(self, 'electric_field_vm', _real(self.electric_field_vm, 'RF field [V/m]', nonnegative=True))
        object.__setattr__(self, 'polarization', _q(self.polarization, 'RF q'))


@dataclass(frozen=True)
class AppliedLaser:
    """Applied wavelength [nm], power [mW], 1/e² radius [mm], cyclic offset [MHz]."""
    wavelength_nm: float
    power_mw: float
    beam_waist_mm: float
    detuning_mhz: float
    polarization: int

    def __post_init__(self):
        for name in ('wavelength_nm', 'beam_waist_mm'):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        object.__setattr__(self, 'power_mw', _real(self.power_mw, 'laser power [mW]', nonnegative=True))
        object.__setattr__(self, 'detuning_mhz', _real(self.detuning_mhz, 'laser offset [MHz]'))
        object.__setattr__(self, 'polarization', _q(self.polarization, 'laser q'))


@dataclass(frozen=True)
class PreparedHyperfineState:
    """One explicitly prepared HFS state; no mF-to-mJ interpretation."""
    n: int
    l: int
    j: float
    f: float
    mf: float

    def __post_init__(self):
        n, l = _integer(self.n, 'n', minimum=1), _integer(self.l, 'l')
        j = _half(self.j, 'j', nonnegative=True)
        f, mf = _half(self.f, 'F', nonnegative=True), _half(self.mf, 'mF')
        # Integer arithmetic avoids float overflow for an invalid huge l.
        if l >= n or 2 * j not in ((2 * l - 1, 2 * l + 1) if l else (1,)):
            raise PhysicsConfigurationError('LAB-002: invalid alkali n,l,j state.')
        if abs(mf) > f or not (f - mf).is_integer():
            raise PhysicsConfigurationError('LAB-002: invalid mF projection for F.')
        for key, value in zip(('n', 'l', 'j', 'f', 'mf'), (n, l, j, f, mf)):
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class PreparedChannel:
    """Single-channel preparation assumption on one explicitly declared axis.

    Rydberg n,l,j remain search outputs. Fixed mJ values constrain that search
    later; they do not imply experimental optical pumping or channel isolation.
    """
    ground: PreparedHyperfineState
    excited: PreparedHyperfineState
    rydberg1_mj: float
    rydberg2_mj: float
    quantization_axis: tuple

    def __post_init__(self):
        if not all(isinstance(s, PreparedHyperfineState) for s in (self.ground, self.excited)):
            raise PhysicsConfigurationError('LAB-002: preparation requires explicit HFS state declarations.')
        for name in ('rydberg1_mj', 'rydberg2_mj'):
            value = _half(getattr(self, name), name)
            if (value - math.floor(value)) != .5:
                raise PhysicsConfigurationError('LAB-002: alkali FS Rydberg mJ must be half-integer.')
            object.__setattr__(self, name, value)
        if not isinstance(self.quantization_axis, tuple) or len(self.quantization_axis) != 3:
            raise PhysicsConfigurationError('LAB-002: quantization_axis requires a three-component tuple.')
        axis = tuple(_real(v, 'quantization axis') for v in self.quantization_axis)
        if not math.isclose(math.hypot(*axis), 1., rel_tol=0., abs_tol=1e-12):
            raise PhysicsConfigurationError('LAB-002: quantization axis must be a unit vector; no normalization.')
        object.__setattr__(self, 'quantization_axis', axis)


@dataclass(frozen=True)
class LabScanBounds:
    """Inclusive n/windows, orbital cap and maximum RF mismatch [Hz].

    Wavelength windows [nm] and RF window [Hz] are required declarations of
    hardware access. They are not guessed from the applied wavelengths.
    """
    n_min: int
    n_max: int
    max_l: int
    rf_tolerance_hz: float
    probe_window_nm: tuple
    coupling_window_nm: tuple
    rf_window_hz: tuple

    def __post_init__(self):
        for key in ('n_min', 'n_max', 'max_l'):
            object.__setattr__(self, key, _integer(getattr(self, key), key, minimum=0 if key == 'max_l' else 1))
        if self.n_max < self.n_min or self.max_l > 3:
            raise PhysicsConfigurationError('LAB-002: reversed n bounds or orbital cap outside current S/P/D/F scan.')
        object.__setattr__(self, 'rf_tolerance_hz', _real(self.rf_tolerance_hz, 'RF mismatch tolerance [Hz]', nonnegative=True))
        for key in ('probe_window_nm', 'coupling_window_nm', 'rf_window_hz'):
            object.__setattr__(self, key, _window(getattr(self, key), key))


@dataclass(frozen=True)
class ExperimentalLabRequest:
    """Complete, hashable applied-input declaration for RF-first lab selection.

    No default preparation or laboratory bounds. LO defines the selection
    carrier; IF retains signal-minus-LO sign. Atomic resonance and detuning
    are computed later and never replace these applied frequencies.
    """
    probe: AppliedLaser
    coupling: AppliedLaser
    lo: AppliedRF
    signal: AppliedRF
    preparation: PreparedChannel
    bounds: LabScanBounds
    optical_energy_convention: OpticalEnergyConvention = OpticalEnergyConvention.FS_CENTROID

    def __post_init__(self):
        try:
            policy = OpticalEnergyConvention(self.optical_energy_convention)
        except (TypeError, ValueError) as exc:
            raise PhysicsConfigurationError('LAB-003: unknown optical energy convention; no default substitution.') from exc
        object.__setattr__(self, 'optical_energy_convention', policy)
        for name, kind in (('probe', AppliedLaser), ('coupling', AppliedLaser), ('lo', AppliedRF),
                           ('signal', AppliedRF), ('preparation', PreparedChannel), ('bounds', LabScanBounds)):
            if not isinstance(getattr(self, name), kind):
                raise PhysicsConfigurationError(f'LAB-002: {name} requires {kind.__name__}.')
        if self.lo.polarization != self.signal.polarization:
            raise PhysicsConfigurationError('LAB-002: LO and signal must drive the same prepared RF channel.')
        g, e = self.preparation.ground, self.preparation.excited
        if (abs(g.l - e.l) != 1 or abs(g.j - e.j) > 1 or abs(g.f - e.f) > 1
                or (g.f == e.f == 0) or e.mf - g.mf != self.probe.polarization):
            raise PhysicsConfigurationError('LAB-002: prepared probe channel violates E1 or declared q.')
        if self.preparation.rydberg2_mj - self.preparation.rydberg1_mj != self.lo.polarization:
            raise PhysicsConfigurationError('LAB-002: prepared RF projections disagree with declared q.')
        for name, value, window in (('probe', self.probe.wavelength_nm, self.bounds.probe_window_nm),
                                    ('coupling', self.coupling.wavelength_nm, self.bounds.coupling_window_nm),
                                    ('LO', self.lo.frequency_hz, self.bounds.rf_window_hz),
                                    ('signal', self.signal.frequency_hz, self.bounds.rf_window_hz)):
            if not window[0] <= value <= window[1]:
                raise PhysicsConfigurationError(f'LAB-002: applied {name}={value} is outside declared hardware window {window}.')

    @property
    def selection_carrier_hz(self):
        return self.lo.frequency_hz

    @property
    def signed_if_mhz(self):
        return (self.signal.frequency_hz - self.lo.frequency_hz) / 1e6

    def validate_for_atom(self, atom):
        """Check isotope-dependent preparation using actual atom properties.

        This validates existence of the declared angular-momentum components,
        not a nonzero ARC DME, optical pumping or experimental isolation.
        """
        if atom is None:
            raise PhysicsConfigurationError('LAB-002: preparation validation requires the actual atom.')
        nuclear_spin = _half(getattr(atom, 'I', None), 'atomic I', nonnegative=True)
        ground_n = _integer(getattr(atom, 'groundStateN', None), 'atomic groundStateN', minimum=1)
        g, e = self.preparation.ground, self.preparation.excited
        if (g.n, g.l, g.j) != (ground_n, 0, .5):
            raise PhysicsConfigurationError('LAB-002: prepared ground state disagrees with the actual alkali atom.')
        for state in (g, e):
            if (not abs(nuclear_spin - state.j) <= state.f <= nuclear_spin + state.j
                    or not (nuclear_spin + state.j - state.f).is_integer()):
                raise PhysicsConfigurationError('LAB-002: prepared F is incompatible with atomic I and j.')
        mi = e.mf + self.coupling.polarization - self.preparation.rydberg1_mj
        if abs(mi) > nuclear_spin or not (nuclear_spin - mi).is_integer():
            raise PhysicsConfigurationError('LAB-002: prepared coupling channel has no allowed nuclear projection mI.')
        if max(abs(self.preparation.rydberg1_mj), abs(self.preparation.rydberg2_mj)) > self.bounds.max_l + .5:
            raise PhysicsConfigurationError('LAB-002: declared mJ cannot fit any j within the orbital scan cap.')
        return self
