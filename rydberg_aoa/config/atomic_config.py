"""
Atomic level definitions and ARC integration.
"""

import copy
import functools

import numpy as np
import scipy as sc
from .atom_species import get_arc_atom, LEGACY_ATOM
import sys
from typing import Optional
from .constants import US_TO_S_FACTOR, VAPOR_CELL_TEMPERATURE_K
from .laser_rf_config import LASER_PARAMS, RF_PARAMS, LO_RF_PARAMS, SIGNAL_RF_PARAMS, calculate_laser_rabi_frequency, calculate_rf_rabi_frequency, get_signal_rf_rabi_frequency
from ..arc import LaserParams, RFParams
from ..arc.rabi_frequency import calculate_rabi_frequency as calc_omega_laser, calculate_rf_rabi_frequency as calc_omega_rf
from . import parameters as params
from .atomic_resolution import (
    AtomicResolutionMode,
    describe_resolution_request,
    prescribed_cache_identity,
    resolve_resolution_request,
    resolution_request_identity,
)
from ..exceptions import (
    PhysicsCalculationError,
    PhysicsConfigurationError,
    PhysicsError,
)
ATOM = None


_ATOMIC_LEVELS_CACHE = {}
_DETUNING_CACHE = {}
_DIPOLE_MATRIX_CACHE = {}
_OPTIMIZATION_RESULTS_CACHE = {}
ATOM_SPEC = LEGACY_ATOM
_INITIALIZED_ATOM_IDENTITY = None


def _atom_identity():
    return (ATOM_SPEC, type(ATOM).__module__, type(ATOM).__qualname__, id(ATOM),
            getattr(ATOM, 'mass', None), getattr(ATOM, 'I', None), getattr(ATOM, 'groundStateN', None),
            getattr(ATOM, 'preferQuantumDefects', None), getattr(ATOM, 'cpp_numerov', None))


def require_configured_atom():
    if ATOM is None:
        raise PhysicsConfigurationError('ATOM-001: call configure_atom() or initialize_atomic_physics() first.')
    return ATOM


def configure_atom(spec):
    """Select ARC isotope only while unpublished; no ladder/field mutation."""
    global ATOM, ATOM_SPEC
    from .atom_species import arc_density
    if is_atomic_physics_initialized():
        raise PhysicsConfigurationError('ATOM-001: reset atomic physics before selecting an atom.')
    atom = get_arc_atom(spec)
    density = arc_density(atom, spec, params.VAPOR_CELL_TEMPERATURE_K)
    if ATOM is not atom or ATOM_SPEC != spec:
        for cache in (_ATOMIC_LEVELS_CACHE, _DETUNING_CACHE, _DIPOLE_MATRIX_CACHE, _OPTIMIZATION_RESULTS_CACHE):
            cache.clear()
    ATOM, ATOM_SPEC = atom, spec
    params.CALCULATED_N0 = params.SI_Constants.N0 = density

# ---------------------------------------------------------------------------
# ARCH-001 — explicit atomic-physics initialization seam
# ---------------------------------------------------------------------------
# Level optimization and every derived calculation (dipoles, detunings, decay
# rates, probe/coupling/RF Rabi frequencies) used to run at MODULE IMPORT time,
# so merely importing a physics module started the whole ARC ladder pipeline
# and could surface the QN-001A coupling boundary as an import side effect.
# They now run ONLY inside initialize_atomic_physics() — an explicit runtime
# step called by the production entry point.
#
# The compatibility globals below are None until that call SUCCEEDS. They are
# published in a single step at the end of initialization, so a failed run can
# never be observed as a partially valid runtime (ARCH-001 transactionality).
_ATOMIC_PHYSICS_INITIALIZED = False

# MODE-001: identity of the atomic-resolution request that produced the
# published runtime.  A second ``initialize_atomic_physics()`` call must not
# silently hand back EXPERIMENTAL results when the caller now asks for a
# PRESCRIBED ladder (or a different declared ladder).  ``None`` until the
# first successful initialization.
_INITIALIZED_RESOLUTION_IDENTITY = None
# MODE-001: the exact request object that produced the published runtime.
# Plain no-argument readers (``get_cached_dipole_matrix()``,
# ``get_cached_detunings()``, ``get_arc_calculated_rf_dipole()``) resolve
# their cache key from this request rather than the live switch, so they can
# never self-heal from the wrong mode's branch after an explicit
# ``resolution_request=`` initialization.  ``None`` until the first success;
# cleared together with the identity by ``_clear_published_atomic_state()``.
_INITIALIZED_REQUEST = None
# RF-DIR-001B: the resolved energy orientation of the |3> <-> |4> (R1 <-> R2)
# RF edge, published together with the rest of the runtime by
# initialize_atomic_physics() and read through the guarded
# get_rf_orientation().  ``None`` until the first successful initialization;
# cleared by _clear_published_atomic_state().
_RF_ORIENTATION = None

ATOMIC_LEVELS = None
ARC_RF_DIPOLE_EA0 = None
GAMMA_21 = None
GAMMA_32 = None
GAMMA_43 = None
OMEGA_P = None
OMEGA_C = None
OMEGA_RF_BASE = None
OMEGA_SIGNAL_RF_BASE = None
DELTA_P = None
DELTA_C = None
DELTA_RF = None


def is_atomic_physics_initialized() -> bool:
    """ARCH-001: True only after a fully successful explicit initialization."""
    return _ATOMIC_PHYSICS_INITIALIZED


def require_atomic_physics_initialized() -> None:
    """ARCH-001 guard for every reader of a resolved atomic quantity.

    Never initializes implicitly (no auto-initialize-on-read) and never
    substitutes a default value: reading atomic physics before the explicit
    initialization is a configuration error, per the F05 fail-loudly contract.

    Raises:
        PhysicsConfigurationError: if initialize_atomic_physics() has not
            completed successfully yet.
    """
    if not _ATOMIC_PHYSICS_INITIALIZED:
        raise PhysicsConfigurationError(
            "Atomic physics has not been initialized. "
            "Call initialize_atomic_physics() before requesting resolved "
            "atomic quantities. ARCH-001: no implicit initialization and no "
            "default substitution (F05)."
        )
    from .laboratory import ExperimentalLabRequest
    if _INITIALIZED_ATOM_IDENTITY is not None and _INITIALIZED_ATOM_IDENTITY != _atom_identity():
        raise PhysicsConfigurationError('ATOM-001: selected atom identity changed after publication; explicitly reset.')
    if isinstance(_INITIALIZED_REQUEST, ExperimentalLabRequest):
        from .laboratory_runtime import require_current_laboratory_runtime
        require_current_laboratory_runtime(sys.modules[__name__])


def _clear_published_atomic_state() -> None:
    """Un-publish every compatibility global and clear the initialized flag.

    Used before a ``force=True`` recomputation so that a failed forced run can
    never leave values from the PREVIOUS successful state advertised next to
    caches that the retried pipeline has already rewritten.
    """
    global _ATOMIC_PHYSICS_INITIALIZED
    global _INITIALIZED_ATOM_IDENTITY
    global _INITIALIZED_RESOLUTION_IDENTITY
    global _INITIALIZED_REQUEST
    global ATOMIC_LEVELS, ARC_RF_DIPOLE_EA0
    global GAMMA_21, GAMMA_32, GAMMA_43
    global OMEGA_P, OMEGA_C, OMEGA_RF_BASE, OMEGA_SIGNAL_RF_BASE
    global DELTA_P, DELTA_C, DELTA_RF
    global _RF_ORIENTATION

    _RF_ORIENTATION = None
    ATOMIC_LEVELS = None
    ARC_RF_DIPOLE_EA0 = None
    GAMMA_21 = GAMMA_32 = GAMMA_43 = None
    OMEGA_P = OMEGA_C = OMEGA_RF_BASE = OMEGA_SIGNAL_RF_BASE = None
    DELTA_P = DELTA_C = DELTA_RF = None
    _ATOMIC_PHYSICS_INITIALIZED = False
    _INITIALIZED_ATOM_IDENTITY = None
    _INITIALIZED_RESOLUTION_IDENTITY = None
    _INITIALIZED_REQUEST = None
    _RF_ORIENTATION = None

def _generate_cache_key(probe_params, coupling_params, rf_params):
    """Generate a unique cache key based on key parameters.

    N-1 fix: λ_p (probe) and λ_c (coupling) are both part of the key. The previous
    implementation merged ``{**probe, **coupling}`` whose shared ``wavelength_nm``
    field silently kept only the coupling value, so a probe-wavelength change never
    invalidated the caches (stale levels/dipoles were returned without any error).

    Args:
        probe_params (dict): Probe laser parameters (wavelength_nm, ...).
        coupling_params (dict): Coupling laser parameters (wavelength_nm, power_mw, ...).
        rf_params (dict): RF parameters (frequency_hz, electric_field_vm).

    Returns:
        tuple: Cache key (λ_p [nm], λ_c [nm], f_RF [Hz], P_c [mW], E_RF [V/m]).
    """
    return (
        float(probe_params['wavelength_nm']),             # λ_p — probe transition wavelength [nm]
        float(coupling_params['wavelength_nm']),          # λ_c — coupling transition wavelength [nm]
        float(rf_params['frequency_hz']),                 # f_RF — RF field frequency [Hz]
        round(float(coupling_params['power_mw']), 1),     # P_c — coupling laser power [mW]
        round(float(rf_params['electric_field_vm']), 2),  # E_RF — RF field amplitude [V/m]
    )


# ---------------------------------------------------------------------------
# MODE-001 — atomic-resolution mode boundary
# ---------------------------------------------------------------------------
def _initialized_resolution_request():
    """The request that produced the published runtime (MODE-001).

    Post-initialization readers that must stay inside the mode which was
    actually resolved — rather than inside whatever the live switch says
    afterward — derive their cache key from this request.

    Returns:
        tuple ``(request, from_switch)`` where ``request`` is the stored
        initialized request.  When no initialization has produced one yet
        (``_INITIALIZED_REQUEST is None``) the configured switch is consulted
        instead (legacy behaviour, still correct for in-seam readers called
        with an explicit key) and ``from_switch`` is True.
    """
    if _INITIALIZED_REQUEST is not None:
        return _INITIALIZED_REQUEST, False
    return resolve_resolution_request(), True


def _resolution_cache_key(resolution_request=None) -> tuple:
    """Cache key of the CURRENT atomic-resolution request (MODE-001).

    EXPERIMENTAL requests produce the historical five-field key *bit for bit*,
    so no pre-existing cache entry moves and no production behaviour changes
    when the switch is left at its default.

    A PRESCRIBED request appends :func:`prescribed_cache_identity` — the four
    declared states (including both Rydberg ``mJ``), the probe/coupling
    wavelengths and polarisations and the LO/signal RF frequency, field and
    polarisation.  The tuple ARITY therefore differs, so a prescribed entry
    can never collide with an experimental one, and two different declared
    ladders can never share an entry.

    Args:
        resolution_request: anything :func:`resolve_resolution_request`
            accepts; ``None`` consults the configured switch.

    Returns:
        tuple: 5-field experimental key or the extended prescribed key.

    Raises:
        PhysicsConfigurationError: an ambiguous/invalid configured mode, or a
            prescribed request whose declared fields are incomplete.
    """
    from .laboratory import ExperimentalLabRequest
    if isinstance(resolution_request, ExperimentalLabRequest):
        from .laboratory_runtime import laboratory_identity
        return laboratory_identity(ATOM, resolution_request)
    request = resolve_resolution_request(resolution_request)
    base = _generate_cache_key(LASER_PARAMS['probe'], LASER_PARAMS['coupling'],
                               RF_PARAMS)
    # Preserve legacy Rb85 key format; all other isotope/fraction/object keys
    # occupy a distinct namespace. Selection also clears every pipeline cache.
    if (ATOM_SPEC != LEGACY_ATOM or type(ATOM).__name__ in ('Rubidium87', 'Caesium')
            or (type(ATOM).__name__ == 'Rubidium85' and ATOM is not get_arc_atom(LEGACY_ATOM))):
        base = ('ATOM-001', _atom_identity()) + base
    if request.mode is AtomicResolutionMode.EXPERIMENTAL:
        return base
    return base + prescribed_cache_identity(
        request, LASER_PARAMS['probe'], LASER_PARAMS['coupling'],
        RF_PARAMS, SIGNAL_RF_PARAMS)


def _level_as_dict(level) -> dict:
    """Public ``AtomicLevel`` fields as a plain dict (same shape the
    Experimental branch caches, so every downstream reader sees one shape)."""
    return {k: v for k, v in level.__dict__.items() if not k.startswith('_')}


def _configured_q(field_params, name: str) -> int:
    """Polarisation q of a configured field, validated in {-1, 0, +1}."""
    q_raw = (field_params.get('polarization')
             if isinstance(field_params, dict)
             else getattr(field_params, 'polarization', None))
    try:
        q = int(q_raw)
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError(
            f"the configured {name} polarisation is missing or is not an "
            f"integer ({q_raw!r}); q must be one of -1, 0, +1. MODE-001: the "
            "declared channel needs its declared polarisation."
        ) from exc
    if float(q_raw) != float(q) or q not in (-1, 0, 1):
        raise PhysicsConfigurationError(
            f"the configured {name} polarisation must be one of -1, 0, +1 "
            f"(got {q_raw!r}). MODE-001: no q fallback."
        )
    return q


def _declared_dipole_matrix(atom, level_objects: dict) -> dict:
    """Dipole matrix of the THREE DECLARED channels — no q scan, no mJ scan.

    MODE-001 §10/§11: the Experimental branch builds this cache through
    ``arc/dipole_matrix.calculate_dipole_matrix`` ->
    ``calculate_dipole_matrix_element``, whose hyperfine branch sweeps
    ``q in {-1, 0, +1}`` and assumes the destination ``mF2 = 0``, and whose
    Rydberg branch takes the maximum over magnetic sublevels.  All three are
    exactly the searches PRESCRIBED mode forbids.

    Instead each leg is evaluated by the ONE QN primitive that owns it, on the
    exact channel the user declared:

        probe      ground   -> excited    HFS -> HFS   QN-001A.1
        coupling   excited  -> rydberg1   HFS -> FS    QN-001B
        rf         rydberg1 -> rydberg2   FS  -> FS    QN-001C

    Each primitive returns the SIGNED element in ``e*a0`` from exactly one ARC
    call; the magnitude is converted with the project-wide ``EA0_TO_CM``.

    Args:
        atom: the ARC atom.
        level_objects: the four declared levels as ``AtomicLevel`` objects.

    Returns:
        dict: ``{'probe'|'coupling'|'rf': {'dipole_si', 'transition',
        'level1', 'level2'}}`` — the same structure the Experimental branch
        publishes, so ``get_cached_dipole_matrix()`` consumers are unchanged.

    Raises:
        PhysicsConfigurationError / PhysicsCalculationError: any primitive
            failure propagates (F05 — never replaced by a scanned value).
    """
    from ..arc.dipole import (
        EA0_TO_CM,
        calculate_fs_to_fs_dipole_matrix_element,
        calculate_hfs_to_fs_dipole_matrix_element,
        calculate_hfs_to_hfs_dipole_matrix_element,
    )

    g, e = level_objects['ground'], level_objects['excited']
    r1, r2 = level_objects['rydberg1'], level_objects['rydberg2']

    declared = (
        ('probe', g, e, LASER_PARAMS['probe'],
         calculate_hfs_to_hfs_dipole_matrix_element),
        ('coupling', e, r1, LASER_PARAMS['coupling'],
         calculate_hfs_to_fs_dipole_matrix_element),
        # The LO RF field defines the driven R1->R2 channel; SIGNAL_RF_PARAMS
        # drives the SAME channel and is checked for consistency separately.
        ('rf', r1, r2, RF_PARAMS, calculate_fs_to_fs_dipole_matrix_element),
    )

    matrix = {}
    for name, src, dst, field_params, primitive in declared:
        q = _configured_q(field_params, name)
        signed_ea0 = primitive(atom, src, dst, q)   # ONE declared channel
        try:
            magnitude_si = abs(float(signed_ea0)) * EA0_TO_CM
        except (TypeError, ValueError) as exc:
            raise PhysicsCalculationError(
                f"the declared {name} dipole is not numeric ({signed_ea0!r}). "
                "MODE-001: no fallback is permitted."
            ) from exc
        if not np.isfinite(magnitude_si) or magnitude_si <= 0.0:
            raise PhysicsCalculationError(
                f"the declared {name} channel {src.label} -> {dst.label} "
                f"(q={q}) produced an unusable dipole {signed_ea0!r} e*a0 -> "
                f"{magnitude_si!r} C*m. MODE-001: the declared channel is "
                "never replaced by a scanned one (F05)."
            )
        matrix[name] = {
            'dipole_si': magnitude_si,
            'transition': f"{src.label} -> {dst.label}",
            'level1': src,
            'level2': dst,
            # SIGNED |d| in e*a0, bit-identical to what the primitive returned,
            # so ARC_RF_DIPOLE_EA0 is taken from the single ARC call instead of
            # being re-derived through a C*m round trip (the sign stays
            # available for any phase-sensitive consumer).
            'dipole_ea0': float(signed_ea0),
        }
    return matrix


def _require_lo_signal_rf_polarisation_consistency() -> None:
    """MODE-001 §19/§20: LO and signal RF drive ONE R1 -> R2 coherence.

    The heterodyne model beats a local-oscillator field against a signal field
    on the SAME Rydberg transition, so both must address the SAME magnetic
    sublevel ``mJ2 = mJ1 + q``.  QN-001C already computes each field's Rabi
    frequency from its OWN polarisation (never a shared hard-coded one), which
    means a disagreement would silently drive two different channels while the
    model still treated them as one beat note.

    Inconsistency therefore fails loudly instead of being reconciled: neither
    value is rewritten, guessed or scanned.

    Raises:
        PhysicsConfigurationError: when LO and signal declare different q.
    """
    q_lo = _configured_q(RF_PARAMS, 'LO RF')
    q_signal = _configured_q(SIGNAL_RF_PARAMS, 'signal RF')
    if q_lo != q_signal:
        raise PhysicsConfigurationError(
            "the LO RF field (polarization="
            f"{RF_PARAMS.get('polarization')}) and the signal RF field "
            f"(polarization={SIGNAL_RF_PARAMS.get('polarization')}) drive the "
            "same rydberg1 -> rydberg2 transition with DIFFERENT magnetic "
            f"channels (q_LO={q_lo} vs q_signal={q_signal}), so they would "
            "not beat on one coherence. MODE-001 §19/§20: the declared values "
            "are reported, never reconciled, defaulted or scanned."
        )


def _initialize_prescribed_levels(resolution_request) -> dict:
    """MODE-001 PRESCRIBED step 1: materialise the DECLARED ladder.

    Zero search — no ``SingleStageLevelOptimizer``, no level finder, no
    max-over-mJ dipole sweep, no q scan and no ``arc/dipole_matrix`` legacy RF
    cache.  The four declared states are copied verbatim into the canonical
    ``ATOMIC_LEVELS`` shape and every derived quantity is computed by ARC for
    exactly that channel.

    Wavelength immutability (MODE-001 §13)
        The Experimental branch rewrites
        ``LASER_PARAMS['coupling']['wavelength_nm']`` to whichever ARC
        transition the optimizer picked.  This branch never writes to
        ``LASER_PARAMS`` / ``RF_PARAMS``: the applied wavelengths are *inputs*,
        so a detuning far from resonance is a legitimate prescribed result
        rather than something to be repaired.

    Detuning convention (MODE-001 §14)
        ``calculate_all_detunings`` already expresses the project convention
        ``Delta = omega_applied - omega_atomic`` in ``rad/us`` and reads its
        applied values from the PASSED parameter dicts, so the same function is
        reused unchanged — no second detuning implementation is introduced.

    Args:
        resolution_request: a validated PRESCRIBED request.

    Returns:
        dict: ``{'ground','excited','rydberg1','rydberg2'}`` public field dicts.

    Raises:
        PhysicsConfigurationError: not a PRESCRIBED request, or inconsistent
            LO / signal RF polarisation.
        PhysicsCalculationError: detuning or dipole evaluation failed (F05).
    """
    request = resolve_resolution_request(resolution_request)
    if not request.is_prescribed:
        raise PhysicsConfigurationError(
            "_initialize_prescribed_levels requires a PRESCRIBED request; "
            f"got {request.mode!r}. MODE-001: the two branches are never "
            "interchanged."
        )

    spec = request.prescribed
    levels = {role: _level_as_dict(level)
              for role, level in spec.levels().items()}
    if type(ATOM).__name__ in ('Rubidium85', 'Rubidium87', 'Caesium'):
        from .atom_species import validate_declared_ladder
        validate_declared_ladder(ATOM, levels, _configured_q(LASER_PARAMS['coupling'], 'coupling'))
    cache_key = _resolution_cache_key(request)

    if cache_key in _ATOMIC_LEVELS_CACHE:
        # Same declared ladder + same field parameters: nothing to recompute.
        return _ATOMIC_LEVELS_CACHE[cache_key]

    # The two RF fields must address one channel before anything is computed.
    _require_lo_signal_rf_polarisation_consistency()

    print("=== PRESCRIBED atomic resolution (MODE-001) ===")
    print(describe_resolution_request(request))
    print(f"Applied probe wavelength: "
          f"{LASER_PARAMS['probe']['wavelength_nm']} nm")
    print(f"Applied coupling wavelength: "
          f"{LASER_PARAMS['coupling']['wavelength_nm']} nm "
          "(prescribed: never rewritten)")
    print(f"Applied RF frequency: {RF_PARAMS['frequency_hz']/1e9:.3f} GHz")
    print()

    # ---- detunings: Delta = omega_applied - omega_atomic [rad/us] ----------
    try:
        from ..physics.detuning import calculate_all_detunings
        from ..arc.types import AtomicLevel

        level_objects = {role: AtomicLevel(**fields)
                         for role, fields in levels.items()}
        delta_p, delta_c, delta_rf = calculate_all_detunings(
            ATOM, level_objects, LASER_PARAMS['probe'],
            LASER_PARAMS['coupling'], RF_PARAMS)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(
            "Prescribed detuning calculation failed for the declared ladder "
            f"(probe={LASER_PARAMS['probe'].get('wavelength_nm')} nm, "
            f"coupling={LASER_PARAMS['coupling'].get('wavelength_nm')} nm, "
            f"f_RF={RF_PARAMS.get('frequency_hz')} Hz): "
            f"{type(exc).__name__}: {exc}. "
            "MODE-001/F05: no resonance substitution — a far-detuned applied "
            "wavelength is a valid prescribed input, not an error to repair."
        ) from exc

    _DETUNING_CACHE[cache_key] = {
        'delta_p_rad_us': delta_p,
        'delta_c_rad_us': delta_c,
        'delta_rf_rad_us': delta_rf,
    }

    # ---- dipole matrix of the declared channels (no scan) ------------------
    dipole_matrix = _declared_dipole_matrix(ATOM, level_objects)
    _DIPOLE_MATRIX_CACHE[cache_key] = dipole_matrix
    for _name in ('probe', 'coupling', 'rf'):
        _entry = dipole_matrix[_name]
        print(f"  {_name.capitalize()} dipole ({_entry['transition']}): "
              f"{_entry['dipole_si']:.2e} C*m")

    # Same reader contract as the Experimental branch: the RF dipole MAGNITUDE
    # in e*a0, taken exactly from the primitive (no C*m round trip).
    _OPTIMIZATION_RESULTS_CACHE[cache_key] = {
        'dipole_rf': abs(dipole_matrix['rf']['dipole_ea0']),
    }

    _ATOMIC_LEVELS_CACHE[cache_key] = levels
    print("PRESCRIBED resolution cached (wavelengths unchanged).")
    print()
    return levels


def _initialize_atomic_levels():
    """Initialize atomic levels using single-stage optimization with ARC database integration.

    This function performs level optimization based on current laser and RF parameters,
    caching results to avoid redundant calculations when parameters remain unchanged.

    Returns:
        dict: Atomic levels dictionary containing 'ground', 'excited', 'rydberg1', 'rydberg2' levels.

    Raises:
        PhysicsCalculationError: If no valid result is returned; no retry or
            widening of the configured selection domain is permitted.
    """

    from ..arc.interface import ARCInterface
    from ..level_finder import SingleStageLevelOptimizer

    print(f"=== Optimizing atomic levels for new parameters ===")
    print(f"RF Frequency: {RF_PARAMS['frequency_hz']/1e9:.2f} GHz")
    print(f"Coupling wavelength: {LASER_PARAMS['coupling']['wavelength_nm']:.1f} nm")

    optimizer = SingleStageLevelOptimizer(arc_atom=ATOM)

    result = optimizer.optimize(
        LaserParams(**LASER_PARAMS['probe']),
        LaserParams(**LASER_PARAMS['coupling']),
        RFParams(**RF_PARAMS)
    )

    if result and 'levels' in result:
        _levels = result['levels']

        # Extract optimization metrics (AUTO-001A-R2: the FS-only selection
        # carries no dipole entry — nothing to extract, nothing fabricated)
        opt_metrics = result.get('optimization_metrics', {})
        calculated_coupling_wavelength = opt_metrics.get('coupling_wavelength_nm')

        # Use calculated coupling wavelength for dynamic parameters
        if calculated_coupling_wavelength:
            LASER_PARAMS['coupling']['wavelength_nm'] = calculated_coupling_wavelength
            print(f"Updated coupling wavelength to {calculated_coupling_wavelength:.1f} nm")
        else:
            print(f"Coupling wavelength remains: {LASER_PARAMS['coupling']['wavelength_nm']} nm (no calculation from ARC)")
        # Generate cache key from current parameters (explicit λ_p/λ_c — N-1 fix)
        cache_key = _generate_cache_key(LASER_PARAMS['probe'], LASER_PARAMS['coupling'], RF_PARAMS)
        # Check if we have cached result for these parameters
        if cache_key in _ATOMIC_LEVELS_CACHE:
            return _ATOMIC_LEVELS_CACHE[cache_key]
        
        # Store both levels and optimization info
        cached_result = {
                'ground': {k: v for k, v in _levels['ground'].__dict__.items() if not k.startswith('_')},
                'excited': {k: v for k, v in _levels['excited'].__dict__.items() if not k.startswith('_')},
                'rydberg1': {k: v for k, v in _levels['rydberg1'].__dict__.items() if not k.startswith('_')},
                'rydberg2': {k: v for k, v in _levels['rydberg2'].__dict__.items() if not k.startswith('_')},
            }

            # Calculate detunings using the new detuning calculator module
        try:
                from ..physics.detuning import calculate_all_detunings
                from ..arc.types import AtomicLevel

                # Convert dictionary levels back to AtomicLevel objects for detuning calculation
                level_objects = {
                    'ground': AtomicLevel(**cached_result['ground']),
                    'excited': AtomicLevel(**cached_result['excited']),
                    'rydberg1': AtomicLevel(**cached_result['rydberg1']),
                    'rydberg2': AtomicLevel(**cached_result['rydberg2'])
                }

                # Calculate detunings
                probe_params = LASER_PARAMS['probe']
                coupling_params = LASER_PARAMS['coupling']
                rf_params = RF_PARAMS
                delta_p, delta_c, delta_rf = calculate_all_detunings(
                    ATOM, level_objects, probe_params, coupling_params, rf_params
                )

                # Cache detunings
                _DETUNING_CACHE[cache_key] = {
                    'delta_p_rad_us': delta_p,
                    'delta_c_rad_us': delta_c,
                    'delta_rf_rad_us': delta_rf
                }

        except PhysicsError:
            raise
        except Exception as exc:
            # F05: the designated detuning calculation failed. The historical
            # behavior cached a RESONANCE (all-zero) detuning set, silently
            # redefining the physics of the model. Fail loudly instead.
            raise PhysicsCalculationError(
                "Detuning calculation failed for the optimized levels "
                f"(probe={LASER_PARAMS['probe'].get('wavelength_nm')} nm, "
                f"coupling={LASER_PARAMS['coupling'].get('wavelength_nm')} nm, "
                f"f_RF={RF_PARAMS.get('frequency_hz')} Hz): "
                f"{type(exc).__name__}: {exc}. "
                "F05: no zero/resonance detuning substitution."
            ) from exc

        # AUTO-001A-R2: EXPERIMENTAL level initialization computes NO dipole
        # matrix. The legacy ``arc/dipole_matrix`` maximum-over-mJ scan is not
        # the declared QN-001B/C channel path, so neither the FS-selection
        # result nor this consumer derives one; a dipole cache entry appears
        # only for a PRESCRIBED (fully declared) ladder. The channel-dependent
        # Rabi frequencies still fail fast at their explicit missing-mJ
        # boundary in step 4 (unchanged) instead of being fed by a scanned
        # dipole.

            # Cache optimization results separately. AUTO-001A-R2: store only
        # what was actually produced — no fabricated zero dipole (F05); the
        # readers keep their ``.get('dipole_rf', 0)`` default.
        _opt_results = {'success': True}
        if 'dipole_rf' in opt_metrics:
            _opt_results['dipole_rf'] = opt_metrics['dipole_rf']
        _OPTIMIZATION_RESULTS_CACHE[cache_key] = _opt_results

        # Cache the result
        _ATOMIC_LEVELS_CACHE[cache_key] = cached_result
        print("Atomic level optimization complete and cached.")
                # λ_p = c / f_21 — probe wavelength from ARC ground→excited transition
        # f_21 = ARC.getTransitionFrequency(ground, excited) in Hz
        f_21_hz = ATOM.getTransitionFrequency(
            _levels['ground'].n, _levels['ground'].l, _levels['ground'].j,
            _levels['excited'].n, _levels['excited'].l, _levels['excited'].j
        )
        params.SI_Constants.LAMBDA_P = sc.constants.c / f_21_hz
        print(f"ARC: Dynamic probe wavelength = {params.SI_Constants.LAMBDA_P*1e9:.4f} nm")
        return cached_result
    else:
        raise PhysicsCalculationError(
            "LAB-001: Cannot determine atomic levels inside the configured "
            "laser/RF selection domain. F05: no fallback result, tolerance "
            "widening or automatic retry.")

def get_arc_calculated_rf_dipole():
    """
    Get the ARC-calculated RF dipole moment from the optimization results.

    ARCH-001: during ``initialize_atomic_physics()`` the unguarded internal
    reader ``_resolve_arc_rf_dipole()`` is used instead, so initialization
    itself never trips the initialized-state guard.

    Returns:
        float: Dipole moment in ea₀ units (0 if not available)
    """
    require_atomic_physics_initialized()
    return _resolve_arc_rf_dipole()


def _resolve_arc_rf_dipole(cache_key: Optional[tuple] = None):
    """Unguarded ARC RF-dipole cache read (initialization-internal use).

    ``cache_key`` defaults to the key of the request that PRODUCED the runtime
    (MODE-001 ``_INITIALIZED_REQUEST``), falling back to the CURRENT
    resolution's key when no initialization has stored one yet, so a
    PRESCRIBED run reads the entry written by the prescribed branch (a key of
    different arity) instead of falling through to the absent experimental
    entry.
    """
    if cache_key is None:
        request, _from_switch = _initialized_resolution_request()
        cache_key = _resolution_cache_key(request)
    opt_results = _OPTIMIZATION_RESULTS_CACHE.get(cache_key, {})
    return opt_results.get('dipole_rf', 0)

    # Initialize levels at module load time (will use current parameters)
# ARCH-001: level optimization no longer runs at import time — it runs inside
# initialize_atomic_physics() (defined at the end of this module).
def get_cached_dipole_matrix():
    """
    Get the cached dipole matrix for the current optimized levels.

    LAB-004 explicit laboratory runtime: validate all applied fields and
    cached quantities, then return a detached copy. Missing or changed data
    raises a project error without recalculation or substitution.

    Self-healing (Issue #4): on a cache miss the matrix is recomputed from
    ATOMIC_LEVELS through the canonical ``calculate_dipole_matrix`` ARC path
    (identical to arc_level_finder.py) instead of returning None, so downstream
    consumers never crash on a stale or missing entry.

    Returns:
        dict: Dipole matrix with 'probe', 'coupling', 'rf' transitions or None if not available

    Raises:
        PhysicsConfigurationError: if atomic physics has not been explicitly
            initialized (ARCH-001: no auto-initialize-on-read).
    """
    require_atomic_physics_initialized()
    from .laboratory import ExperimentalLabRequest
    if isinstance(_INITIALIZED_REQUEST, ExperimentalLabRequest):
        from .laboratory_runtime import read_laboratory_dipoles
        return read_laboratory_dipoles(sys.modules[__name__])
    # MODE-001: prefer the request that PRODUCED the runtime over the live
    # configured switch (they differ after `resolution_request=` inits).
    request, _from_switch = _initialized_resolution_request()
    cache_key = _resolution_cache_key(request)

    if cache_key in _DIPOLE_MATRIX_CACHE:
        return _DIPOLE_MATRIX_CACHE[cache_key]

    # Lazy self-heal: rebuild mu_ij from ATOMIC_LEVELS.
    #
    # MODE-001 §11: a PRESCRIBED resolution must NEVER reach
    # ``calculate_dipole_matrix`` — that legacy path sweeps q and takes the
    # maximum over magnetic sublevels, which is precisely the search the
    # declared mode forbids.  It is therefore rebuilt from the declared
    # channels through the QN-001A.1/B/C primitives instead.
    if ATOM is not None and isinstance(ATOMIC_LEVELS, dict) and ATOMIC_LEVELS:
        try:
            from ..arc.types import AtomicLevel
            level_objects = {
                name: AtomicLevel(**lvl) if isinstance(lvl, dict) else lvl
                for name, lvl in ATOMIC_LEVELS.items()
            }
            if request.is_prescribed:
                matrix = _declared_dipole_matrix(ATOM, level_objects)
            else:
                from ..arc.dipole_matrix import calculate_dipole_matrix
                matrix = calculate_dipole_matrix(level_objects, ATOM)
            if matrix:
                _DIPOLE_MATRIX_CACHE[cache_key] = matrix
                return matrix
        except PhysicsError:
            raise
        except Exception as e:
            print(f"Warning: Dipole matrix self-heal failed: {e}")

    return None



# ARCH-001: ARC_RF_DIPOLE_EA0 is published by initialize_atomic_physics()
# (it depends on the optimization results, so it cannot exist before init).

# Calculate atomic density from vapor pressure using ARC
def calculate_atomic_density(atom, temperature_k):
    """
    Calculate rubidium vapor density at given temperature using ARC.

    Args:
        atom: ARC Rubidium85 atom object
        temperature_k: Temperature in Kelvin

    Returns:
        float: Atomic number density in m^-3
    """
    if atom is None:
        raise PhysicsConfigurationError(
            "ARC atom is unavailable; the rubidium vapor number density "
            f"at T={temperature_k} K cannot be calculated. "
            "F05: no default-density substitution."
        )

    try:
        density = atom.getNumberDensity(temperature_k)
    except Exception as exc:
        raise PhysicsCalculationError(
            f"ARC getNumberDensity failed at T={temperature_k} K: "
            f"{type(exc).__name__}: {exc}. "
            "F05: no default-density substitution."
        ) from exc

    if density is None or not np.isfinite(density) or density <= 0:
        raise PhysicsCalculationError(
            f"ARC returned an unusable vapor number density {density!r} m^-3 "
            f"at T={temperature_k} K. F05: no default-density substitution."
        )
    return density


# ---------------------------------------------------------------------------
# ARCH-001TX — transactional boundary around initialize_atomic_physics()
# ---------------------------------------------------------------------------
def _snapshot_config_mapping(mapping: dict) -> dict:
    """ARCH-001TX-R1: snapshot a configuration mapping without losing the
    identity of its nested dictionaries.

    Two records are kept per mapping:

    * ``objects`` — the mapping's ORIGINAL member objects at snapshot time,
      kept by reference, so a member that initialization later replaces or
      removes can be reinstalled under its original identity;
    * ``contents`` — a deep copy of the members' values at snapshot time,
      so an in-place rewrite (e.g. the step-1
      ``LASER_PARAMS['coupling']['wavelength_nm'] = ...``) can be undone
      inside the ORIGINAL member object.

    The actual configuration structure is depth-2 with scalar leaves
    (``LASER_PARAMS['probe' | 'coupling']`` hold only floats/ints; the RF
    dictionaries and ``LEVEL_SELECTION_TOLERANCES`` are flat), so restoring
    ``contents`` into the original members is a complete content restore.
    """
    return {"objects": dict(mapping), "contents": copy.deepcopy(mapping)}


def _rollback_config_mapping(mapping: dict, snapshot: dict) -> None:
    """ARCH-001TX-R1: restore a mapping snapshotted by
    :func:`_snapshot_config_mapping`.

    Keys added during the transaction are removed.  Every snapshotted key is
    restored to its ORIGINAL member object (undoing any replacement or
    removal), and that object's own contents are restored IN PLACE — so a
    reference retained before the run (``probe_ref = LASER_PARAMS['probe']``)
    observes both the original identity and the original values afterwards.
    """
    objects = snapshot["objects"]
    contents = snapshot["contents"]
    for key in [key for key in mapping if key not in contents]:
        del mapping[key]
    for key, saved in contents.items():
        original = objects[key]
        if isinstance(original, dict):
            mapping[key] = original
            for member in [member for member in original
                           if member not in saved]:
                del original[member]
            original.update(saved)
        else:
            mapping[key] = saved



def _snapshot_pipeline_transaction() -> dict:
    """ARCH-001TX: capture every mutable state a pipeline run can rewrite.

    ARCH-001's publish-on-success contract already protects the compatibility
    globals; this snapshot covers the CONFIGURATION state that steps 1-4
    rewrite *before* any later failure can occur:

    * ``LASER_PARAMS`` / ``RF_PARAMS`` / ``LO_RF_PARAMS`` /
      ``SIGNAL_RF_PARAMS`` — the Experimental branch rewrites
      ``LASER_PARAMS['coupling']['wavelength_nm']`` while the prescribed
      branch must keep treating the applied fields as read-only inputs;
      the snapshot keeps the ORIGINAL nested member objects (ARCH-001TX-R1:
      ``LASER_PARAMS['probe' | 'coupling']``), so retained references observe
      both the restored identity and the restored values;
    * ``params.SI_Constants.LAMBDA_P`` — rewritten from ``c / f_21``;
    * ``params.LEVEL_SELECTION_TOLERANCES`` — captured defensively; LAB-001
      removed the former 10x widening and automatic retry;
    * ``params.OMEGA_S_BASE`` / ``params.OMEGA_LO_BASE`` — rewritten near
      the end of step 4, while step 5 can still fail;
    * the four pipeline caches — step 1 writes level/detuning/optimization
      entries (the prescribed branch also writes a dipole entry) before
      step 4 can raise.

    ARC's internal caches are out of scope: they are keyed by quantum
    numbers and are not part of this module's published state.

    Returns:
        dict: opaque rollback snapshot for :func:`_rollback_pipeline_transaction`.
    """
    return {
        "atom": ATOM,
        "atom_spec": ATOM_SPEC,
        "number_density": params.CALCULATED_N0,
        "si_number_density": params.SI_Constants.N0,
        "laser_params": _snapshot_config_mapping(LASER_PARAMS),
        "rf_params": _snapshot_config_mapping(RF_PARAMS),
        "lo_rf_params": _snapshot_config_mapping(LO_RF_PARAMS),
        "signal_rf_params": _snapshot_config_mapping(SIGNAL_RF_PARAMS),
        "lambda_p": params.SI_Constants.LAMBDA_P,
        "level_selection_tolerances": _snapshot_config_mapping(
            params.LEVEL_SELECTION_TOLERANCES),
        "omega_s_base": params.OMEGA_S_BASE,
        "omega_lo_base": params.OMEGA_LO_BASE,
        "atomic_levels_cache": dict(_ATOMIC_LEVELS_CACHE),
        "detuning_cache": dict(_DETUNING_CACHE),
        "dipole_matrix_cache": dict(_DIPOLE_MATRIX_CACHE),
        "optimization_results_cache": dict(_OPTIMIZATION_RESULTS_CACHE),
    }


def _rollback_pipeline_transaction(snapshot: dict) -> None:
    """ARCH-001TX: restore a :func:`_snapshot_pipeline_transaction` snapshot.

    Every configuration mapping is restored so that ALL existing references
    observe the restored state — both the root dictionaries re-exported from
    ``laser_rf_config`` and the ORIGINAL nested member dictionaries such as
    ``LASER_PARAMS['probe' | 'coupling']`` (ARCH-001TX-R1 reinstalls the
    entry member objects and restores their contents in place, instead of
    installing deep-copied replacements that would orphan references held by
    earlier readers).

    The four pipeline caches keep the plain shallow ``clear()`` + ``update()``
    restore: every pipeline writer inserts whole entries
    (``_CACHE[cache_key] = value``) and no production code mutates an
    existing entry value in place, so a shallow snapshot — which keeps the
    original value objects by reference — removes exactly the entries the
    failed run added.  (A future in-place write to a pre-existing entry
    value would fall outside this coverage; none exists today.)

    The published globals and the initialized flag are deliberately NOT
    touched here.  On a ``force=True`` attempt they were cleared on purpose
    by ``_clear_published_atomic_state()`` and must stay unpublished; on a
    first attempt they were never written (ARCH-001 publish-on-success).
    """
    global ATOM, ATOM_SPEC
    ATOM, ATOM_SPEC = snapshot['atom'], snapshot['atom_spec']
    params.CALCULATED_N0 = snapshot['number_density']
    params.SI_Constants.N0 = snapshot['si_number_density']
    _rollback_config_mapping(LASER_PARAMS, snapshot["laser_params"])
    _rollback_config_mapping(RF_PARAMS, snapshot["rf_params"])
    _rollback_config_mapping(LO_RF_PARAMS, snapshot["lo_rf_params"])
    _rollback_config_mapping(SIGNAL_RF_PARAMS, snapshot["signal_rf_params"])
    params.SI_Constants.LAMBDA_P = snapshot["lambda_p"]
    _rollback_config_mapping(params.LEVEL_SELECTION_TOLERANCES,
                             snapshot["level_selection_tolerances"])
    params.OMEGA_S_BASE = snapshot["omega_s_base"]
    params.OMEGA_LO_BASE = snapshot["omega_lo_base"]
    _ATOMIC_LEVELS_CACHE.clear()
    _ATOMIC_LEVELS_CACHE.update(snapshot["atomic_levels_cache"])
    _DETUNING_CACHE.clear()
    _DETUNING_CACHE.update(snapshot["detuning_cache"])
    _DIPOLE_MATRIX_CACHE.clear()
    _DIPOLE_MATRIX_CACHE.update(snapshot["dipole_matrix_cache"])
    _OPTIMIZATION_RESULTS_CACHE.clear()
    _OPTIMIZATION_RESULTS_CACHE.update(snapshot["optimization_results_cache"])


def _transactional_atomic_initialization(func):
    """ARCH-001TX: wrap the seam in a configuration-state transaction.

    ``initialize_atomic_physics`` keeps its publish-on-success structure
    untouched; this wrapper snapshots the mutable configuration state on
    entry and — if and only if — the pipeline raises, rolls that state back
    before re-raising the ORIGINAL exception unchanged (type, message and
    ``__cause__`` chain preserved — F05: no value substitution, no error
    translation).  On success the snapshot is discarded and every value
    computed by the pipeline survives exactly as before, so the successful
    EXPERIMENTAL/PRESCRIBED behaviour is unchanged.

    ``functools.wraps`` keeps the public signature, docstring and
    ``inspect.signature`` view of the seam identical to the unwrapped
    function.
    """
    @functools.wraps(func)
    def _wrapper(*, force: bool = False, resolution_request=None) -> bool:
        transaction = _snapshot_pipeline_transaction()
        try:
            return func(force=force, resolution_request=resolution_request)
        except BaseException:
            _rollback_pipeline_transaction(transaction)
            raise
    return _wrapper


def _initialize_laboratory_physics(request, *, force=False):
    """LAB-004 branch of the canonical seam; outer decorator owns rollback."""
    global _ATOMIC_PHYSICS_INITIALIZED, _INITIALIZED_RESOLUTION_IDENTITY, _INITIALIZED_REQUEST
    global _INITIALIZED_ATOM_IDENTITY
    global ATOMIC_LEVELS, ARC_RF_DIPOLE_EA0, GAMMA_21, GAMMA_32, GAMMA_43
    global OMEGA_P, OMEGA_C, OMEGA_RF_BASE, OMEGA_SIGNAL_RF_BASE
    global DELTA_P, DELTA_C, DELTA_RF, _RF_ORIENTATION
    from .laboratory_runtime import (laboratory_identity, require_matching_applied_fields,
                                     build_laboratory_runtime, populate_laboratory_caches)
    module = sys.modules[__name__]
    require_matching_applied_fields(module, request)
    if ATOM is None:
        configure_atom(ATOM_SPEC)
    identity = laboratory_identity(ATOM, request)
    if _ATOMIC_PHYSICS_INITIALIZED and not force:
        if identity != _INITIALIZED_RESOLUTION_IDENTITY:
            raise PhysicsConfigurationError('LAB-004: runtime was initialized for a different request/atom; pass force=True explicitly.')
        require_atomic_physics_initialized()
        return True
    _clear_published_atomic_state()
    runtime = build_laboratory_runtime(ATOM, request)
    # Recheck the fields/atom before committing a potentially expensive scan.
    require_matching_applied_fields(module, request)
    if laboratory_identity(ATOM, request) != runtime.identity:
        raise PhysicsConfigurationError('LAB-004: atom identity changed during initialization; no partial publication.')
    populate_laboratory_caches(module, runtime)
    rabi = runtime.selection.rabi
    params.SI_Constants.LAMBDA_P = request.probe.wavelength_nm * 1e-9
    params.OMEGA_S_BASE, params.OMEGA_LO_BASE = rabi.signal, rabi.lo
    ATOMIC_LEVELS = _ATOMIC_LEVELS_CACHE[identity]
    ARC_RF_DIPOLE_EA0 = runtime.selection.winner.rf_ea0
    GAMMA_21, GAMMA_32, GAMMA_43 = runtime.rates
    OMEGA_P, OMEGA_C, OMEGA_RF_BASE, OMEGA_SIGNAL_RF_BASE = rabi.probe, rabi.coupling, rabi.lo, rabi.signal
    DELTA_P, DELTA_C, DELTA_RF = runtime.detunings
    _RF_ORIENTATION = runtime.selection.rf_orientation
    _INITIALIZED_RESOLUTION_IDENTITY, _INITIALIZED_REQUEST = identity, request
    _INITIALIZED_ATOM_IDENTITY = _atom_identity()
    _ATOMIC_PHYSICS_INITIALIZED = True
    return True


def reset_atomic_physics():
    """Explicitly unpublish runtime and clear its caches before input changes.

    Applied fields are preserved. No initialization, tuning or recalculation
    occurs; consumers must initialize explicitly again before physics reads.
    """
    _clear_published_atomic_state()
    for cache in (_ATOMIC_LEVELS_CACHE, _DETUNING_CACHE, _DIPOLE_MATRIX_CACHE, _OPTIMIZATION_RESULTS_CACHE):
        cache.clear()


def get_laboratory_runtime():
    """Return immutable LAB-004 metadata of a guarded laboratory runtime."""
    require_atomic_physics_initialized()
    from .laboratory import ExperimentalLabRequest
    if not isinstance(_INITIALIZED_REQUEST, ExperimentalLabRequest):
        raise PhysicsConfigurationError('LAB-004: published runtime was not initialized from a laboratory request.')
    from .laboratory_runtime import require_current_laboratory_runtime
    return require_current_laboratory_runtime(sys.modules[__name__])


@_transactional_atomic_initialization
def initialize_atomic_physics(*, force: bool = False,
                              resolution_request=None) -> bool:
    """ARCH-001: explicitly run the atomic level optimization and derived physics.

    CANONICAL initialization entry point (the only one). Everything that used
    to execute at module import time now lives here, in the original order and
    with the original formulas:

        1. atomic resolution (MODE-001 — see below);
        2. ARC RF dipole read-back;
        3. spontaneous decay rates (GAMMA_21 / GAMMA_32 / GAMMA_43);
        4. Rabi frequencies (probe, coupling, LO RF, signal RF) plus the
           params.OMEGA_S_BASE / params.OMEGA_LO_BASE heterodyne update;
        5. detunings (DELTA_P / DELTA_C / DELTA_RF).

    Step 1 branches on the atomic-resolution mode (MODE-001):

        Explicit ExperimentalLabRequest (LAB-004)
            RF-first optical/prepared-channel selection; full request/atom
            identity, unchanged applied fields and coherent derived caches.
            Optical HFS policy and frequency offsets follow that request.
            Pass it as resolution_request; the public signature is unchanged.

        EXPERIMENTAL (the DEFAULT)
            ``_initialize_atomic_levels()`` — the historical RF-first
            ``SingleStageLevelOptimizer`` search.  AUTO-001A-R2: the search
            is FS-only — it fills the level and detuning caches, computes NO
            dipole matrix, and still rewrites the coupling wavelength and
            ``params.SI_Constants.LAMBDA_P`` exactly as before.

        PRESCRIBED
            ``_initialize_prescribed_levels()`` — the four declared states are
            used verbatim.  No level search, no mJ scan, no q scan, no legacy
            ``arc/dipole_matrix`` RF max-scan cache; applied wavelengths are
            never rewritten (§13) and the dipole matrix is built from the QN
            -001A.1/B/C declared-channel primitives only.

    Semantics (ARCH-001):

      * EXPLICIT - nothing in the package calls this implicitly; readers only
        guard via require_atomic_physics_initialized(), they never auto-init.
      * IDEMPOTENT - a second call returns at once without recomputing the ARC
        pipeline, unless ``force=True`` is passed.
      * MODE-CONSISTENT - a second call whose resolution request differs from
        the one that produced the published runtime FAILS LOUDLY instead of
        returning another mode's numbers (MODE-001).
      * TRANSACTIONAL - the compatibility globals and the initialized flag are
        published only after EVERY step above succeeded. On failure the
        original exception propagates unchanged (F05: no value substitution)
        and the runtime stays uninitialized; nothing partial is advertised.
        ARCH-001TX extends this to the CONFIGURATION state: a transactional
        wrapper snapshots ``LASER_PARAMS``/``RF_PARAMS``,
        ``SI_Constants.LAMBDA_P``, ``LEVEL_SELECTION_TOLERANCES``,
        ``params.OMEGA_*_BASE`` and the four pipeline caches on entry and
        rolls them back on failure, so a failed attempt (e.g. the documented
        QN-001A coupling boundary) can never leave a rewritten coupling
        wavelength or a half-populated cache behind for later readers.
        ARCH-001TX-R1 also preserves the identity of the nested
        member dictionaries (``LASER_PARAMS['probe' | 'coupling']``):
        references held before the attempt remain valid afterwards.

    With the current Experimental physics state, step 4 raises the documented
    PhysicsConfigurationError at the coupling HFS->FS boundary. That is the
    physics contract, not an architecture failure: the flag then stays False
    and no Rabi frequency is published.  (MODE-001 documents a way to declare
    that channel explicitly rather than search for it.)

    Args:
        force: re-run the pipeline even if already initialized. Not part of
            ordinary production execution.
        resolution_request: MODE-001 — an ``AtomicResolutionRequest``, a mode
            (enum or its string value), or ``None`` to consult the configured
            switch in ``atomic_resolution`` (whose default is EXPERIMENTAL, so
            existing no-argument callers keep today's behaviour).

    Returns:
        bool: True when atomic physics is fully initialized.

    Raises:
        PhysicsConfigurationError: the QN-001A coupling boundary (or any other
            F05 fail-fast condition) raised inside the pipeline, an ambiguous
            resolution configuration, or a mode/ladder change requested
            without ``force=True``.
        RuntimeError: the preserved ARC atom / decay / Rabi contracts of the
            original import-time code.
    """
    global _ATOMIC_PHYSICS_INITIALIZED
    global _INITIALIZED_RESOLUTION_IDENTITY
    global _INITIALIZED_REQUEST
    global ATOMIC_LEVELS, ARC_RF_DIPOLE_EA0
    global GAMMA_21, GAMMA_32, GAMMA_43
    global OMEGA_P, OMEGA_C, OMEGA_RF_BASE, OMEGA_SIGNAL_RF_BASE
    global DELTA_P, DELTA_C, DELTA_RF
    global _RF_ORIENTATION, _INITIALIZED_ATOM_IDENTITY

    from .laboratory import ExperimentalLabRequest
    if isinstance(resolution_request, ExperimentalLabRequest):
        return _initialize_laboratory_physics(resolution_request, force=force)
    request = resolve_resolution_request(resolution_request)

    if not request.is_prescribed:
        # LAB-002: selection, geometry and heterodyne must describe the same
        # applied RF pair. Reject conflicting views before idempotency/ARC;
        # never select one stale copy as an implicit authority.
        from .rf_configuration import read_configured_rf_fields
        read_configured_rf_fields()

    if ATOM is None:
        configure_atom(ATOM_SPEC)
    request_identity = resolution_request_identity(request)
    if (ATOM_SPEC != LEGACY_ATOM or type(ATOM).__name__ in ('Rubidium87', 'Caesium')
            or (type(ATOM).__name__ == 'Rubidium85' and ATOM is not get_arc_atom(LEGACY_ATOM))):
        request_identity = ('ATOM-001', _atom_identity(), request_identity)

    if _ATOMIC_PHYSICS_INITIALIZED and not force:
        if request_identity != _INITIALIZED_RESOLUTION_IDENTITY:
            # Returning the currently published runtime would hand an
            # EXPERIMENTAL result to a PRESCRIBED caller (or vice versa), or
            # one declared ladder's numbers to a different declared ladder.
            raise PhysicsConfigurationError(
                "atomic physics is already initialized for resolution "
                f"{_INITIALIZED_RESOLUTION_IDENTITY!r}, but "
                f"{request_identity!r} was requested. MODE-001: the runtime "
                "is not silently reinterpreted — pass force=True to re-run "
                "the pipeline for the new resolution."
            )
        require_atomic_physics_initialized()
        return True  # idempotent: the ARC pipeline runs once

    if force and _ATOMIC_PHYSICS_INITIALIZED:
        # A forced re-run rewrites the level / detuning / dipole caches; if it
        # now fails, values from the previous success must not stay advertised
        # next to those rewritten caches.
        _clear_published_atomic_state()

    # ---- step 1: atomic resolution (MODE-001 boundary) ---------------------
    if request.is_prescribed:
        levels = _initialize_prescribed_levels(request)
    else:
        # Historical RF-first optimizer; behaviour unchanged.
        levels = _initialize_atomic_levels()
    # ---- step 2: ARC RF dipole read-back (cache read) ----
    # ARCH-001: unguarded internal read — the flag is still False here.
    cache_key = _resolution_cache_key(request)
    arc_rf_dipole_ea0 = _resolve_arc_rf_dipole(cache_key)

    # ---- steps 3 + 4: the code that used to run at module level, moved ----

    # ---- step 3: spontaneous decay rates (formulas unchanged) ----
    #
    # RF-DIR-001B: the RF EDGE IS NOW EVALUATED IN ENERGY ORDER.
    #
    # MODE-001 §15 deferred this ("RF direction is reported, not fixed");
    # RF-DIR-001 is that follow-up.  The optical decay chain stays in ladder
    # order (excited -> ground, rydberg1 -> excited — both energy-ascending
    # for this system); the |3> <-> |4> RF edge is resolved from the SIGNED
    # ARC transition frequency through arc.rf_orientation (RF-DIR-001A), and
    # ``gamma_43`` is the spontaneous rate of the ENERGY-DOWNWARD channel
    # (production pair: R1(26D5/2) -> R2(27P3/2)).  The resolved orientation
    # is published (``_RF_ORIENTATION`` / ``get_rf_orientation()``) and
    # threaded by the solver seams into ``create_sensor_model``, which builds
    # the top-to-bottom jump; the Hamiltonian itself is NOT modified here
    # (RF-DIR-001C owns the rotating-frame sign).
    if ATOM is not None:
        print("=== ARC Atomic Calculations ===")
        try:
            from ..arc.decay_rates import (
                calculate_decay_rates,
                resolve_rydberg_orientation,
            )
            gamma_21, gamma_32, gamma_43 = calculate_decay_rates(
                ATOM,
                levels['ground'],
                levels['excited'],
                levels['rydberg1'],
                levels['rydberg2']
            )
            # RF-DIR-001B: resolve and publish the energy direction of the RF
            # edge (single source: the signed ARC transition frequency).
            rf_orientation = resolve_rydberg_orientation(
                ATOM, levels['rydberg1'], levels['rydberg2']
            )
            print(f"ARC: Dynamic level decay rates calculated")
            print(f"ARC: Gamma_21 = {gamma_21:.2f} 1/us (excited -> ground)")
            print(f"ARC: Gamma_32 = {gamma_32:.2f} 1/us (rydberg1 -> excited)")
            print(f"ARC: Gamma_43 = {gamma_43:.3e} 1/us (energy-downward RF edge: "
                  f"{rf_orientation.decay_description})")
            print(f"ARC: RF edge direction = {rf_orientation.direction:+d}, "
                  f"carrier |nu| = {rf_orientation.carrier_hz / 1e9:.6f} GHz")
            print()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            raise
    else:
        print("ERROR: ARC atomic database not available - cannot calculate spontaneous decay rates (Einstein A coefficients)")
        print("Please install ARC (Alkali Rydberg Calculator) to enable atomic physics calculations")
        raise RuntimeError("ARC database required for atomic decay rate calculations")

    # Calculate Rabi Frequencies using ARC (following Bussey et al. 2024 Figure 2)
    if ATOM is not None:
        print("=== ARC Rabi Frequency Calculations ===")

        # Define level labels dynamically
        ground_label = f"{levels['ground']['n']}{['S','P','D','F', 'G', 'H', 'K'][levels['ground']['l']]}{levels['ground']['j']}"
        excited_label = f"{levels['excited']['n']}{['S','P','D','F', 'G', 'H', 'K'][levels['excited']['l']]}{levels['excited']['j']}"
        rydberg1_label = f"{levels['rydberg1']['n']}{['S','P','D','F', 'G', 'H', 'K'][levels['rydberg1']['l']]}{levels['rydberg1']['j']}"
        rydberg2_label = f"{levels['rydberg2']['n']}{['S','P','D','F', 'G', 'H', 'K'][levels['rydberg2']['l']]}{levels['rydberg2']['j']}"

        # Probe laser: 5S1/2 -> 5P3/2
        omega_p_rads = calculate_laser_rabi_frequency(
            ATOM, levels['ground'], levels['excited'], LASER_PARAMS['probe']
        )
        omega_p = omega_p_rads / US_TO_S_FACTOR  # Convert rad/s to rad/μs
        print(f"ARC: Probe laser ({ground_label} -> {excited_label}) | Omega = {omega_p:.2f} rad/us ({omega_p_rads/1e6/2/np.pi:.1f} MHz)")

        # Coupling laser: excited state -> first rydberg state
        omega_c_rads = calculate_laser_rabi_frequency(
            ATOM, levels['excited'], levels['rydberg1'], LASER_PARAMS['coupling']
        )
        omega_c = omega_c_rads / US_TO_S_FACTOR  # Convert rad/s to rad/μs
        print(f"ARC: Coupling laser ({excited_label} -> {rydberg1_label}) | Omega = {omega_c:.2f} rad/us ({omega_c_rads/1e6/2/np.pi:.1f} MHz)")

        # RF field: first Ryberg state -> secondRyberg state
        omega_rf_rads = calculate_rf_rabi_frequency(
            ATOM, levels['rydberg1'], levels['rydberg2'], RF_PARAMS
        )
        omega_rf_base = omega_rf_rads / US_TO_S_FACTOR  # Convert rad/s to rad/μs
        print(f"ARC: LO RF field ({rydberg1_label} -> {rydberg2_label}) | Omega = {omega_rf_base:.2f} rad/us ({omega_rf_rads/1e6/2/np.pi:.1f} MHz)")

        # Signal RF field: 50D5/2 -> 51P3/2 (calculated from SIGNAL_RF_PARAMS)
        omega_signal_rf_rads = get_signal_rf_rabi_frequency(
            ATOM, levels['rydberg1'], levels['rydberg2'], SIGNAL_RF_PARAMS
        )
        omega_signal_rf_base = omega_signal_rf_rads / US_TO_S_FACTOR  # Convert rad/s to rad/μs
        print(f"ARC: Signal RF field ({rydberg1_label} -> {rydberg2_label}) | Omega = {omega_signal_rf_base:.2f} rad/us ({omega_signal_rf_rads/1e6/2/np.pi:.1f} MHz)")

        # Use the already calculated RF Rabi frequencies for heterodyne (fixes zero amplitude bug)
        from . import parameters as params
        omega_s_mhz = omega_signal_rf_base  /2/np.pi  # Convert rad/μs to MHz
        omega_lo_mhz = omega_rf_base /2/np.pi         # Convert rad/μs to MHz

        # Update parameters with physics-based Rabi frequency values
        params.OMEGA_S_BASE = omega_signal_rf_base
        params.OMEGA_LO_BASE = omega_rf_base

        print(f"Updated heterodyne Rabi frequencies using calculated ARC values:")
        print(f"  OMEGA_S_BASE (signal): {omega_s_mhz:.3f} MHz = {params.OMEGA_S_BASE:.3f} rad/us")
        print(f"  OMEGA_LO_BASE (LO): {omega_lo_mhz:.3f} MHz = {params.OMEGA_LO_BASE:.3f} rad/us")
        print()
    else:
        print("ERROR: ARC atomic database not available - cannot calculate Rabi frequencies")
        print("Rabi frequencies require dipole matrix elements from ARC database")
        print("Please install ARC (Alkali Rydberg Calculator) to enable atomic physics calculations")
        raise RuntimeError("ARC database required for Rabi frequency calculations")
    # ---- step 5: detunings (RF-003 body, unchanged) ----
    # The key carries the resolution identity, so a PRESCRIBED run reads the
    # detunings written by the prescribed branch (MODE-001).
    delta_p, delta_c, delta_rf = _resolve_cached_detunings(cache_key)

    # ---- publish in one step: only now does the runtime become valid ----
    ATOMIC_LEVELS = levels
    ARC_RF_DIPOLE_EA0 = arc_rf_dipole_ea0
    GAMMA_21, GAMMA_32, GAMMA_43 = gamma_21, gamma_32, gamma_43
    _RF_ORIENTATION = rf_orientation
    OMEGA_P = omega_p
    OMEGA_C = omega_c
    OMEGA_RF_BASE = omega_rf_base
    OMEGA_SIGNAL_RF_BASE = omega_signal_rf_base
    DELTA_P, DELTA_C, DELTA_RF = delta_p, delta_c, delta_rf
    # MODE-001: record WHICH resolution produced these numbers so a later call
    # asking for a different one is refused instead of answered with them.
    # The full request is kept as well, so no-argument readers resolve their
    # cache key from the mode that was actually resolved, not from whatever
    # the live configured switch says afterward.
    _INITIALIZED_RESOLUTION_IDENTITY = request_identity
    _INITIALIZED_REQUEST = request
    _INITIALIZED_ATOM_IDENTITY = _atom_identity()
    _ATOMIC_PHYSICS_INITIALIZED = True
    return True

def print_arc_summary():
    """
    Print summary of ARC-calculated atomic parameters.
    Call this function explicitly to show ARC results when running simulations.
    """
    require_atomic_physics_initialized()
    print("=== ARC Atomic Parameters Summary ===")
    if ATOM is not None:
        print("[OK] ARC database available - using dynamically calculated atomic parameters")

        # Print levels
        print("  Atomic levels:")
        try:
            ground_label = f"{ATOMIC_LEVELS['ground']['n']}{['S','P','D','F'][ATOMIC_LEVELS['ground'].get('l', 0)]}{ATOMIC_LEVELS['ground'].get('j', 0.5)}"
            excited_label = f"{ATOMIC_LEVELS['excited']['n']}{['S','P','D','F'][ATOMIC_LEVELS['excited'].get('l', 1)]}{ATOMIC_LEVELS['excited'].get('j', 1.5)}"
            rydberg1_label = f"{ATOMIC_LEVELS['rydberg1']['n']}{['S','P','D','F'][ATOMIC_LEVELS['rydberg1'].get('l', 2)]}{ATOMIC_LEVELS['rydberg1'].get('j', 2.5)}"
            rydberg2_label = f"{ATOMIC_LEVELS['rydberg2']['n']}{['S','P','D','F'][ATOMIC_LEVELS['rydberg2'].get('l', 1)]}{ATOMIC_LEVELS['rydberg2'].get('j', 1.5)}"
            print(f"    Ground: {ground_label}")
            print(f"    Excited: {excited_label}")
            print(f"    Rydberg1: {rydberg1_label}")
            print(f"    Rydberg2: {rydberg2_label}")
        except:
            print("    (Level info not available)")

        print("  Decay rates:")
        print(f"    Gamma_21 = {GAMMA_21:.2f} 1/us (excited -> ground)")
        print(f"    Gamma_32 = {GAMMA_32:.2f} 1/us (rydberg1 -> excited)")
        print(f"    Gamma_43 = {GAMMA_43:.3e} 1/us (energy-downward RF edge, RF-DIR-001B)")
        if _RF_ORIENTATION is not None:
            print(f"    RF edge orientation: {_RF_ORIENTATION.decay_description} "
                  f"(direction {_RF_ORIENTATION.direction:+d}, carrier "
                  f"{_RF_ORIENTATION.carrier_hz / 1e9:.6f} GHz)")
        print("  Rabi frequencies:")
        print(f"    Probe: Omega = {OMEGA_P:.2f} rad/us")
        print(f"    Coupling: Omega = {OMEGA_C:.2f} rad/us")
        print(f"    RF: Omega = {OMEGA_RF_BASE:.2f} rad/us")
        print("  Decoherence: Laser linewidths included per Bussey et al. (2024)")
    else:
        print("[WARN] ARC database not available - no manual fallback values are "
              "published (F05); decay rates exist only after "
              "initialize_atomic_physics() and are event rates in 1/us "
              "(s^-1 / 1e6, never 2*pi)")
    print()


def _resolve_cached_detunings(cache_key: Optional[tuple] = None):
    """
    Retrieve calculated detunings from the atomic configuration cache.

    Unguarded RF-003 body: called by initialize_atomic_physics() while the
    initialization flag is still False. External readers must go through
    get_cached_detunings(), which applies the ARCH-001 guard.

    MODE-001: ``cache_key`` defaults to the key of the request that PRODUCED
    the runtime (``_INITIALIZED_REQUEST``), falling back to the CURRENT
    resolution's key when no initialization has stored one yet, so a
    PRESCRIBED run reads the detunings written under the extended prescribed
    key rather than missing the experimental 5-field entry.

    RF-003 fail-loudly contract: a genuinely MISSING cache entry raises
    RuntimeError instead of silently returning zeros, while an entry that is
    PRESENT and legitimately zero (e.g. coupling on resonance, Δ_c = 0.0) is
    returned without error — entry presence, not value, distinguishes the
    failure case from the valid-zero case.

    Returns:
        tuple: (delta_p_rad_us, delta_c_rad_us, delta_rf_rad_us)

    Raises:
        RuntimeError: If the resolved cache key is absent from _DETUNING_CACHE.
    """
    # Same explicit-parameter key as every other consumer (N-1 fix),
    # extended by the resolution identity in PRESCRIBED mode (MODE-001).
    if cache_key is None:
        request, _from_switch = _initialized_resolution_request()
        cache_key = _resolution_cache_key(request)

    if cache_key not in _DETUNING_CACHE:
        raise RuntimeError(
            "Detuning cache miss for key "
            f"{cache_key} — the explicit atomic-resolution step "
            "(initialize_atomic_physics) has not populated detunings for the "
            "current laser/RF parameters and resolution mode. "
            "Refusing to return silent zeros (RF-003)."
        )

    detunings = _DETUNING_CACHE[cache_key]
    required_keys = ('delta_p_rad_us', 'delta_c_rad_us', 'delta_rf_rad_us')
    missing = [key for key in required_keys if key not in detunings]
    if missing:
        # F05: an incomplete cache entry must never be read as 0.0 (resonance).
        raise PhysicsCalculationError(
            f"Detuning cache entry for key {cache_key} is incomplete "
            f"(missing {missing}); refusing to substitute 0.0 (resonance). "
            "F05: no silent resonance substitution."
        )

    delta_p = detunings['delta_p_rad_us']
    delta_c = detunings['delta_c_rad_us']
    delta_rf = detunings['delta_rf_rad_us']

    for label, value in (('delta_p', delta_p), ('delta_c', delta_c),
                         ('delta_rf', delta_rf)):
        if not np.isfinite(value):
            raise PhysicsCalculationError(
                f"Cached {label} detuning is not finite ({value!r}) for key "
                f"{cache_key}. F05: no silent substitution."
            )

    return delta_p, delta_c, delta_rf


def get_cached_detunings():
    """Return the cached detunings (delta_p, delta_c, delta_rf) in rad/us.

    ARCH-001: this guarded public accessor never initializes implicitly. The
    RF-003 fail-loudly contract is unchanged - a genuinely missing cache entry
    still raises RuntimeError instead of silently returning zeros.

    Returns:
        tuple: (delta_p, delta_c, delta_rf) in rad/us.

    Raises:
        PhysicsConfigurationError: before initialize_atomic_physics() ran.
        RuntimeError: RF-003 - the cache entry for the current parameters is
            missing (refusing to substitute resonance zeros).
    """
    require_atomic_physics_initialized()
    return _resolve_cached_detunings()


def get_rf_orientation():
    """RF-DIR-001B: resolved energy orientation of the |3> <-> |4> RF edge.

    ARCH-001 guarded reader: never initializes implicitly. Returns the
    RFTransitionOrientation published by ``initialize_atomic_physics()``
    (resolved from the signed ARC transition frequency, RF-DIR-001A), or
    ``None`` when no orientation has been published — nothing is fabricated
    (F05).

    The solver seams thread this object into ``create_sensor_model`` so the
    RF spontaneous-emission jump follows the energy order; ``None`` keeps the
    historical ladder-index build.

    Returns:
        RFTransitionOrientation | None

    Raises:
        PhysicsConfigurationError: before initialize_atomic_physics() ran.
    """
    require_atomic_physics_initialized()
    return _RF_ORIENTATION


def select_optimal_levels(probe_params, coupling_params, rf_params):
    """Select optimal atomic levels for the Rydberg sensor using two-stage optimization.

    Args:
        probe_params (LaserParams): Probe laser parameters.
        coupling_params (LaserParams): Coupling laser parameters.
        rf_params (RFParams): RF parameters.

    Returns:
        dict: Dictionary containing selected atomic levels.
    """
    from ..arc import LaserParams, RFParams

    # Single-stage ARC optimizer is the current canonical architecture; the
    # historical two-stage class was retired from level_finder, so binding it
    # here made every call fail with ImportError before doing any work.
    from ..level_finder import SingleStageLevelOptimizer

    optimizer = SingleStageLevelOptimizer(arc_atom=ATOM)
    return optimizer.optimize(probe_params, coupling_params, rf_params)


# ARCH-001: DELTA_P / DELTA_C / DELTA_RF are published by
# initialize_atomic_physics(); there is no module-level detuning read.
