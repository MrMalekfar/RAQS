"""
Steady-state spectrum calculations.

The numerical path optionally averages the EIT response over Maxwell-Boltzmann
velocity classes (Doppler broadening) at the LINEAR-RESPONSE level (C-002):

    rho_ens(Δ_p) = Σ_i w_i · rho(Δ_p + Δ_Dp(v_i), Δ_c + Δ_Dc(v_i))
    T(Δ_p)       = BeerLambert(rho_ens)          # exactly ONE propagation step

with weights w_i and velocities v_i from `physics.doppler.generate_velocity_classes`.
Ensembling at the coherence/susceptibility level — rather than averaging per-class
transmissions — is the physically correct ordering for a weak-probe vapour: the
macroscopic polarization is the weighted sum of the class coherences, and the
optical field then propagates through the ensemble exactly once.
The analytical path remains the zero-velocity reference solution and is not
Doppler-averaged.
"""

import numpy as np
from qutip import liouvillian, steadystate
from ..physics.hamiltonian import create_sensor_model
from ..arc.prepared_decay import PreparedRadiativeGraph
from ..exceptions import PhysicsConfigurationError
from ..physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
from ..physics.doppler import (
    generate_velocity_classes,
    get_doppler_shifts,
    integrate_maxwell_response,
    resonance_velocity_hints,
    velocity_scale_hint,
)
from ..utils.validation import check_conservation_laws, validate_steady_state
from ..config import atomic_config as acfg
from ..config import parameters as params
from ..config.laser_rf_config import LASER_PARAMS
from ..analytical import solutions
from .noise_seam import apply_noise_seam
from ..physics.lindblad import (
    calculate_transit_rate_per_us,
    resolve_effective_transit_rate,
)
from ..physics.lindblad.calculate_dephasing_rates import calculate_dephasing_rates
from ..physics.lindblad.solve_steady_state_direct import solve_steady_state_direct


# Steady-state backend seam (DOP-003S); optionally selected by the spectrum profile.
_BACKENDS = ("qutip", "direct")


def _steady_state_solve(H, c_ops, backend):
    """Dispatch the steady-state solve.

    ``"qutip"`` → ``qutip.steadystate`` (reference backend, unchanged
    physics); ``"direct"`` → exact dense trace-constrained solve of the
    SAME Liouvillian (DOP-003S, ~40x faster per call).  Both solve the
    identical master equation; the choice is a performance seam only.

    NUM-SS-001: both branches pass their result through the SAME fail-loud
    certification contract (``utils.validation.validate_steady_state``)
    before the state can reach coherence, transmission or any other
    consumer.  The ``"direct"`` branch is certified inside
    ``solve_steady_state_direct``, against the very Liouvillian it solved
    (no second construction, no duplicate work); the ``"qutip"`` branch is
    certified here.  Neither branch repairs, renormalizes or replaces a
    rejected state (F05) — a rejection propagates unchanged.
    """
    if backend == "direct":
        return solve_steady_state_direct(H, c_ops)
    if backend == "qutip":
        rho = steadystate(H, c_ops)
        # Stationarity is measured against the ORIGINAL Liouvillian of the
        # same (H, c_ops) pair the solver was handed.
        validate_steady_state(
            rho, liouvillian(H, list(c_ops)),
            context="_steady_state_solve(qutip)", backend="qutip",
        )
        return rho
    raise ValueError(
        "steady_state_backend must be one of %r, got %r" % (_BACKENDS, backend)
    )



def _resolve_transit_rate(transit_rate_per_us=None, transit_enabled=None):
    """PHY-SW-001 gated lazy default (delegates to the shared resolver).

    ``transit_enabled=False`` → 0.0 (no reset collapse operators, regardless
    of any explicit rate); ``None`` → ``params.TRANSIT_ENABLED``; ``True``
    with ``None`` rate → configuration-derived effective Markovian rate.
    """
    return resolve_effective_transit_rate(transit_rate_per_us, transit_enabled)


def _homogeneous_gamma_scale(p_gammas, p_dephasing, transit_rate):
    """Homogeneous coherence scale [rad/μs] driving the velocity-hint stencil.

    Uses the largest decay/dephasing/reset rate of the excited-level block;
    this is a conservative width for EIT/AT features.
    """
    gammas = (p_gammas.outgoing_rates_per_us if isinstance(p_gammas, PreparedRadiativeGraph)
              else list(p_gammas) if p_gammas is not None else [])
    deph = list(p_dephasing) if p_dephasing is not None else []
    transit = float(transit_rate) if transit_rate is not None else 0.0
    scale = float(transit)
    for g in gammas:
        scale = max(scale, float(g))
    for d in deph:
        scale = max(scale, float(d))
    # Avoid a degenerate zero scale (e.g. pure synthetic inputs).
    return scale if scale > 1e-30 else 1.0


def _breakpoint_stencil_for(v_center, dv_scale):
    """Breakpoint stencil around a resonance hint (SAMPLING AID only).

    Symmetric factors are REQUIRED: a one-sided stencil leaves the
    negative-velocity flank of a resonance feature without internal
    breakpoints and quad_vec silently misses that half (measured 50%
    error on a synthetic narrow Gaussian).
    """
    factors = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    return [v_center + s * f * dv_scale
            for f in factors for s in (1.0, -1.0)]


def _clean_breakpoints(breakpoints, n_sigma=5.0, cap=200, temperature_k=None, atom_mass_kg=None):
    """Deduplicate, sort, drop points at/outside the domain, cap the count.

    Velocities are m/s; the domain bound is computed from the configured
    N_SIGMA and the Rb mass/temperature from parameters (adaptive hints are
    only used in the production adaptive path where these exist).
    """
    from ..physics.doppler.adaptive_maxwell import get_velocity_std as _vs

    v_max = n_sigma * _vs(params.VAPOR_CELL_TEMPERATURE_K if temperature_k is None else temperature_k,
                            acfg.require_configured_atom().mass if atom_mass_kg is None else atom_mass_kg)
    pts = []
    for bp in breakpoints:
        b = float(bp)
        if -v_max < b < v_max:
            pts.append(round(b, 6))
    pts = sorted(set(pts))
    # merge points that are within 1e-3 m/s of each other
    merged = []
    for p in pts:
        if not merged or abs(p - merged[-1]) > 1e-3:
            merged.append(p)
    return merged[:cap]


def _solve_probe_coherence(
    delta_p_rad,        # Total probe detuning Δ_p [rad/μs] (center + scan + Doppler shift)
    delta_c_rad,        # Total coupling detuning Δ_c [rad/μs] (base + Doppler shift)
    omega_p, omega_c, omega_rf,
    delta_rf_rad,       # RF detuning Δ_rf [rad/μs]
    p_gammas, p_dephasing,
    noise_model=None,   # RF-004: optional noise model applied at the solver seam
    transit_rate_per_us=None,  # DPH-002B: effective transit reset rate [1/μs]; None → config default
    steady_state_backend="qutip",  # DOP-003S: "qutip" (reference) or "direct"
    rf_orientation=None,  # RF-DIR-001B: energy direction of the RF-edge jump; None → historical ladder-index build
):
    """
    Single-point steady-state solve returning the probe coherence.

    Builds the 4-level cascade model at the given detunings, solves ρ_ss from
    the Lindblad master equation, and returns the probe coherence
    ρ₂₁ = Tr[σ₂₁·ρ_ss] (Bussey Eq. 8) BEFORE any transmission conversion.
    This is the quantity that must be ensemble-averaged across velocity
    classes (C-002) before Beer-Lambert propagation is applied ONCE.

    ``steady_state_backend`` selects ONLY the linear-algebra backend that
    solves the stationary Liouvillian problem: both ``"qutip"`` and
    ``"direct"`` solve the identical master equation built from the same
    ``create_sensor_model`` Hamiltonian and collapse operators (DOP-003S
    parity: |Δρ₂₁| ≲ 1e-16). The physics seam (noise model) is applied
    identically before either backend is invoked.
    """
    H, c_ops, obs_probe_coherence = create_sensor_model(
        omega_p, omega_c, omega_rf,
        delta_p_rad, delta_c_rad, delta_rf_rad,
        p_gammas, p_dephasing,
        transit_rate_per_us=(_resolve_transit_rate() if transit_rate_per_us is None else transit_rate_per_us),
        rf_orientation=rf_orientation,
    )
    # RF-004 seam: post-model / pre-solver hook. With noise_model=None this is
    # a strict identity — the Liouvillian reaching `steadystate` is unchanged
    # object-for-object (frozen-baseline preservation by construction).
    # The seam's documented contract returns exactly (H, c_ops); the
    # observable is forwarded as read-only context and is never modified, so
    # `obs_probe_coherence` from `create_sensor_model` stays authoritative.
    H, c_ops = apply_noise_seam(
        H, c_ops, obs=obs_probe_coherence, noise_model=noise_model
    )
    # ρ_ss — stationary state of the Lindblad master equation (dρ/dt = 0);
    # backend is a performance seam only (DOP-003S), physics is identical.
    rho_ss = _steady_state_solve(H, c_ops, steady_state_backend)
    # ⟨σ₂₁⟩ = Tr[σ₂₁·ρ_ss] — probe coherence feeding χ ∝ ρ₂₁/Ω_p (Bussey Eq. 8)
    rho_21_num = (obs_probe_coherence * rho_ss).tr()
    return rho_21_num


def _solve_probe_transmission(
    delta_p_rad,        # Total probe detuning Δ_p [rad/μs] (center + scan + Doppler shift)
    delta_c_rad,        # Total coupling detuning Δ_c [rad/μs] (base + Doppler shift)
    omega_p, omega_c, omega_rf,
    delta_rf_rad,       # RF detuning Δ_rf [rad/μs]
    p_gammas, p_dephasing,
    noise_model=None,   # RF-004: optional noise model applied at the solver seam
    transit_rate_per_us=None,  # DPH-002B: effective transit reset rate [1/μs]; None → config default
    steady_state_backend="qutip",  # DOP-003S: forwarded to _solve_probe_coherence
    rf_orientation=None,  # RF-DIR-001B: forwarded to _solve_probe_coherence
):
    """
    Single-point steady-state solve returning the numerical probe transmission.

    Identical physics to ``_solve_probe_coherence`` followed by the
    susceptibility/Beer-Lambert conversion. Retained for single-point
    consumers; the Doppler ensemble path (``simulate_spectrum``) averages
    the coherence across velocity classes FIRST and converts ONCE (C-002).
    """
    rho_21_num = _solve_probe_coherence(
        delta_p_rad, delta_c_rad,
        omega_p, omega_c, omega_rf,
        delta_rf_rad, p_gammas, p_dephasing,
        noise_model=noise_model,
        transit_rate_per_us=transit_rate_per_us,
        steady_state_backend=steady_state_backend,
        rf_orientation=rf_orientation,
    )
    # T = exp(-2 α₀ d Im{χ}) — Beer-Lambert transmission from susceptibility
    return calculate_transmission_from_rho21(rho_21_num)


def simulate_spectrum(
    omega_rf,        # RF Rabi frequency for THIS run (0 or OMEGA_RF_BASE) — R-2A: AUTHORITATIVE
    detuning_list_mhz,   # R-2A: explicit scan [MHz] honored EXACTLY; None → configured default
    # PHYSICS PARAMETERS (Floats/Lists)
    p_omegas,            # (omega_p, omega_c, omega_rf_bundle) — R-2A: 3rd slot RETIRED, ignored
    p_deltas,            # (delta_p_center, delta_c, delta_rf)
    p_gammas,            # [gamma_21, gamma_32, gamma_43]; gamma_43 = energy-downward RF-edge rate (RF-DIR-001B)
    p_dephasing,         # [gamma_2, gamma_3, gamma_4]
    absorption_const,    # R-2A: DEAD/STALE compatibility argument — accepted, never read
    noise_model=None,    # RF-004: optional noise model applied at the solver seam
    transit_rate_per_us=None,  # DPH-002B: effective transit reset rate [1/μs]; None → config default
    steady_state_backend=None,  # DOP-003S internal seam; None → production default
    doppler_enabled=None,       # PHY-SW-001: None → params.DOPPLER_ENABLED
    transit_enabled=None,       # PHY-SW-001: None → params.TRANSIT_ENABLED (switch wins over explicit rate)
    laser_linewidth_dephasing_enabled=None,  # PHY-SW-001: None → config flag
    collision_dephasing_enabled=None,        # PHY-SW-001: None → config flag
    rf_orientation=None,  # RF-DIR-001B/AN-RF-001: RF-edge energy orientation — forwarded to BOTH the numerical builds and the analytical branch; None → historical build
    doppler_settings=None,  # LAB-007: explicit thermal/quadrature inputs
    transmission_prefactors=None,  # explicit (od_factor, K_chi)
    analytical_enabled=True,
):
    """
    Sweeps the probe detuning and finds the steady-state response.

    When ``params.DOPPLER_ENABLED`` is True, each scan point is averaged over
    thermal velocity classes (Doppler broadening) — adaptively via the
    production Maxwell integrator (DOP-002I) or fixed quadrature for the
    legacy methods; otherwise a single v=0 class reproduces the cold-atom
    result.

    R-2A parameter contract (one source of truth, no silent overwrite):

    * ``omega_rf`` (first positional argument) is the SINGLE authoritative RF
      Rabi frequency for this run (0.0 = RF-off, ``acfg.OMEGA_RF_BASE`` =
      RF-on). It is threaded unchanged into every ``create_sensor_model``
      build (H_34 = -Ω_rf/2) and into the analytical path.
    * ``p_omegas[2]`` is a RETIRED aggregate-bundle slot: it is unpacked but
      deliberately NEVER assigned to ``omega_rf``. Historically this unpack
      silently overwrote the explicit argument (defect R-2); production
      ``runner.py`` had to zero the bundle slot to compensate. Existing
      callers passing consistent values are unaffected.
    * ``detuning_list_mhz``: when not ``None``, the returned scan is EXACTLY
      this sequence (values, ordering, length preserved); when ``None``, the
      configured default ``linspace(-SCAN_RANGE_MHZ, SCAN_RANGE_MHZ, N_SCAN)``
      is generated. Historically the explicit list was ignored (defect R-2).
    * ``absorption_const`` is accepted for caller compatibility but is never
      read: the transmission conversion re-derives its constants from
      configuration internally (R-2A investigation; cleanup candidate R-2B).
    * ``rf_orientation`` (RF-DIR-001B/AN-RF-001): energy orientation of the
      ``|3> <-> |4>`` edge.  The SAME value supplied by the caller (production
      ``runner.py`` resolves it once via ``atomic_config.get_rf_orientation()``)
      is forwarded to every ``create_sensor_model`` build (RF-edge jump
      direction, RF-DIR-001B; state-4 rotating-frame diagonal sign,
      RF-DIR-001C) AND to the Appendix-A analytical branch (AN-RF-001), so the
      numerical and analytical responses always share one rotating-frame
      convention.  ``None`` keeps the historical build in BOTH branches
      (pre-RF-DIR-001B bit-compatibility).

    ``steady_state_backend`` selects the linear-algebra method (DOP-003S),
    optionally via the spectrum profile. ``None`` (default) resolves to
    ``"direct"`` (exact dense trace-constrained Liouvillian solve, ~40x
    faster) on the adaptive Doppler hot path and ``"qutip"`` everywhere else.
    An explicit value (``"qutip"`` / ``"direct"``) overrides everywhere. Both
    backends solve the identical master equation; parity |Δρ₂₁| ≲ 1e-16.

    PHY-SW-001 effect switches (all default ``None`` = production model,
    bit-identical to pre-PHY-SW-001 behavior):

    * ``doppler_enabled``: ``False`` → exactly one v=0 response (no Maxwell
      integration, no velocity classes); ``True`` → force Doppler averaging
      even if the global flag is off.
    * ``transit_enabled``: ``False`` → 0.0 transit rate (NO reset collapse
      operators), overriding even an explicit ``transit_rate_per_us``.
    * ``laser_linewidth_dephasing_enabled`` / ``collision_dephasing_enabled``:
      when either is given, ``p_dephasing`` is re-derived from the production
      budget with the requested component gates; when both are ``None`` the
      caller-supplied ``p_dephasing`` is used exactly as passed.

    Returns:
        tuple: (detuning_scan, transmission_numerical, transmission_analytic)
    """
    if isinstance(p_gammas, PreparedRadiativeGraph):
        rf_orientation = p_gammas.require_orientation(rf_orientation)
        if analytical_enabled:
            raise PhysicsConfigurationError('RELAX-001A: the legacy reduced analytical formula does not '
                                            'implement the prepared radiative graph; disable it explicitly.')
    # R-2A: ONE source of truth for the scan grid. An explicit
    # ``detuning_list_mhz`` is honored EXACTLY (values, ordering, length);
    # ``None`` regenerates the configured default linspace.
    if detuning_list_mhz is None:
        detuning_scan_list = np.linspace(-params.SCAN_RANGE_MHZ, params.SCAN_RANGE_MHZ, params.N_SCAN)
    else:
        detuning_scan_list = np.asarray(detuning_list_mhz, dtype=float).ravel()
        if detuning_scan_list.size == 0:
            raise ValueError(
                "detuning_list_mhz must be a non-empty sequence of scan "
                "points when provided (got an empty sequence)"
            )
    probe_transmission = []
    probe_transmission_analytic = []

    # DPH-002B: resolve the transit reset rate ONCE per spectrum and thread it
    # to every class solve (explicit production threading).
    # PHY-SW-001: the explicit ``transit_enabled`` switch (when given) wins
    # over any explicit rate — False ⇒ 0.0 ⇒ no reset collapse operators.
    transit_rate = _resolve_transit_rate(
        transit_rate_per_us, transit_enabled=transit_enabled
    )

    # PHY-SW-001: Doppler gate. ``None`` → production configuration flag
    # (default-path invariance); an explicit value overrides for the whole
    # spectrum WITHOUT touching the global configuration.
    doppler_on = (
        bool(params.DOPPLER_ENABLED) if doppler_enabled is None
        else bool(doppler_enabled)
    )

    # PHY-SW-001: dephasing-component gates. When either switch is given
    # explicitly, p_dephasing is re-derived from the production budget
    # (`calculate_dephasing_rates`) with the requested component gates; the
    # untouched source is the identical configuration, so a caller passing
    # the default budget gets identical numbers. When BOTH switches are None
    # the caller-supplied p_dephasing is used EXACTLY as passed (no hidden
    # recomputation — default-path invariance).
    if (laser_linewidth_dephasing_enabled is not None
            or collision_dephasing_enabled is not None):
        p_dephasing = calculate_dephasing_rates(
            laser_linewidth_enabled=(
                params.LASER_LINEWIDTH_DEPHASING_ENABLED
                if laser_linewidth_dephasing_enabled is None
                else bool(laser_linewidth_dephasing_enabled)
            ),
            collision_dephasing_enabled=(
                params.COLLISION_DEPHASING_ENABLED
                if collision_dephasing_enabled is None
                else bool(collision_dephasing_enabled)
            ),
        )


    # DOP-003S: resolve the internal steady-state backend. Production default
    # is "direct" (exact dense trace-constrained solve, ~40x faster) on the
    # adaptive Doppler hot path; "qutip" reference elsewhere. An explicit
    # argument (parity tests) overrides everywhere. Physics identical.
    if steady_state_backend is None:
        ss_backend_adaptive = "direct"
        ss_backend_default = "qutip"
    elif steady_state_backend in _BACKENDS:
        ss_backend_adaptive = ss_backend_default = steady_state_backend
    else:
        raise ValueError(
            "steady_state_backend must be one of %r or None, got %r"
            % (_BACKENDS, steady_state_backend)
        )

    # Unpack tuples for clarity.
    # R-2A: the explicit ``omega_rf`` argument is the SINGLE source of truth
    # for the RF Rabi frequency of this run. The third slot of ``p_omegas``
    # is a retired bundle field and is deliberately NOT assigned to
    # ``omega_rf`` (historical bug: this unpack silently overwrote the
    # explicit argument, so callers had to zero the bundle slot to make an
    # RF-off request effective).
    omega_p, omega_c, _omega_rf_bundle_retired = p_omegas
    delta_p_center, delta_c, delta_rf = p_deltas

    # --- VELOCITY CLASSES (Doppler averaging setup, done once per spectrum) ---
    temperature = params.VAPOR_CELL_TEMPERATURE_K if doppler_settings is None else doppler_settings.temperature_k
    method = params.DOPPLER_METHOD if doppler_settings is None else doppler_settings.method
    n_classes = params.DOPPLER_N_CLASSES if doppler_settings is None else doppler_settings.n_classes
    n_sigma = params.DOPPLER_N_SIGMA if doppler_settings is None else doppler_settings.n_sigma
    counter = params.DOPPLER_COUNTER_PROPAGATING if doppler_settings is None else doppler_settings.counter_propagating
    rtol = params.DOPPLER_ADAPTIVE_RTOL if doppler_settings is None else doppler_settings.rtol
    atol = params.DOPPLER_ADAPTIVE_ATOL if doppler_settings is None else doppler_settings.atol
    def transmit(coherence):
        if transmission_prefactors is None:
            return calculate_transmission_from_rho21(coherence)
        return calculate_transmission_from_rho21(coherence, absorption_prefactors=transmission_prefactors)

    if doppler_on:
        mass_kg_val = float(acfg.require_configured_atom().mass if doppler_settings is None
                            else doppler_settings.atom_mass_kg)
        # Adaptive method does not use discrete classes; still resolve
        # wavelengths once for the response function.
        if method != "adaptive":
            # Maxwell-Boltzmann quadrature nodes {v_i} [m/s] and weights {w_i}, Σw_i = 1
            velocities_ms, class_weights = generate_velocity_classes(
                n_classes=n_classes,
                temperature_k=temperature,
                atom_mass_kg=mass_kg_val,
                method=method,
                n_sigma=n_sigma,
            )
        else:
            velocities_ms, class_weights = np.array([0.0]), np.array([1.0])
        # Laser wavelengths λ from config → wavenumbers k = 2π/λ in get_doppler_shifts
        lambda_probe_m = LASER_PARAMS['probe']['wavelength_nm'] * 1e-9
        lambda_coupling_m = LASER_PARAMS['coupling']['wavelength_nm'] * 1e-9
    else:
        # Single zero-velocity class ≡ no Doppler averaging (cold-atom limit T→0)
        velocities_ms, class_weights = np.array([0.0]), np.array([1.0])
        lambda_probe_m = lambda_coupling_m = 0.0

    for dp_mhz in detuning_scan_list:
        # Convert MHz scan offset to rad/μs: Δ_scan = 2π · δν[MHz]
        scan_offset_rad = 2 * np.pi * dp_mhz
        # Total probe detuning = Center (usually 0) + Scan Offset
        current_delta_p = delta_p_center + scan_offset_rad

        # --- DOPPLER-ENSEMBLED NUMERICAL RESPONSE (C-002) ---
        # The macroscopic linear response is the weighted sum of the class
        # coherences; Beer-Lambert propagation is applied ONCE to the
        # ensemble afterwards:
        #   rho_ens(Δ_p) = Σ_i w_i · rho(Δ_p + δp(v_i), Δ_c + δc(v_i))
        #   T(Δ_p)       = transmission(rho_ens)
        # Averaging per-class transmissions instead would square the
        # nonlinearity of the exponential and hide genuine EIT/AT structure
        # at finite optical depth (C-002 finding).
        if doppler_on and method == "adaptive":
            # ---- production adaptive Maxwell integration (DOP-002I) ----
            # rho_ens(Δ_p) = ∫ dv f_MB(v) ρ21(v; Δ_p) — complex coherence
            # integrated FIRST, then converted to transmission ONCE.
            #
            # Local per-scan-point response cache: quad_vec may request the
            # same velocity more than once; cache lifetime is ONE ensemble
            # integration (rho21 also depends on probe detuning and RF state,
            # so a global cross-detuning cache would be wrong).
            _rho_cache = {}

            def rho_of_v(v_ms):
                key = round(float(v_ms), 10)
                if key in _rho_cache:
                    return _rho_cache[key]
                d_dp_shift, d_dc_shift = get_doppler_shifts(
                    float(v_ms), lambda_probe_m, lambda_coupling_m,
                    counter_propagating=counter,
                )
                val = _solve_probe_coherence(
                    current_delta_p + d_dp_shift, delta_c + d_dc_shift,
                    omega_p, omega_c, omega_rf, delta_rf,
                    p_gammas, p_dephasing,
                    noise_model=noise_model,
                    transit_rate_per_us=transit_rate,
                    steady_state_backend=ss_backend_adaptive,
                    rf_orientation=rf_orientation,
                )
                _rho_cache[key] = val
                return val

            # Parameter-derived resonance hints (SAMPLING AIDS only).
            hint_centers = resonance_velocity_hints(
                delta_p_center, delta_c, scan_offset_rad, delta_rf,
                omega_rf, lambda_probe_m, lambda_coupling_m,
                counter_propagating=counter,
            )
            dv_scale = velocity_scale_hint(
                _homogeneous_gamma_scale(p_gammas, p_dephasing, transit_rate),
                lambda_probe_m, lambda_coupling_m,
                counter_propagating=counter,
            )
            breakpoints = []
            for vc in hint_centers:
                breakpoints.extend(_breakpoint_stencil_for(vc, dv_scale))
            # Keep hints strictly interior; deduplicate; cap the count.
            breakpoints = _clean_breakpoints(breakpoints, n_sigma=n_sigma,
                                              temperature_k=temperature, atom_mass_kg=mass_kg_val)

            rho_ensemble = integrate_maxwell_response(
                rho_of_v,
                temperature_k=temperature,
                atom_mass_kg=mass_kg_val,
                n_sigma=n_sigma,
                epsrel=rtol,
                epsabs=atol,
                breakpoints=breakpoints,
                **({'require_convergence': True} if doppler_settings is not None else {}),
            )
        else:
            rho_ensemble = 0.0 + 0.0j
            for v_ms, w_i in zip(velocities_ms, class_weights):
                if doppler_on:
                    d_dp_shift, d_dc_shift = get_doppler_shifts(
                        float(v_ms), lambda_probe_m, lambda_coupling_m,
                        counter_propagating=counter,
                    )
                else:
                    d_dp_shift = d_dc_shift = 0.0
                rho_ensemble += w_i * _solve_probe_coherence(
                    current_delta_p + d_dp_shift, delta_c + d_dc_shift,
                    omega_p, omega_c, omega_rf, delta_rf,
                    p_gammas, p_dephasing,
                    noise_model=noise_model,
                    transit_rate_per_us=transit_rate,
                    steady_state_backend=ss_backend_default,
                    rf_orientation=rf_orientation,
                )
        transmission_num = transmit(rho_ensemble)
        probe_transmission.append(transmission_num)

        if analytical_enabled:
            # Analytical calculation
            gamma_2_eff = p_gammas[0] + p_dephasing[0]
            analytical_inputs = ({} if doppler_settings is None else
                                 dict(detunings_rad_per_us=p_deltas,
                                      omegas_rad_per_us=(omega_p, omega_c)))
            rho_21_analytic = (solutions.calculate_analytical_rho21(
                scan_offset_rad, acfg, omega_rf, gamma_2_eff=gamma_2_eff,
                rf_orientation=rf_orientation,
                **analytical_inputs,
            ))
            transmission_analytic = transmit(rho_21_analytic)
            probe_transmission_analytic.append(transmission_analytic)


    return detuning_scan_list, np.array(probe_transmission), (np.array(probe_transmission_analytic) if analytical_enabled else None)
