"""
Time-domain heterodyne simulations.
"""

import numpy as np
import qutip as qt
from qutip import steadystate, mesolve
from ..exceptions import PhysicsCalculationError, PhysicsError
from ..physics.hamiltonian import create_sensor_model
from ..arc.prepared_decay import PreparedRadiativeGraph
from ..utils.numerical import extract_if_component
from ..utils.validation import validate_steady_state
from ..utils.density_validation import validate_density_matrix
from .noise_seam import apply_noise_seam
from ..config import parameters as params


def atomic_if_polarity(rf_orientation):
    """RF-DIR-001D: polarity of the atomic IF-coefficient chain.

    Derived ONLY from the established RF-DIR-001A energy orientation
    (``rf_orientation.direction``); it is not an independent flag:

        +1.0  legacy / upward edge (direction = +1) -> historical chain
        -1.0  downward edge (direction = -1) -> the co-rotating RWA keeps
              the conjugate field component, which conjugates the
              recovered IF coefficient: the time-exponent sign flips,
              the injected / spatial phase on ``omega_s`` does not.

    ``None`` keeps the historical +1.0 chain bit-identically.
    """
    if rf_orientation is not None and rf_orientation.direction < 0:
        return -1.0
    return 1.0


def signal_phase_factor(t, signed_if_mhz, rf_orientation=None):
    """exp(+i * 2*pi * sigma * f_IF * t), sigma = atomic_if_polarity.

    ``signed_if_mhz`` is the SIGNED laboratory IF (f_signal - f_LO) in
    MHz; the time-exponent sign follows the RF-edge energy orientation,
    never an independent flag (RF-DIR-001D section 3).
    """
    sigma = atomic_if_polarity(rf_orientation)
    return np.exp(1j * 2 * np.pi * (sigma * signed_if_mhz) * t)


def signal_coefficient(t, omega_s, signed_if_mhz, rf_orientation=None):
    """H_43(t) coefficient of the time-dependent RF signal drive.

        coefficient(t) = omega_s * exp(+i * 2*pi * sigma * f_IF * t)

    ``omega_s`` is the complex signal Rabi amplitude carrying the
    injected / spatial phase (|Omega| * exp(+i*phi)); the -0.5 * |4><3|
    operator factor lives in the worker.  For rf_orientation None and
    direction=+1 this equals the historical
    ``omega_s * np.exp(1j * 2 * np.pi * if * t)`` bit-identically.
    """
    return omega_s * signal_phase_factor(t, signed_if_mhz, rf_orientation)


def signal_coefficient_dag(t, omega_s, signed_if_mhz, rf_orientation=None):
    """Hermitian partner H_34(t) = conj(H_43(t)) by construction, so the
    time-dependent RF drive satisfies H(t) = H(t)^dagger at every t
    (RF-DIR-001D section 15), for every orientation and complex omega_s.
    """
    return np.conj(signal_coefficient(t, omega_s, signed_if_mhz, rf_orientation))


def run_single_heterodyne_worker(
        i,                  # Element index
        omega_s_complex,    # Signal Rabi Freq (Complex, includes phase)
        tlist,              # Time array
        if_freq,            # IF Frequency
        # PHYSICS PARAMETERS (Floats)
        omega_p, omega_c, omega_lo,
        delta_p, delta_c, delta_rf,
        gammas, dephasing_rates_list,
        noise_model=None,          # RF-004: optional noise model (None default keeps the worker picklable)
        transit_rate_per_us=None,  # DPH-002B: effective transit reset rate [1/μs]; None → config default
        rf_orientation=None,       # RF-DIR-001B: energy direction of the RF-edge jump (picklable frozen dataclass); None → historical ladder-index build
        rf_drive=None,             # NOISE-005: precomputed common RF envelope, never a stochastic callback
        _coherence_only=False,     # DOP-TD-001: certified complex trace before IF
        temporal_ode=None,         # NUM-ODE-002: explicit immutable temporal controls
    ):
    """
    Worker function that builds the model locally using passed floats.
    No ARC imports here!
    """
    stage = "model construction"
    try:
        if isinstance(gammas, PreparedRadiativeGraph):
            # The signal's rotating-frame phase must use the same orientation as H/collapse.
            rf_orientation = gammas.require_orientation(rf_orientation)
        if temporal_ode is not None:
            from ..config.run_numerics import TemporalODE
            if not isinstance(temporal_ode,TemporalODE):
                raise PhysicsCalculationError('NUM-ODE-002: typed temporal ODE controls required.')
        if not _coherence_only:
            print(f"Simulating element {i+1} on CPU...")

        # Convert back to numpy arrays for computation
        tlist_np = np.array(tlist)
        drive_evaluator = None
        if rf_drive is not None:
            if noise_model is not None:
                raise PhysicsCalculationError('NOISE-005: legacy Hamiltonian noise seam cannot be combined with explicit RF drive without a joint model contract.')
            from .rf_drive_noise import RFDriveRealization, RFDriveEvaluator
            if not isinstance(rf_drive, RFDriveRealization) or rf_drive.times_us[-1] <= tlist_np[-1]:
                raise PhysicsCalculationError('NOISE-005: valid RF realization with explicit padded horizon required.')
            # Only concrete immutable records use the optimization. Existing
            # specializations retain their original virtual-method dispatch.
            drive_evaluator = RFDriveEvaluator(rf_drive) if type(rf_drive) is RFDriveRealization else rf_drive

        # DPH-002B: resolve the transit reset rate (lazy config default)
        if transit_rate_per_us is None:
            from ..physics.lindblad import calculate_transit_rate_per_us
            transit_rate_per_us = calculate_transit_rate_per_us()

        # 1. Build Static Hamiltonian & Collapse Ops (Pure Numerical)
        H_static, c_ops, obs = create_sensor_model(
            omega_p, omega_c, omega_lo,
            delta_p, delta_c, delta_rf,
            gammas, dephasing_rates_list,
            transit_rate_per_us=transit_rate_per_us,
            rf_orientation=rf_orientation,
        )
        # RF-004 seam: strict identity for noise_model=None (baseline-safe).
        # The seam stays INSIDE the worker so Qobjs never cross process
        # boundaries — plain floats, frozen orientation, and optionally a
        # frozen numeric RFDriveRealization cross the boundary, never Qobjs.
        # Documented seam contract: returns exactly (H, c_ops); the observable
        # is read-only context and `obs` from `create_sensor_model` is kept.
        stage = "noise seam"
        H_static, c_ops = apply_noise_seam(
            H_static, c_ops, obs=obs, noise_model=noise_model
        )

        # 2. Calculate Initial Steady State
        # This is fast for a 4x4 system (microseconds), faster than pickling Qobjs.
        stage = "initial steady state"
        rho0 = steadystate(H_static, c_ops)
        # NUM-SS-001: rho0 is a steady-state density matrix that is consumed
        # immediately as the initial condition of mesolve and therefore feeds
        # the sensor output.  It passes the SAME fail-loud certification the
        # steady-state backends use, measured against the ORIGINAL
        # Liouvillian of the very (H_static, c_ops) pair the solver was
        # given.  Nothing is repaired or replaced on rejection (F05).
        stage = "steady-state certification"
        validate_steady_state(
            rho0, qt.liouvillian(H_static, list(c_ops)),
            context="run_single_heterodyne_worker(rho0)", backend="qutip",
        )

        # 3. Define Time-Dependent Signal Hamiltonian
        # Signal field couples Rydberg levels |3> <-> |4>
        s34 = qt.basis(4, 2) * qt.basis(4, 3).dag()
        H_S_op = -0.5 * s34.dag()
        H_S_op_dag = -0.5 * s34

        # Time-dependent coefficients (RF-DIR-001D)
        # Use python complex/float types to ensure pickling safety.
        # The single exponent construction lives in signal_phase_factor;
        # the time-exponent sign follows the RF-edge energy orientation
        # (sigma = atomic_if_polarity(rf_orientation)) and the dag partner
        # is the exact conjugate, so H(t) = H(t)^dagger at every t.
        w_sig = complex(omega_s_complex)
        if_val = float(if_freq)

        def H_S_coeff(t, args=None):
            nominal = signal_coefficient(t, w_sig, if_val, rf_orientation)
            if rf_drive is not None:
                from .rf_drive_noise import drive_signal_coefficient
                return drive_signal_coefficient(t, nominal, drive_evaluator)
            return nominal

        def H_S_coeff_dag(t, args=None):
            if rf_drive is not None:
                return np.conj(H_S_coeff(t, args))
            return signal_coefficient_dag(t, w_sig, if_val, rf_orientation)

        # Total Time-Dependent Hamiltonian
        H_t = [H_static, [H_S_op, H_S_coeff], [H_S_op_dag, H_S_coeff_dag]]
        if rf_drive is not None:
            # H_static already contains nominal LO. Add only its colored
            # correction, with the original Hermitian operator pairing.
            lo_operator = -.5*(np.conj(omega_lo)*s34.dag()+omega_lo*s34)
            def lo_correction(t, args=None):
                amplitude, _, _ = drive_evaluator.at(t)
                return amplitude-1.
            H_t.append([lo_operator, lo_correction])
            if any(rf_drive.probe_frequency_hz) or any(rf_drive.coupling_frequency_hz):
                from ..physics.hamiltonian.laser_frequency_noise import laser_frequency_terms
                H_t.extend(laser_frequency_terms(drive_evaluator))
            if any(v != 1. for v in rf_drive.probe_intensity+rf_drive.coupling_intensity):
                from ..physics.hamiltonian.laser_intensity_noise import laser_intensity_terms
                H_t.extend(laser_intensity_terms(drive_evaluator, omega_p, omega_c))

        # 4. Run Time Evolution (BDF, stiff-stable).  NUM-TD-001: an e_ops
        # callback certifies the density matrix at EVERY requested output time
        # (trace / Hermiticity / positivity, no stationarity) BEFORE qt.expect
        # turns it into the observable, so an invalid temporal state can never
        # become heterodyne / AoA data.  Temporal GRID IDENTITY: each callback
        # must deliver exactly tlist[k] for its running output index k
        # (ordered, unique, on-grid, finite) — a duplicated, reordered,
        # off-grid or non-finite timestamp, or an excess callback past the
        # grid end, is rejected BEFORE density certification and BEFORE
        # qt.expect, so the certified trace provably lives on the dispatched
        # output grid.  Exact equality (no tolerance) is the installed-stack
        # contract: QuTiP 5.2.1 passes tlist[0] verbatim (solver_base.run)
        # and clamps every later output time to the requested tlist point
        # (Integrator.run -> scipy zvode, itask=1 overshoot-and-interpolate).
        # normalize_output=False is set explicitly so certification sees the
        # RAW solver output (QuTiP's default True would hide ODE trace drift).
        # Preserve legacy options unless typed profile controls are declared.
        stage = "mesolve"
        options = qt.Options(method='bdf', nsteps=5000, normalize_output=False)
        if temporal_ode is not None:
            options.update(temporal_ode.options())
        certified = [0]

        def _certified_observable(t, state):
            output_index = certified[0]
            if output_index >= tlist_np.shape[0]:
                raise PhysicsCalculationError(
                    f"run_single_heterodyne_worker: element {i} received an "
                    f"EXCESS certification callback at output index "
                    f"{output_index} (actual t={float(t)!r}us) after the "
                    f"{tlist_np.shape[0]}-point output grid was completed; "
                    f"expected no further timestamp (NUM-TD-001).")
            expected_t = tlist_np[output_index]
            if not np.isfinite(t) or t != expected_t:
                raise PhysicsCalculationError(
                    f"run_single_heterodyne_worker: element {i} output index "
                    f"{output_index} expected t={float(expected_t)!r}us but "
                    f"the certification callback received t={float(t)!r}us; "
                    f"the temporal grid requires an ordered, unique, on-grid "
                    f"and finite timestamp (NUM-TD-001).")
            validate_density_matrix(
                state,
                context=f"run_single_heterodyne_worker(rho[t={t:g}us], "
                        f"element {i})",
                expected_dim=4,
            )
            certified[0] += 1
            return qt.expect(obs, state)

        result = mesolve(H_t, rho0, tlist, c_ops, [_certified_observable],
                         options=options)

        # NUM-TD-001: the callback must have certified the FULL output grid.
        # A solver that returns finite expectations without invoking it (a
        # mock, or a solver that skips states) is NOT certified and must fail.
        if certified[0] != len(tlist_np):
            raise PhysicsCalculationError(
                f"run_single_heterodyne_worker: element {i} certified only "
                f"{certified[0]} of {len(tlist_np)} output states; the temporal "
                f"density matrix was NOT certified across the full grid "
                f"(NUM-TD-001).")

        # 5. Extract Data — validate the observable BEFORE np.imag so a
        # non-finite real component is never silently discarded (F05-WORKER-001).
        stage = "observable extraction"
        expect = getattr(result, "expect", None)
        if expect is None or len(expect) < 1:
            raise PhysicsCalculationError(
                f"run_single_heterodyne_worker: element {i} produced no "
                f"observable expectation (result.expect missing or empty); the "
                f"worker cannot continue (F05-WORKER-001).")
        trace = np.asarray(expect[0])
        if trace.ndim != 1 or trace.shape[0] != tlist_np.shape[0]:
            raise PhysicsCalculationError(
                f"run_single_heterodyne_worker: element {i} observable trace "
                f"length {trace.shape} does not match the dispatched time grid "
                f"({tlist_np.shape[0]},); the result is malformed "
                f"(F05-WORKER-001).")
        if trace.dtype == np.bool_ or trace.dtype == object \
                or trace.dtype.kind in ("U", "S", "V") \
                or not np.issubdtype(trace.dtype, np.number):
            raise PhysicsCalculationError(
                f"run_single_heterodyne_worker: element {i} observable trace is "
                f"non-numeric or boolean (dtype {trace.dtype}); it is rejected "
                f"before np.imag (F05-WORKER-001).")
        if not np.all(np.isfinite(trace)):
            raise PhysicsCalculationError(
                f"run_single_heterodyne_worker: element {i} observable trace "
                f"contains non-finite values; a non-finite real component must "
                f"NOT be discarded via np.imag (F05-WORKER-001).")
        if _coherence_only:
            return trace
        signal_trace = np.imag(trace)

        # Extract Amplitude and Phase at IF frequency
        stage = "IF extraction"
        complex_amp, xf, yf = extract_if_component(
            signal_trace,
            tlist,
            if_val,
            params.ATOMIC_TRANSIENT_DELAY_US
        )
        print(f"DEBUG: extract_if_component returned: {type(complex_amp)}, {type(xf)}, {type(yf)}")
        results = (complex_amp, xf, yf, i, signal_trace)
        #print(f"DEBUG: Worker returning tuple of length {len(results)}")
        return results
    except PhysicsError:
        # F05-WORKER-001: every designated physics failure (PhysicsError and
        # its subclasses PhysicsCalculationError / PhysicsConfigurationError)
        # MUST abort the run, propagating UNCHANGED with its type, message and
        # cause chain.  Converting it into a None-filled tuple would substitute
        # data for a failure and hand `None` to phase extraction — the fallback
        # F05 forbids.
        raise
    except Exception as exc:
        # F05-WORKER-001: any other failure also aborts visibly, wrapped with
        # the element index and failing stage and chaining the original cause.
        # No retry, no solver-settings change, no state/signal substitution.
        raise PhysicsCalculationError(
            f"run_single_heterodyne_worker: element {i} failed at stage "
            f"'{stage}': {exc}") from exc
