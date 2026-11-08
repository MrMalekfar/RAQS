"""RF-004 focused regression tests: no-op noise-seam contract.

Scope: validates ONLY the architectural seam ``simulation.noise_seam`` —
not any noise physics (none exists by mandate). Tests 1, 2 and 5 build
synthetic two-level Qobjs and stay lightweight; Tests 3 and 4 load the real
configuration/signature surface once to prove no-global-mutation and
backward-compatible caller contracts.
"""
import copy
import inspect

import pytest
import qutip as qt

from rydberg_aoa.simulation.noise_seam import apply_noise_seam

# --- Synthetic stand-ins mirroring the create_sensor_model outputs ----------
# σ = |0⟩⟨1| — lowering operator (decay-jump shape √γ·|i⟩⟨j|)
_SIGMA_DOWN = qt.basis(2, 0) * qt.basis(2, 1).dag()
# σ† = |1⟩⟨0| — its adjoint, standing in for the coherence observable σ₂₁ role
_OBS = _SIGMA_DOWN.dag()


def _make_model(gamma_decay=0.5, gamma_deph=0.25):
    """
    Synthetic (H, c_ops) with the same mathematical structure as the real
    4-level model: diagonal detuning + RWA interaction Hamiltonian, and a
    Lindblad collapse set mixing decay jumps and pure-dephasing projectors.
    """
    # H = Δ·|1⟩⟨1| − (Ω/2)(σ + σ†) — diagonal + Hermitian off-diagonal form
    delta, omega = 0.3, 2.0
    H = (
        delta * qt.basis(2, 1).proj()
        - 0.5 * omega * (_SIGMA_DOWN + _SIGMA_DOWN.dag())
    )
    # C_decay = √γ · |0⟩⟨1| ; C_deph = √γφ · |1⟩⟨1| — dissipator channels
    c_ops = [
        gamma_decay ** 0.5 * _SIGMA_DOWN,
        gamma_deph ** 0.5 * qt.basis(2, 1).proj(),
    ]
    return H, c_ops


# ---------------------------------------------------------------------------
# Test 1 — default (omitted) behavior is strict identity
# ---------------------------------------------------------------------------
def test_default_is_strict_identity():
    """Omitting noise_model returns the same objects: no copies, no reorder."""
    H, c_ops = _make_model()

    H_out, c_out = apply_noise_seam(H, c_ops)

    # Identity (object) checks prove the Liouvillian input is untouched —
    # stronger than value equality; guarantees bit-identical solver input.
    assert H_out is H
    assert c_out is c_ops           # the list container itself, not rebuilt
    assert len(c_out) == 2          # channel count unchanged (no added c_ops)
    assert c_out[0] is c_ops[0] and c_out[1] is c_ops[1]


# ---------------------------------------------------------------------------
# Test 2 — explicit no-noise configuration equals omission
# ---------------------------------------------------------------------------
def test_explicit_none_equals_omission():
    """Passing noise_model=None explicitly reproduces the omitted-kwarg result."""
    H, c_ops = _make_model()

    omitted = apply_noise_seam(H, c_ops, obs=_OBS)
    explicit = apply_noise_seam(H, c_ops, obs=_OBS, noise_model=None)

    assert explicit[0] is omitted[0] and explicit[1] is omitted[1]
    assert explicit[0] is H         # and both are the original H


# ---------------------------------------------------------------------------
# Test 3 — no global/config mutation from seam usage
# ---------------------------------------------------------------------------
def test_no_global_or_config_mutation():
    """Seam usage (None and pass-through probe) leaves all config state intact."""
    # Real configuration modules — heavy one-time ARC init, reused session-wide
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.config import atomic_config as acfg
    from rydberg_aoa.config.laser_rf_config import LASER_PARAMS

    # Deep-value snapshot of the state the seam consumers rely upon
    snap_levels = copy.deepcopy(acfg.ATOMIC_LEVELS)
    snap_laser = copy.deepcopy(LASER_PARAMS)
    snap_sim = (
        params.DOPPLER_ENABLED, params.DOPPLER_N_CLASSES,
        params.SCAN_RANGE_MHZ, params.N_SCAN, params.N_ELEMENTS,
    )
    snap_units = (
        params.CELL_LENGTH, params.US_TO_S_FACTOR, acfg.OMEGA_P, acfg.GAMMA_21,
    )

    def probe(h, c, o):             # interface-only probe: forwards, mutates nothing
        return h, c

    H, c_ops = _make_model()
    apply_noise_seam(H, c_ops, noise_model=None)
    apply_noise_seam(H, c_ops, obs=_OBS, noise_model=probe)

    # Deep copies isolate even in-place edits of nested config structures
    assert copy.deepcopy(acfg.ATOMIC_LEVELS) == snap_levels
    assert copy.deepcopy(LASER_PARAMS) == snap_laser
    assert (
        params.DOPPLER_ENABLED, params.DOPPLER_N_CLASSES,
        params.SCAN_RANGE_MHZ, params.N_SCAN, params.N_ELEMENTS,
    ) == snap_sim
    assert (
        params.CELL_LENGTH, params.US_TO_S_FACTOR,
        acfg.OMEGA_P, acfg.GAMMA_21,
    ) == snap_units


# ---------------------------------------------------------------------------
# Test 4 — backward-compatible caller contracts (existing callers unchanged)
# ---------------------------------------------------------------------------
def test_existing_caller_contracts_unchanged():
    """Simulation consumers gained ONLY trailing optional parameters.

    DPH-002B appended ``transit_rate_per_us`` after the existing
    ``noise_model`` position. The backward-compatibility contract verified
    here (RF-004 + DPH-002B + DOP-003S):

    1. all pre-DPH-002B positional parameters retain the same ordering;
    2. ``noise_model`` retains its previous positional index;
    3. ``noise_model`` default remains ``None``;
    4. ``transit_rate_per_us`` is appended after it;
    5. old callers that supply ``noise_model`` positionally still resolve
       to ``noise_model`` (never ``transit_rate_per_us``);
    6. return contracts remain unchanged (arity covered by the RF-004
       caller tests and the DPH-002B worker/steady-state caller tests);
    7. trailing optionals keep their exact documented order: RF-DIR-001B's
       ``rf_orientation`` follows the original numerical switches; before it sit the
       DOP-003S ``steady_state_backend`` (steady-state callables; both values
       solve the identical master equation, see test_dop003s_direct_solver.py)
       and the DPH-002B ``transit_rate_per_us``. Pre-RF-DIR-001B this check
       claimed ``steady_state_backend`` was last for BOTH steady-state
       callables — true only for ``_solve_probe_transmission``; the PHY-SW-001
       switches already sat after it in ``simulate_spectrum``. LAB-007 adds
       three trailing profile options without moving any original position.
    """
    import rydberg_aoa.simulation.steady_state as ss        # noqa: F401
    import rydberg_aoa.simulation.time_domain as td         # noqa: F401

    for func in (
        ss.simulate_spectrum,
        ss._solve_probe_transmission,
        td.run_single_heterodyne_worker,
    ):
        spec = inspect.signature(func)
        names = list(spec.parameters)
        # (2)+(3) noise_model keeps its position and None default
        assert "noise_model" in names
        assert spec.parameters["noise_model"].default is None
        # (4) transit_rate_per_us appended directly after it
        assert "transit_rate_per_us" in names
        assert names.index("transit_rate_per_us") == names.index("noise_model") + 1
        # RF-DIR-001B: the new seam is opt-in (default None)
        assert "rf_orientation" in names
        assert spec.parameters["rf_orientation"].default is None
        # (7) Original ordering plus explicitly declared trailing extensions:
        #     worker:            [..., transit_rate_per_us, rf_orientation]
        #     transmission:      [..., transit_rate_per_us,
        #                         steady_state_backend, rf_orientation]
        #     simulate_spectrum: [..., steady_state_backend, doppler_enabled,
        #                         transit_enabled, laser_linewidth_dephasing_enabled,
        #                         collision_dephasing_enabled, rf_orientation]
        if func is td.run_single_heterodyne_worker:
            assert names[-2:] == ["transit_rate_per_us", "rf_orientation"]
        elif func is ss._solve_probe_transmission:
            assert names[-3:] == [
                "transit_rate_per_us",
                "steady_state_backend",
                "rf_orientation",
            ]
        else:
            legacy_names = [
                "omega_rf", "detuning_list_mhz", "p_omegas", "p_deltas",
                "p_gammas", "p_dephasing", "absorption_const", "noise_model",
                "transit_rate_per_us", "steady_state_backend", "doppler_enabled",
                "transit_enabled", "laser_linewidth_dephasing_enabled",
                "collision_dephasing_enabled", "rf_orientation"]
            extensions = ["doppler_settings", "transmission_prefactors", "analytical_enabled"]
            assert names == legacy_names + extensions
            for name in legacy_names[7:]:
                assert spec.parameters[name].default is None
            assert [spec.parameters[name].default for name in extensions] == [None, None, True]
            values = [object() for _ in legacy_names]
            legacy_bound = spec.bind(*values)
            assert list(legacy_bound.arguments) == legacy_names
            for name, value in zip(legacy_names, values):
                assert legacy_bound.arguments[name] is value
            legacy_bound.apply_defaults()
            assert [legacy_bound.arguments[name] for name in extensions] == [None, None, True]
        # (1)+(5) positional binding: supplying exactly one extra positional
        # after the pre-DPH-002B required arguments binds to noise_model
        n_required = sum(
            1 for p in spec.parameters.values() if p.default is inspect.Parameter.empty
        )
        sentinel = object()
        bound = spec.bind(*([sentinel] * n_required + [sentinel]))
        assert bound.arguments["noise_model"] is sentinel
        assert "transit_rate_per_us" not in bound.arguments


# ---------------------------------------------------------------------------
# Test 5 — future extension point (interface contract only, NO noise physics)
# ---------------------------------------------------------------------------
def test_extension_point_accepts_injected_callable():
    """Injected model receives (H, c_ops, obs) verbatim; its valid 2-tuple
    return becomes exactly the (H', c') pair handed to the solver boundary."""
    H, c_ops = _make_model()

    # Pass-through model: proven the seam forwards inputs by identity and
    # dispatches its output unmodified — while applying zero physics here.
    received = {}

    def passthrough_model(h, c, o):
        received.update(h=h, c=c, o=o)
        return h, c

    H_out, c_out = apply_noise_seam(H, c_ops, obs=_OBS, noise_model=passthrough_model)
    assert received == {"h": H, "c": c_ops, "o": _OBS}
    assert H_out is H and c_out is c_ops

    # Output-replacement contract: an injected model's own objects are what
    # propagate downstream (this is how future noise terms would reach QuTiP).
    H_alt = H * 1.0                 # value-equal, object-distinct stand-in
    c_alt = [c_ops[0]]              # container produced by the "model" itself

    def replacing_model(h, c, o):
        return H_alt, c_alt

    H_out2, c_out2 = apply_noise_seam(H, c_ops, obs=None, noise_model=replacing_model)
    assert H_out2 is H_alt and c_out2 is c_alt


def test_seam_rejects_malformed_configurations_fail_loud():
    """Malformed noise configs raise instead of silently corrupting the
    Liouvillian reaching QuTiP (fail-loud, per the RF-003 philosophy)."""
    H, c_ops = _make_model()

    with pytest.raises(TypeError):      # non-callable configuration
        apply_noise_seam(H, c_ops, noise_model="dephasing")

    with pytest.raises(TypeError):      # callable returning a non-2-tuple
        apply_noise_seam(H, c_ops, noise_model=lambda h, c, o: h)
