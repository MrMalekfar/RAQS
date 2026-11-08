"""
Create the QuTiP Hamiltonian and collapse operators for the 4-level sensor.

Implements the 4-level cascade system |1⟩→|2⟩→|3⟩→|4⟩ in the rotating wave
approximation (RWA) with ℏ = 1. The system represents a Rydberg electromagnetically
induced transparency (EIT) sensor with:
  - Probe field coupling |1⟩↔|2⟩ at Rabi frequency Ω_p
  - Coupling field coupling |2⟩↔|3⟩ at Rabi frequency Ω_c
  - RF/LO field coupling |3⟩↔|4⟩ at Rabi frequency Ω_rf

Reference: Bussey et al. (2024), "Numerical model of N-level cascade systems"
"""

import numpy as np  # np.sqrt for collapse operator normalization
import qutip as qt  # qt.basis for Fock states, qt.Qobj for operators

from ..lindblad.create_transit_reset_collapse_ops import create_transit_reset_collapse_ops
from ...arc.prepared_decay import PreparedRadiativeGraph


def create_sensor_model(
    omega_p: float,        # Ω_p — Probe Rabi frequency [rad/μs]
    omega_c: float,        # Ω_c — Coupling Rabi frequency [rad/μs]
    omega_rf_lo: float,    # Ω_rf — RF/LO Rabi frequency [rad/μs]
    delta_p: float,        # Δ_p = ω_probe - ω_21 — Probe detuning [rad/μs]
    delta_c: float,        # Δ_c = ω_coupling - ω_32 — Coupling detuning [rad/μs]
    delta_rf: float,       # Δ_rf = ω_rf - ω_43 — RF detuning [rad/μs]
    gammas: list[float] | PreparedRadiativeGraph,
    dephasing_rates_list: list[float],  # [γ_deph2, γ_deph3, γ_deph4] — Dephasing rates [rad/μs]
    transit_rate_per_us: float = 0.0,   # Effective Markovian transit rate Γ_t [1/μs]
    rf_orientation=None,   # RF-DIR-001B/001C: RFTransitionOrientation | None — energy direction of the |3>↔|4> RF edge: spontaneous-emission jump (001B) + sign of Delta_rf in the state-4 rotating-frame diagonal (001C); None → historical build (bit-identical)
) -> tuple:
    """
    Creates the QuTiP Hamiltonian and collapse operators for the 4-level sensor.

    Constructs H = H_detuning + H_int in the rotating frame (ℏ = 1) following
    Bussey et al. (2024). Collapse operators include spontaneous emission jumps
    and pure dephasing (transit, collision, Doppler broadening).

    Parameters
    ----------
    omega_p : float
        Ω_p — Probe Rabi frequency for |1⟩↔|2⟩ transition [rad/μs].
        Ω_p = μ_21 · E_probe / ℏ, where μ_21 is the dipole matrix element.
    omega_c : float
        Ω_c — Coupling Rabi frequency for |2⟩↔|3⟩ transition [rad/μs].
    omega_rf_lo : float
        Ω_rf — RF/LO Rabi frequency for |3⟩↔|4⟩ transition [rad/μs].
    delta_p : float
        Δ_p = ω_probe - ω_21 — Probe detuning from |1⟩→|2⟩ resonance [rad/μs].
    delta_c : float
        Δ_c = ω_coupling - ω_32 — Coupling detuning from |2⟩→|3⟩ resonance [rad/μs].
    delta_rf : float
        Δ_rf = ω_rf - ω_43 — RF detuning from |3⟩→|4⟩ resonance [rad/μs].
    gammas : list[float] or PreparedRadiativeGraph
        [γ_21, γ_32, γ_43] — Spontaneous decay rates for each transition [1/μs].
        (RF-DIR-001B: γ_43 is the energy-downward RF-edge spontaneous rate,
        produced by ``arc.decay_rates.calculate_decay_rates``.)
        A named graph instead builds ONLY its prepared-state internal jumps,
        including nonadjacent E1 edges. It replaces all three legacy jumps;
        outside-space loss/return is not modelled. Its RF orientation is used
        and any conflicting explicit orientation is rejected.
    dephasing_rates_list : list[float]
        [γ_deph2, γ_deph3, γ_deph4] — Pure dephasing rates on levels |2⟩, |3⟩, |4⟩ [rad/μs].
        (DPH-002B: linewidth + collision ONLY; transit must NOT be included here.)
    transit_rate_per_us : float, optional
        Effective Markovian transit (loss/repopulation) rate Γ_t [1/μs].
        When > 0, four reset collapse operators C_j = √Γ_t·|1⟩⟨j| (j = 1..4)
        are appended, implementing Σ_j D[C_j](ρ) = Γ_t(|1⟩⟨1|Tr ρ − ρ):
        coherences decay at Γ_t, excited populations relax at Γ_t, ground
        state is replenished, trace preserved exactly. Default 0.0 keeps the
        exact pre-DPH-002B model (zero-transit invariant).
    rf_orientation : RFTransitionOrientation, optional
        RF-DIR-001B: energy orientation of the ``|3> <-> |4>`` (R1 <-> R2)
        edge, published by ``atomic_config.get_rf_orientation()``.  When
        provided, the RF spontaneous-emission jump is built from its
        upper/lower ladder indices (top→bottom, i.e. ``|4><3|`` for a
        DOWNWARD edge).  ``None`` (default) keeps the historical ladder-index
        jump ``|3><4|`` bit-for-bit (pre-RF-DIR-001B compatibility).

    Returns
    -------
    tuple[qt.Qobj, list[qt.Qobj], qt.Qobj]
        H : qt.Qobj — Total system Hamiltonian (4×4) in rotating frame.
        c_ops : list[qt.Qobj] — Collapse operators for Lindblad master equation.
        obs_probe_coherence : qt.Qobj — σ_{21} = |2⟩⟨1|, the probe coherence operator.
    """
    if isinstance(gammas, PreparedRadiativeGraph):
        rf_orientation = gammas.require_orientation(rf_orientation)

    # --- BASIS STATES ---
    # {|1⟩, |2⟩, |3⟩, |4⟩} — Fock basis for the 4-level cascade system
    # |i⟩ = qt.basis(4, i) is a 4×1 column vector with 1 at position i
    s1, s2, s3, s4 = [qt.basis(4, i) for i in range(4)]

    # --- ATOMIC TRANSITION (LADDER) OPERATORS ---
    # σ_{ij} = |i⟩⟨j| — Transition operator from |j⟩ to |i⟩
    # σ_{21} = |2⟩⟨1|, σ_{12} = σ_{21}^† = |1⟩⟨2|
    s12 = s1 * s2.dag()  # σ_{12} = |1⟩⟨2| — Lowering operator for |2⟩→|1⟩
    s21 = s12.dag()      # σ_{21} = |2⟩⟨1| — Raising operator for |1⟩→|2⟩ (probe coherence)
    s23 = s2 * s3.dag()  # σ_{23} = |2⟩⟨3| — Lowering operator for |3⟩→|2⟩
    s32 = s23.dag()      # σ_{32} = |3⟩⟨2| — Raising operator for |2⟩→|3⟩ (coupling coherence)
    s34 = s3 * s4.dag()  # σ_{34} = |3⟩⟨4| — Lowering operator for |4⟩→|3⟩
    s43 = s34.dag()      # σ_{43} = |4⟩⟨3| — Raising operator for |3⟩→|4⟩ (RF coherence)

    # --- DETUNING HAMILTONIAN (ROTATING FRAME, ℏ = 1) ---
    # RF-DIR-001C - energy-oriented RF rotating frame (oracle:
    # tests/test_rfdir001c_hamiltonian_direction.py, independent two-state RWA).
    # The LABORATORY detuning is orientation-independent:
    #     Delta_rf = w_rf - |nu_signed|            (positive above the carrier)
    # but the co-rotating field exponential flips with the energy direction:
    #     upward   (E4 > E3): |4> rotates at +w_rf -> H44 gains +Delta_rf
    #     downward (E4 < E3): |4> rotates at -w_rf -> H44 gains -Delta_rf
    # rf_orientation=None (legacy) and direction=+1 keep the historical
    # +Delta_rf term bit-identically; only direction=-1 flips its sign.
    # The RF off-diagonal block below is untouched (Hermitian pairing only).
    rf_sign = 1.0
    if rf_orientation is not None:
        _upper = int(rf_orientation.upper_ladder_index)
        _lower = int(rf_orientation.lower_ladder_index)
        if {_upper, _lower} != {3, 4}:
            raise ValueError(
                "rf_orientation must describe the |3> <-> |4> RF edge "
                f"(got upper={_upper}, lower={_lower}). "
                "RF-DIR-001B/F05: no silent substitution."
            )
        if rf_orientation.is_downward:
            rf_sign = -1.0
    # H_det = sum_i Delta_i |i><i| (rotating frame, hbar = 1); Bussey et al.
    # (2024) Eq. (2) diag(0, D_p, D_p+D_c, D_p+D_c+D_rf) is the upward/legacy form.
    H_detuning = (delta_p) * s2.proj() + \
                 (delta_p + delta_c) * s3.proj() + \
                 (delta_p + delta_c + rf_sign * delta_rf) * s4.proj()

    # --- INTERACTION HAMILTONIAN (RWA, ℏ = 1) ---
    # H_int = -½ Σ_i Ω_i (σ_{i+1,i} + σ_{i,i+1})  — Dipole interaction in RWA
    # Following Bussey et al. (2024) Eq. (3): off-diagonals are -Ω/2 for each coupled pair
    H_int = -0.5 * (omega_p * s21 + omega_p * s12 +       # Ω_p coupling |1⟩↔|2⟩
                    omega_c * s32 + omega_c * s23 +       # Ω_c coupling |2⟩↔|3⟩
                    np.conj(omega_rf_lo) * s43 + omega_rf_lo * s34)  # Ω_rf coupling |3⟩↔|4⟩

    # --- TOTAL HAMILTONIAN ---
    H = H_detuning + H_int

    # --- COLLAPSE OPERATORS (LINDBLAD MASTER EQUATION) ---
    # Lindblad superoperator: 𝒟[ρ] = Σ_i (C_i ρ C_i^† - ½{C_i^† C_i, ρ})
    c_ops = []

    # 1. Spontaneous Emission (quantum jumps) — C_γ = √γ |i⟩⟨j|
    if isinstance(gammas, PreparedRadiativeGraph):
        for edge in gammas.edges:
            if edge.rate_per_us > 0:
                jump = qt.basis(4, edge.target) * qt.basis(4, edge.source).dag()
                c_ops.append(np.sqrt(edge.rate_per_us) * jump)
    elif gammas[0] > 0:
        c_ops.append(np.sqrt(gammas[0]) * s12)  # C_21 = √γ_21 · σ_{12} — |2⟩→|1⟩ decay
    if not isinstance(gammas, PreparedRadiativeGraph) and gammas[1] > 0:
        c_ops.append(np.sqrt(gammas[1]) * s23)  # C_32 = √γ_32 · σ_{23} — |3⟩→|2⟩ decay
    if not isinstance(gammas, PreparedRadiativeGraph) and gammas[2] > 0:
        if rf_orientation is None:
            # Legacy (pre-RF-DIR-001B) ladder-index jump — bit-identical.
            c_ops.append(np.sqrt(gammas[2]) * s34)  # C_43 = √γ_43 · σ_{34} — |4⟩→|3⟩ decay
        else:
            # RF-DIR-001B: the RF-edge jump follows the ENERGY order
            # (top→bottom); the direction comes from the published
            # orientation's ladder indices, never from the variable name.
            upper_index = int(rf_orientation.upper_ladder_index)
            lower_index = int(rf_orientation.lower_ladder_index)
            if {upper_index, lower_index} != {3, 4}:
                raise ValueError(
                    "rf_orientation must describe the |3> <-> |4> RF edge "
                    f"(got upper={upper_index}, lower={lower_index}). "
                    "RF-DIR-001B/F05: no silent substitution."
                )
            rf_jump = (qt.basis(4, lower_index - 1)
                       * qt.basis(4, upper_index - 1).dag())  # |lower><upper|
            c_ops.append(np.sqrt(gammas[2]) * rf_jump)

    # 2. Pure Dephasing Operators — C_deph = √γ_deph |i⟩⟨i|
    # Dephasing on level |2⟩ (probe transition broadening from transit, collisions, Doppler)
    if dephasing_rates_list[0] > 0:
        c_ops.append(np.sqrt(dephasing_rates_list[0]) * s2.proj())  # C_deph2 = √γ_deph2 · |2⟩⟨2|
    # Dephasing on level |3⟩ (intermediate state broadening)
    if dephasing_rates_list[1] > 0:
        c_ops.append(np.sqrt(dephasing_rates_list[1]) * s3.proj())  # C_deph3 = √γ_deph3 · |3⟩⟨3|
    # Dephasing on level |4⟩ (Rydberg state broadening)
    if dephasing_rates_list[2] > 0:
        c_ops.append(np.sqrt(dephasing_rates_list[2]) * s4.proj())  # C_deph4 = √γ_deph4 · |4⟩⟨4|

    # 3. Transit loss/repopulation (DPH-002B) — C_j = √Γ_t |1⟩⟨j|, j = 1..4
    # Effective Markovian reset channel: Σ_j D[C_j](ρ) = Γ_t(|1⟩⟨1|Tr ρ − ρ).
    # Transit is represented ONLY here (never inside dephasing_rates_list,
    # which would double-count the mechanism).
    if transit_rate_per_us is not None and transit_rate_per_us > 0:
        c_ops.extend(create_transit_reset_collapse_ops(transit_rate_per_us))

    # --- OBSERVABLE: PROBE COHERENCE ---
    # ⟨σ_{21}⟩ = Tr[ρ · σ_{21}] — Probe coherence used to compute susceptibility
    # χ ∝ ⟨σ_{21}⟩ / Ω_p  — Linear susceptibility from probe coherence (Bussey et al. Eq. 8)
    obs_probe_coherence = s21

    return H, c_ops, obs_probe_coherence
