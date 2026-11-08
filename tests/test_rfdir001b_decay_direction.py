"""RF-DIR-001B — energy-ordered Rydberg-pair decay (production contract).

Scope of this file (the B ticket ONLY):

* ``decay_rates.resolve_rydberg_orientation`` / ``calculate_decay_rates``:
  the RF-edge decay follows the ENERGY order resolved from the signed ARC
  transition frequency (RF-DIR-001A) — for the production pair the rate is
  R1(26D5/2) -> R2(27P3/2) at ARC's default temperature (0.0 K, one-way);
* ``create_sensor_model(rf_orientation=...)``: the RF spontaneous-emission
  jump is ``|4><3|`` for a DOWNWARD edge and ``|3><4|`` for a synthetic
  UPWARD orientation; ``None`` stays bit-identical to the historical build;
* an independent analytic population-flow oracle (exact master equation for
  a single jump) plus its falsification partner;
* unit conversion s^-1 -> 1/us by /1e6 with NO 2*pi;
* the guarded ``atomic_config.get_rf_orientation()`` reader and the solver
  seam forwarding.

RF-DIR-001C has since landed (rotating-frame RF diagonal sign): the scope
guard below pins that a DOWNWARD build may differ ONLY in |4><4|.  Still
deferred: RF-DIR-001D signed IF / heterodyne / AoA phase convention.
"""

import numpy as np
import pytest
import qutip as qt

from rydberg_aoa.arc.decay_rates import (
    calculate_decay_rates,
    resolve_rydberg_orientation,
)
from rydberg_aoa.arc.rf_orientation import resolve_rf_orientation
from rydberg_aoa.arc.types import AtomicLevel
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.exceptions import PhysicsConfigurationError
from rydberg_aoa.physics.hamiltonian.create_sensor_model import create_sensor_model

# ARC's SIGNED frequency for the production pair (Rb85: 26D5/2 -> 27P3/2),
# taken in LADDER order.  Negative => E(rydberg2) < E(rydberg1): DOWNWARD.
NU_R1_TO_R2_HZ = -1.3264910772876007e11
NU_R2_TO_R1_HZ = +1.3264910772876007e11

DOWNWARD = resolve_rf_orientation(NU_R1_TO_R2_HZ)
UPWARD = resolve_rf_orientation(NU_R2_TO_R1_HZ)

#: Live ARC rate for R1 -> R2 at the production (default) temperature.
R1_TO_R2_RATE_S = 47.944361878374565

GROUND = {"n": 5, "l": 0, "j": 0.5}
EXCITED = {"n": 5, "l": 1, "j": 1.5}
R1 = {"n": 26, "l": 2, "j": 2.5}
R2 = {"n": 27, "l": 1, "j": 1.5}


def _arc_atom():
    """Live ARC atom or a clean skip (mirrors test_rfdir001a)."""
    arc = pytest.importorskip("arc")
    try:
        return arc.Rubidium85()
    except Exception as exc:  # environment-dependent (ARC data unavailable)
        pytest.skip(f"ARC atomic database unavailable: {exc}")


def _rf_collapse_full(rf_orientation, gamma=0.3):
    """The single RF collapse operator built by ``create_sensor_model``.

    Only the RF channel is enabled (``gammas = [0, 0, gamma]``), zero
    dephasing, zero transit and zero couplings, so the list contains exactly
    that one operator.
    """
    _, c_ops, _ = create_sensor_model(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        [0.0, 0.0, gamma], [0.0, 0.0, 0.0],
        rf_orientation=rf_orientation,
    )
    assert len(c_ops) == 1
    return c_ops[0].full()


# ---------------------------------------------------------------------------
# 1. Live ARC: the production pair is DOWNWARD and the rate is one-way
# ---------------------------------------------------------------------------

def test_live_arc_pair_resolves_downward_through_the_primitive():
    """ARC's signed frequency for the production pair -> DOWNWARD (|4><3|)."""
    atom = _arc_atom()
    orientation = resolve_rydberg_orientation(atom, R1, R2)

    nu = atom.getTransitionFrequency(26, 2, 2.5, 27, 1, 1.5)
    assert orientation == resolve_rf_orientation(nu)   # same single source
    assert orientation.signed_frequency_hz == pytest.approx(NU_R1_TO_R2_HZ, rel=1e-9)
    assert orientation.is_downward and not orientation.is_upward
    assert orientation.decay_operator_name == "s43"
    assert (orientation.upper_ladder_index,
            orientation.lower_ladder_index) == (3, 4)
    assert orientation.carrier_hz == pytest.approx(132.64910772876e9, rel=1e-6)

    # dict inputs are accepted exactly like AtomicLevel inputs
    assert resolve_rydberg_orientation(atom, dict(R1), dict(R2)) == orientation
    assert resolve_rydberg_orientation(
        atom, AtomicLevel(**R1), AtomicLevel(**R2)
    ) == orientation


def test_live_arc_rate_is_one_way_energy_downward_at_the_pinned_temperature():
    """Production temperature = ARC default (0.0 K): only R1 -> R2 is active."""
    atom = _arc_atom()

    down_default = atom.getTransitionRate(26, 2, 2.5, 27, 1, 1.5)
    down_explicit = atom.getTransitionRate(
        26, 2, 2.5, 27, 1, 1.5, temperature=0.0
    )
    up = atom.getTransitionRate(27, 1, 1.5, 26, 2, 2.5)

    # The production calls pass NO temperature => ARC's default. Pin it.
    assert down_default == pytest.approx(down_explicit, rel=1e-12)
    assert down_default == pytest.approx(R1_TO_R2_RATE_S, rel=1e-6)
    assert down_default > 0.0
    # No thermal reverse channel at the production temperature (one-way).
    assert up == 0.0

    gamma_21, gamma_32, gamma_43 = calculate_decay_rates(
        atom, GROUND, EXCITED, R1, R2
    )
    assert gamma_21 > 0.0 and gamma_32 > 0.0
    # gamma_43 is the ENERGY-DOWNWARD spontaneous rate: s^-1 / 1e6, no 2*pi.
    assert gamma_43 == pytest.approx(down_default / 1e6, rel=1e-9)
    assert gamma_43 > 1e-6   # genuinely above the removed floor: unchanged by F05-DECAY-001
    assert gamma_43 != pytest.approx(down_default * 2 * np.pi / 1e6, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Direction logic without ARC: signed frequency -> evaluated transition
# ---------------------------------------------------------------------------

def test_calculate_decay_rates_uses_the_signed_frequency_and_energy_order(monkeypatch):
    """The single orientation source is the primitive + signed ARC frequency."""
    import rydberg_aoa.arc.decay_rates as dr

    seen_nu = []
    real_resolver = dr.resolve_rf_orientation

    def _spy(nu):
        seen_nu.append(nu)
        return real_resolver(nu)

    monkeypatch.setattr(dr, "resolve_rf_orientation", _spy)

    class _FakeAtom:
        def __init__(self, nu):
            self.nu = nu
            self.rate_calls = []

        def getTransitionFrequency(self, *args):
            return self.nu

        def getTransitionRate(self, n1, l1, j1, n2, l2, j2):
            self.rate_calls.append((n1, l1, j1, n2, l2, j2))
            if (n1, n2) == (26, 27):
                return 42.0            # R1 -> R2 (downward for nu < 0)
            if (n1, n2) == (27, 26):
                return 24.0            # R2 -> R1 (downward for nu > 0)
            return 7.0

    # --- downward edge: gamma_43 comes from the R1 -> R2 emission rate
    atom = _FakeAtom(-5.0)
    g21, g32, g43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)
    assert seen_nu[-1] == -5.0                     # signed frequency, untouched
    assert atom.rate_calls[-1] == (26, 2, 2.5, 27, 1, 1.5)
    assert g43 == pytest.approx(42.0 / 1e6)
    assert g21 == pytest.approx(7.0 / 1e6)
    assert g32 == pytest.approx(7.0 / 1e6)

    # --- synthetic upward edge: the same resolver flips the evaluation order
    atom_up = _FakeAtom(+5.0)
    _, _, g43_up = calculate_decay_rates(atom_up, GROUND, EXCITED, R1, R2)
    assert seen_nu[-1] == +5.0
    assert atom_up.rate_calls[-1] == (27, 1, 1.5, 26, 2, 2.5)
    assert g43_up == pytest.approx(24.0 / 1e6)

    # --- F05-DECAY-001: a valid zero rate is preserved exactly (no floor)
    class _ZeroRateAtom(_FakeAtom):
        def getTransitionRate(self, *args):
            return 0.0

    _, _, g43_zero = calculate_decay_rates(
        _ZeroRateAtom(-5.0), GROUND, EXCITED, R1, R2
    )
    assert g43_zero == 0.0          # the artificial 1e-6 floor is removed


# ---------------------------------------------------------------------------
# 3. The RF collapse operator follows the ENERGY order (c_ops)
# ---------------------------------------------------------------------------

def test_default_call_is_bit_identical_to_upward():
    """rf_orientation=None == the historical ladder-index jump, bit-for-bit."""
    gamma = 0.3
    default = _rf_collapse_full(None, gamma) / np.sqrt(gamma)
    upward = _rf_collapse_full(UPWARD, gamma) / np.sqrt(gamma)

    np.testing.assert_array_equal(default, upward)

    expected = np.zeros((4, 4))
    expected[2, 3] = 1.0  # |3><4| — the historical ladder-index jump
    np.testing.assert_array_equal(upward, expected)


def test_downward_selects_the_upper_to_lower_jump():
    """A downward edge emits from |3> into |4>: the jump is |4><3|."""
    gamma = 0.3
    downward = _rf_collapse_full(DOWNWARD, gamma) / np.sqrt(gamma)

    expected = np.zeros((4, 4))
    expected[3, 2] = 1.0  # |4><3| — energy top -> bottom
    np.testing.assert_allclose(downward, expected, rtol=0.0, atol=1e-15)

    # Not a silent no-op: the direction genuinely differs from the default.
    assert not np.allclose(
        downward, _rf_collapse_full(None, gamma) / np.sqrt(gamma)
    )


def test_orientation_must_describe_the_rf_edge():
    """F05: an orientation for a different edge never silently substitutes."""
    class _Foreign:
        upper_ladder_index = 2
        lower_ladder_index = 1

    with pytest.raises(ValueError, match="RF-DIR-001B"):
        create_sensor_model(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            [0.0, 0.0, 0.3], [0.0, 0.0, 0.0],
            rf_orientation=_Foreign(),
        )


def test_downward_population_flows_to_the_lower_energy_manifold():
    """Independent analytic oracle for the jump direction.

    With H = 0 and the single jump |4><3| at rate gamma, starting from |3>:

        rho_33(t) = exp(-gamma t),   rho_44(t) = 1 - exp(-gamma t)

    which is the exact solution of the Lindblad master equation.  A wrong
    jump operator (e.g. the historical |3><4|) deviates by O(1).
    """
    gamma = 0.7  # 1/us
    H, c_ops, _ = create_sensor_model(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        [0.0, 0.0, gamma], [0.0, 0.0, 0.0],
        rf_orientation=DOWNWARD,
    )
    rho0 = qt.basis(4, 2).proj()  # |3><3| — the energetic TOP of the edge
    tlist = np.linspace(0.0, 2.0, 11)
    states = qt.mesolve(H, rho0, tlist, c_ops, []).states

    p3 = np.array([np.real(r.full()[2, 2]) for r in states])
    p4 = np.array([np.real(r.full()[3, 3]) for r in states])

    np.testing.assert_allclose(p3, np.exp(-gamma * tlist), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(p4, 1.0 - np.exp(-gamma * tlist), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(p3 + p4, 1.0, rtol=0.0, atol=1e-6)


def test_upward_and_default_jump_are_silent_on_the_higher_manifold():
    """Falsification partner: the historical jump de-excites |4>, not |3>.

    Starting from |3> under the default/upward operator the RF channel is
    identically zero — exactly the silent failure RF-DIR-001B removes for
    the downward production pair.
    """
    gamma = 0.7
    for orientation in (None, UPWARD):
        H, c_ops, _ = create_sensor_model(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            [0.0, 0.0, gamma], [0.0, 0.0, 0.0],
            rf_orientation=orientation,
        )
        rho0 = qt.basis(4, 2).proj()  # |3><3|
        tlist = np.linspace(0.0, 2.0, 5)
        states = qt.mesolve(H, rho0, tlist, c_ops, []).states
        p3 = np.array([np.real(r.full()[2, 2]) for r in states])
        p4 = np.array([np.real(r.full()[3, 3]) for r in states])
        np.testing.assert_allclose(p3, 1.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(p4, 0.0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Scope guards: only the RF diagonal (001C) and the RF jump (001B) may differ
# ---------------------------------------------------------------------------

def test_hamiltonian_differs_only_in_the_rf_diagonal_for_downward():
    """RF-DIR-001C has landed: the DOWNWARD build differs from the legacy
    build ONLY in the |4><4| RF entry (by exactly -2*Delta_rf); every other
    matrix entry, the observable and the RF-DIR-001B jump stay in place."""
    args = (0.2, 0.3, 0.4, 0.1, 0.0, 0.05,
            [0.3, 0.3, 0.15], [0.05, 0.05, 0.05])

    H_default, c_default, obs_default = create_sensor_model(*args)
    H_down, c_down, obs_down = create_sensor_model(*args, rf_orientation=DOWNWARD)

    diff = H_down.full() - H_default.full()
    mask = np.ones(diff.shape, dtype=bool)
    mask[3, 3] = False
    np.testing.assert_array_equal(
        diff[mask], np.zeros(diff.shape, dtype=complex)[mask])
    np.testing.assert_allclose(diff[3, 3], -2.0 * args[5],
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(obs_down.full(), obs_default.full())
    # Only the RF jump differs among the collapse operators.
    assert len(c_down) == len(c_default)
    assert not np.allclose(c_down[2].full(), c_default[2].full())

    for Hm in (H_default.full(), H_down.full()):
        np.testing.assert_allclose(Hm, Hm.conj().T, atol=0.0)


def test_optical_decay_channels_and_dephasing_are_unchanged():
    """Only the RF jump may differ; every other channel stays bit-identical."""
    gammas = [0.5, 0.25, 0.3]
    dephasing = [0.05, 0.05, 0.05]
    args = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, gammas, dephasing)

    _, c_default, _ = create_sensor_model(*args)
    _, c_down, _ = create_sensor_model(*args, rf_orientation=DOWNWARD)

    assert len(c_down) == len(c_default) == 6  # 3 decays + 3 dephasing
    for idx in (0, 1, 3, 4, 5):                # gamma_21, gamma_32, dephasing
        np.testing.assert_array_equal(c_down[idx].full(), c_default[idx].full())

    expected_down = np.zeros((4, 4))
    expected_down[3, 2] = np.sqrt(gammas[2])   # |4><3|
    np.testing.assert_allclose(c_down[2].full(), expected_down,
                               rtol=0.0, atol=1e-15)
    expected_default = np.zeros((4, 4))
    expected_default[2, 3] = np.sqrt(gammas[2])  # |3><4| (historical)
    np.testing.assert_allclose(c_default[2].full(), expected_default,
                               rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 5. Solver seams forward the orientation; config reader is guarded
# ---------------------------------------------------------------------------

def test_solve_probe_coherence_forwards_the_orientation(monkeypatch):
    """_solve_probe_coherence hands the orientation to create_sensor_model."""
    import rydberg_aoa.simulation.steady_state as ss

    captured = []
    real_model = ss.create_sensor_model

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rf_orientation", "MISSING"))
        return real_model(*args, **kwargs)

    monkeypatch.setattr(ss, "create_sensor_model", _spy)

    ss._solve_probe_coherence(
        0.0, 0.0, 0.05, 0.05, 0.0, 0.0,
        [0.05, 0.05, 0.1], [0.01, 0.01, 0.01],
        transit_rate_per_us=0.0,
        rf_orientation=DOWNWARD,
    )
    assert captured[-1] is DOWNWARD

    ss._solve_probe_coherence(
        0.0, 0.0, 0.05, 0.05, 0.0, 0.0,
        [0.05, 0.05, 0.1], [0.01, 0.01, 0.01],
        transit_rate_per_us=0.0,
    )
    assert captured[-1] is None  # default keeps the historical build


def test_simulate_spectrum_forwards_the_orientation(monkeypatch):
    """simulate_spectrum threads the orientation into every solver call."""
    import rydberg_aoa.simulation.steady_state as ss

    captured = []

    def _fake_solve(*args, **kwargs):
        captured.append(kwargs.get("rf_orientation", "MISSING"))
        return 0.0 + 0.0j

    monkeypatch.setattr(ss, "_solve_probe_coherence", _fake_solve)
    # The analytic and transmission conversions read the (uninitialized here)
    # atomic configuration; the forwarding contract needs neither.
    monkeypatch.setattr(
        ss.solutions, "calculate_analytical_rho21", lambda *a, **k: 0.0 + 0.0j
    )
    monkeypatch.setattr(ss, "calculate_transmission_from_rho21", lambda rho: 1.0)

    ss.simulate_spectrum(
        0.0, [0.0],
        (0.05, 0.05, 0.0), (0.0, 0.0, 0.0),
        [0.05, 0.05, 0.1], [0.01, 0.01, 0.01], 0.0,
        doppler_enabled=False, transit_enabled=False,
        rf_orientation=DOWNWARD,
    )
    assert captured
    assert all(orientation is DOWNWARD for orientation in captured)


def test_get_rf_orientation_is_guarded(published_atomic_state):
    """With only the seam flag set nothing is published: reader -> None."""
    assert published_atomic_state.get_rf_orientation() is None


def test_get_rf_orientation_raises_before_initialization(monkeypatch):
    """Before initialize_atomic_physics() the reader fails loudly (F05)."""
    monkeypatch.setattr(acfg, "_ATOMIC_PHYSICS_INITIALIZED", False)
    with pytest.raises(PhysicsConfigurationError):
        acfg.get_rf_orientation()


# ---------------------------------------------------------------------------
# 6. End-to-end production path: live ARC rate to energy-ordered jump
# ---------------------------------------------------------------------------

def test_production_path_builds_energy_ordered_jump_with_live_arc_rate():
    """Live ladder R1=26D5/2 to R2=27P3/2: builder emits |4⟩⟨3| at sqrt(gamma_43).

    Joins the two production stages with NO fakes: the live ARC rates from
    ``calculate_decay_rates`` and the orientation from
    ``resolve_rydberg_orientation`` feed ``create_sensor_model`` exactly as
    ``initialize_atomic_physics()`` and the solver seams do.
    """
    atom = _arc_atom()
    orientation = resolve_rydberg_orientation(atom, R1, R2)
    gamma_21, gamma_32, gamma_43 = calculate_decay_rates(atom, GROUND, EXCITED, R1, R2)

    # Energy-ordered (R1 → R2), one-way at temperature=0.0 K, in 1/us.
    assert (orientation.upper_ladder_index,
            orientation.lower_ladder_index) == (3, 4)
    assert gamma_43 == pytest.approx(R1_TO_R2_RATE_S / 1e6, rel=1e-6)
    np.testing.assert_array_less(0.0, [gamma_21, gamma_32, gamma_43])

    _, c_ops, _ = create_sensor_model(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        [gamma_21, gamma_32, gamma_43], [0.0, 0.0, 0.0],
        rf_orientation=orientation,
    )
    # Decay channels occupy c_ops[0..2]; c_ops[2] is the RF edge.
    assert len(c_ops) == 3
    rf_jump = c_ops[2].full()
    expected = np.zeros((4, 4), dtype=complex)
    expected[3, 2] = np.sqrt(gamma_43)   # |4⟩⟨3|: source |3⟩ = R1, target |4⟩ = R2
    np.testing.assert_allclose(rf_jump, expected, rtol=0.0, atol=1e-15)
    # The historical |3⟩⟨4| element must be empty on this production path.
    assert rf_jump[2, 3] == 0.0
