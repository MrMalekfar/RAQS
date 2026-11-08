"""OP-001 — FROZEN SENSOR OPERATING POINT contract tests.

HET-001 (small-signal heterodyne operating-point audit) selected the
production SENSOR operating point for the CURRENT effective four-level model:

    P_probe   = 0.10 uW  (= 0.0001 mW)
    E_LO      = 1.5 V/m
    E_signal  = 0.005 V/m
    Delta_probe (sensor; time-domain / heterodyne path only) = -0.5 MHz
    f_IF      = 0.1 MHz  (= |f_signal - f_LO|)

OP-001 freezes these values.  This is the selected operating point WITHIN
the current effective four-level model — NOT claimed as a universal physical
optimum and NOT a Simons et al. 2016 reference value.  The reference probe
power (17.5 uW) is documented as a separate REFERENCE / SIMONS INPUT profile
only.

Scope guards:
  * Concept separation (A vs B): the steady-state spectrum scan (A) must stay
    independent of the sensor operating detuning (B); only the time-domain /
    heterodyne sensing path applies SENSOR_PROBE_DETUNING_MHZ
    (SimulationRunner._sensor_probe_detunings).
  * Unit contract (C-001/OP-001): Delta_sensor [rad/us] = 2*pi * [MHz].
    Exactly ONE 2*pi factor — NO extra 2*pi.
  * E_LO and E_signal are distinct fields (Omega_LO / Omega_signal = 300).
"""

import math
import pathlib

import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config.laser_rf_config import (
    LASER_PARAMS,
    LO_RF_PARAMS,
    RF_PARAMS,
    SIGNAL_RF_PARAMS,
)
from rydberg_aoa.simulation.runner import SimulationRunner
from rydberg_aoa.signal_processing.array_geometry import get_array_phases
from tests.atomic_init_support import ensure_resolved_atomic_physics

S2LN2 = math.sqrt(2.0 * math.log(2.0))


@pytest.fixture(scope="module")
def real_physics():
    """Real ARC-derived physics bundle, identical to the runner's inputs.

    ARCH-001T: opts into the explicit ARCH-001 seam through the shared
    support helper (idempotent; skipped with a ``BLOCKED: QN-001A`` reason
    while the coupling mJ contract is open).
    """
    ensure_resolved_atomic_physics()
    dp, dc, drf = acfg.get_cached_detunings()
    return {
        "omegas": (acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE),
        "deltas": (dp, dc, drf),
        "gammas": [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43],
        "acfg": acfg,
    }


@pytest.fixture
def cold_path(monkeypatch):
    """Disable Doppler averaging: exactly one v=0 class per scan point."""
    monkeypatch.setattr(params, "DOPPLER_ENABLED", False, raising=True)


# --- 1. Configuration contract — the five frozen values ---------------------
def test_probe_power_is_sensor_operating_point():
    """P_probe = 0.10 uW = 0.0001 mW (NOT the 17.5 uW Simons reference)."""
    assert LASER_PARAMS["probe"]["power_mw"] == pytest.approx(0.0001, rel=0, abs=0)


def test_lo_field_frozen_1_5_vm():
    assert params.RF_LO_E_FIELD == pytest.approx(1.5, rel=0, abs=0)


def test_signal_field_preserved_0_005_vm():
    assert params.RF_SIGNAL_E_FIELD == pytest.approx(0.005, rel=0, abs=0)


def test_lo_and_signal_are_distinct_fields():
    """E_LO and E_signal must stay distinct configuration owners."""
    assert params.RF_LO_E_FIELD != params.RF_SIGNAL_E_FIELD
    assert LO_RF_PARAMS is not SIGNAL_RF_PARAMS
    assert LO_RF_PARAMS["electric_field_vm"] == pytest.approx(1.5)
    assert SIGNAL_RF_PARAMS["electric_field_vm"] == pytest.approx(0.005)
    # Level-optimization alias follows the LO owner (1.5 V/m), never the signal.
    assert RF_PARAMS["electric_field_vm"] == pytest.approx(1.5)


def test_sensor_probe_detuning_config():
    assert params.SENSOR_PROBE_DETUNING_MHZ == pytest.approx(-0.5, rel=0, abs=0)


def test_if_frequency_is_0_1_mhz():
    assert params.IF_FREQ == pytest.approx(0.1, rel=1e-9)
    assert (
        abs(params.RF_SIGNAL_FREQUENCY - params.RF_LO_FREQUENCY) / 1e6
        == pytest.approx(0.1, rel=1e-9)
    )


def test_beam_radii_preserved_from_beam002():
    """BEAM-002 radii stay frozen (79.836 / 122.302 um) — untouched by OP-001."""
    assert LASER_PARAMS["probe"]["beam_waist_mm"] == pytest.approx(
        94.0 / S2LN2 / 1000.0, rel=1e-12)
    assert LASER_PARAMS["coupling"]["beam_waist_mm"] == pytest.approx(
        144.0 / S2LN2 / 1000.0, rel=1e-12)

# --- 2. Rabi scaling (OP-001 Phase 4): Omega_LO(new) = 3 * Omega_LO(old) ----
def test_omega_lo_signal_ratio_equals_field_ratio(atomic_physics):
    """Same RF transition => Omega proportional to E: ratio must be 300."""
    assert acfg.OMEGA_RF_BASE / acfg.OMEGA_SIGNAL_RF_BASE == pytest.approx(
        1.5 / 0.005, rel=1e-6)


def test_omega_lo_is_three_times_old_0_5_vm_value(atomic_physics):
    """Independently recompute Omega at E=0.5 V/m via the same ARC transition:
    Omega(1.5) must equal exactly 3 * Omega(0.5) (dipole unchanged)."""
    from rydberg_aoa.arc.rabi_frequency import calculate_rf_rabi_frequency
    from rydberg_aoa.arc.types import AtomicLevel, RFParams

    l1 = AtomicLevel(**acfg.ATOMIC_LEVELS["rydberg1"])
    l2 = AtomicLevel(**acfg.ATOMIC_LEVELS["rydberg2"])
    omega_05_rads = calculate_rf_rabi_frequency(
        acfg.ATOM, l1, l2,
        RFParams(frequency_hz=params.RF_LO_FREQUENCY,
                 electric_field_vm=0.5, polarization=0),
    )
    assert acfg.OMEGA_RF_BASE / (omega_05_rads / 1e6) == pytest.approx(
        3.0, rel=1e-9)


# --- 3. MHz -> rad/us unit contract (Phase 6): 2*pi*(-0.5) = -pi -------------
def test_mhz_to_rad_us_conversion_exactly_one_2pi(atomic_physics):
    """Delta_sensor [rad/us] = 2*pi * (-0.5); offset exactly -pi rad/us."""
    runner = SimulationRunner()
    dp0 = acfg.get_cached_detunings()[0]
    dp_sensor, _, _ = runner._sensor_probe_detunings((dp0, 0.0, 0.0))
    assert dp_sensor == pytest.approx(dp0 + 2 * math.pi * (-0.5), rel=1e-12)
    assert dp_sensor - dp0 == pytest.approx(-math.pi, rel=1e-12)


def test_sensor_detuning_value_pinned():
    assert 2 * math.pi * params.SENSOR_PROBE_DETUNING_MHZ == pytest.approx(
        -math.pi, rel=1e-12)


# --- 4. Spectrum-scan independence from sensor operating detuning (Phase 8) --
def test_spectrum_module_does_not_read_sensor_detuning():
    """Structural guard: steady_state.py (scan path A) has no reference to
    the sensor operating detuning (path B) — a future regression wiring the
    sensor detuning into the EIT/AT scan breaks this test."""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "rydberg_aoa" / "simulation" / "steady_state.py").read_text(
        encoding="utf-8")
    assert "SENSOR_PROBE_DETUNING_MHZ" not in src


def test_scan_grid_independent_of_sensor_detuning(real_physics, cold_path,
                                                  monkeypatch):
    """The scan grid is honored EXACTLY whatever the sensor operating
    detuning is set to (monkeypatched to +3.7 MHz => scan unchanged)."""
    from rydberg_aoa.simulation import steady_state as ss

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]
    gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    monkeypatch.setattr(params, "SENSOR_PROBE_DETUNING_MHZ", 3.7)
    scan, _, _ = ss.simulate_spectrum(
        omega_rf=0.0,
        detuning_list_mhz=[0.0],
        p_omegas=(omega_p, omega_c, 0.0),
        p_deltas=(dp, dc, drf),
        p_gammas=gammas,
        p_dephasing=[1.0, 1.0, 1.0],
        absorption_const=None,
        transit_rate_per_us=0.0,
    )
    assert list(scan) == [0.0]


def test_spectrum_solver_uses_unshifted_center_detuning(
        real_physics, cold_path, monkeypatch):
    """The steady-state scan solver receives delta_p UNCHANGED (no sensor
    offset) — the sensor detuning only reaches the time-domain path."""
    from rydberg_aoa.simulation import steady_state as ss

    dp, dc, drf = real_physics["deltas"]
    omega_p, omega_c, _ = real_physics["omegas"]
    gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]

    captured = {}
    real_csm = ss.create_sensor_model

    def passthrough_csm(omega_p_, omega_c_, omega_lo_, delta_p_, delta_c_,
                        delta_rf_, gammas_, dephasing_, **kw):
        captured["delta_p"] = delta_p_
        return real_csm(omega_p_, omega_c_, omega_lo_, delta_p_, delta_c_,
                        delta_rf_, gammas_, dephasing_, **kw)

    monkeypatch.setattr(ss, "create_sensor_model", passthrough_csm)
    ss.simulate_spectrum(
        omega_rf=0.0,
        detuning_list_mhz=[0.0],
        p_omegas=(omega_p, omega_c, 0.0),
        p_deltas=(dp, dc, drf),
        p_gammas=gammas,
        p_dephasing=[1.0, 1.0, 1.0],
        absorption_const=None,
        transit_rate_per_us=0.0,
    )
    assert captured["delta_p"] == pytest.approx(dp, rel=1e-12)

# --- 5. Runner threading — SENSOR detuning reaches the time-domain worker
#    (caller-contract test, RF-004 lesson: validate the caller — patch the
#    runner's OWN binding, exactly as the production caller invokes it) ------
def test_timedomain_worker_receives_sensor_detuning_and_frozen_fields(
        monkeypatch, real_physics):
    """run_time_domain_simulation must pass delta_p + 2*pi*(-0.5), the 1.5 V/m
    LO Rabi, and IF = 0.1 MHz to every worker."""
    import multiprocessing as mp
    import rydberg_aoa.simulation.runner as runner_mod

    dp0, dc0, drf0 = real_physics["deltas"]
    runner = SimulationRunner()

    seen = {}
    pool_used = {"entered": False}

    def fake_worker(i, omega_s, tlist, if_freq, om_p, om_c, om_lo,
                    d_p, d_c, d_rf, gams, deph, noise_model=None,
                    transit_rate_per_us=None):
        seen["delta_p"] = d_p
        seen["if_freq"] = if_freq
        seen["omega_lo"] = om_lo
        seen["omega_s"] = complex(omega_s)
        return (0.5 + 0.25j, np.zeros(2), np.zeros(2), i, np.zeros(4))

    # Patch the CALLER's binding (runner module global), not time_domain's.
    monkeypatch.setattr(runner_mod, "run_single_heterodyne_worker", fake_worker)

    class _PoolStub:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starmap(self, func, args_list):
            pool_used["entered"] = True
            return [func(*a) for a in args_list]

    monkeypatch.setattr(mp, "Pool", _PoolStub)
    monkeypatch.setattr(acfg, "get_cached_detunings",
                        lambda: (dp0, dc0, drf0))

    runner.run_time_domain_simulation()

    assert pool_used["entered"], "mp.Pool was not swapped by the stub"
    assert seen["delta_p"] == pytest.approx(
        dp0 + 2 * math.pi * params.SENSOR_PROBE_DETUNING_MHZ, rel=1e-12)
    assert seen["delta_p"] == pytest.approx(dp0 - math.pi, rel=1e-12)
    assert seen["if_freq"] == pytest.approx(0.1, rel=1e-12)
    assert seen["omega_lo"] == pytest.approx(acfg.OMEGA_RF_BASE, rel=1e-12)
    assert abs(seen["omega_s"]) == pytest.approx(
        acfg.OMEGA_SIGNAL_RF_BASE, rel=1e-12)


# --- 6. AoA array-math regression (beamforming oracle, no physics workers) ---
def test_aoa_beamforming_recovers_true_angle():
    """Synthetic array with the true phase progression must be recovered by
    the production beamforming estimator at the frozen geometry."""
    runner = SimulationRunner()
    runner.setup_array_geometry()
    true = params.TARGET_AOA_DEG
    resp = np.exp(1j * get_array_phases(
        runner.array_positions, params.RF_FREQUENCY, true))
    angles, power, est = runner.estimate_angle_of_arrival(resp)
    assert est == pytest.approx(true, abs=0.25)
    assert angles[np.argmax(power)] == est


# --- 7. Real time-domain execution at the frozen operating point (Phase 9) --
@pytest.mark.slow
def test_time_domain_real_worker_at_frozen_operating_point(atomic_physics):
    """The REAL worker body at P=0.10 uW / E_LO=1.5 V/m / E_sig=0.005 V/m /
    Delta_sensor=-0.5 MHz / IF=0.1 MHz: beat present at 0.1 MHz with bounded
    2IF/IF ratio (small-signal regime).  Single element, no pool (F1-safe)."""
    from rydberg_aoa.physics.lindblad import calculate_transit_rate_per_us
    from rydberg_aoa.physics.lindblad.calculate_dephasing_rates import (
        calculate_dephasing_rates,
    )
    from rydberg_aoa.simulation.time_domain import run_single_heterodyne_worker

    dp0, dc0, drf0 = acfg.get_cached_detunings()
    runner = SimulationRunner()
    dp_sensor, _, _ = runner._sensor_probe_detunings((dp0, dc0, drf0))

    tlist = np.linspace(0.0, params.T_MAX, params.T_STEPS)
    gammas = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]
    res = run_single_heterodyne_worker(
        0, complex(acfg.OMEGA_SIGNAL_RF_BASE, 0.0), tlist,
        params.IF_FREQ,
        acfg.OMEGA_P, acfg.OMEGA_C, acfg.OMEGA_RF_BASE,
        dp_sensor, dc0, drf0,
        gammas, list(calculate_dephasing_rates()),
        None, calculate_transit_rate_per_us(),
    )
    assert len(res) == 5
    assert abs(res[0]) > 0.0
    trace = res[4]
    assert np.mean(np.abs(trace)) > 0.0

    dt = tlist[1] - tlist[0]
    freqs = np.fft.fftfreq(len(tlist), dt)
    spec = np.abs(np.fft.fft(trace)) ** 2
    band = (freqs > 0.05) & (freqs < 0.15)
    peak_f = freqs[band][np.argmax(spec[band])]
    assert peak_f == pytest.approx(params.IF_FREQ, abs=0.01)
    band2 = (freqs > 0.19) & (freqs < 0.21)
    assert spec[band2].max() / spec[band].max() < 1.0
