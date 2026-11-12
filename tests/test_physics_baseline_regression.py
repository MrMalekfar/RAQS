"""Current prepared-channel regression using frozen inputs and independent checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import epsilon_0, hbar

from rydberg_aoa.config.run_profile import load_run_profile
from rydberg_aoa.simulation.profile_execution import execute_profile
from tests.test_lab007_profile_integration import isolated

FIXTURE = Path(__file__).parent / 'fixtures/profile_schema_v1/prescribed_reference.json'
FIXTURE_SHA256 = '4188cbebe7c62d9b08a1e19f73db73bea18694c8470161b3343847a74dd5819b'


@pytest.fixture
def prepared_atomic_physics(isolated):
    from rydberg_aoa.config import atomic_config as ac

    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    execute_profile(load_run_profile(FIXTURE, raw=raw), diagnostics_only=True)
    return ac


def test_declared_states_and_downward_rf(prepared_atomic_physics):
    ac = prepared_atomic_physics
    data = json.loads(FIXTURE.read_bytes())
    for role, expected in data['ladder'].items():
        assert {key: ac.ATOMIC_LEVELS[role][key] for key in expected} == expected
    orientation = ac.get_rf_orientation()
    assert orientation.direction == -1
    lo = data['fields']['lo']
    signal = data['fields']['signal']
    assert ac.OMEGA_RF_BASE / ac.OMEGA_SIGNAL_RF_BASE == pytest.approx(
        lo['electric_field_vm'] / signal['electric_field_vm'], rel=1e-9)
    assert ac.DELTA_RF == pytest.approx(
        2 * np.pi * (lo['frequency_hz'] - abs(orientation.signed_frequency_hz)) / 1e6, rel=1e-9)


def test_absorption_prefactors_follow_si_contract(prepared_atomic_physics):
    from rydberg_aoa.config import parameters as params
    from rydberg_aoa.physics.susceptibility.absorption_const import calculate_absorption_constant

    ac = prepared_atomic_physics
    probe = json.loads(FIXTURE.read_bytes())['fields']['probe']
    dipole = ac.get_cached_dipole_matrix()['probe']['dipole_si']
    density = ac.ATOM.getNumberDensity(params.VAPOR_CELL_TEMPERATURE_K)
    od, susceptibility = calculate_absorption_constant()
    assert od == pytest.approx(np.pi * params.CELL_LENGTH / (probe['wavelength_nm'] * 1e-9), rel=1e-6)
    assert susceptibility == pytest.approx(
        -2 * density * dipole**2 / (epsilon_0 * hbar * ac.OMEGA_P * 1e6), rel=1e-6)


def test_steady_density_at_declared_and_offset_probe(prepared_atomic_physics):
    from qutip import liouvillian, steadystate
    from rydberg_aoa.physics.hamiltonian import create_sensor_model
    from rydberg_aoa.physics.lindblad import calculate_dephasing_rates
    from rydberg_aoa.physics.lindblad.calculate_transit_rate import calculate_transit_rate_per_us
    from rydberg_aoa.utils.validation import validate_steady_state

    ac = prepared_atomic_physics
    omegas = np.array([ac.OMEGA_P, ac.OMEGA_C, ac.OMEGA_RF_BASE])
    decay = [ac.GAMMA_21, ac.GAMMA_32, ac.GAMMA_43]
    dephasing = calculate_dephasing_rates()
    transit = calculate_transit_rate_per_us()
    basis = np.eye(16).reshape(16, 4, 4)
    ground = basis[0]
    jumps_with_rates = list(zip(((0, 1), (1, 2), (3, 2), (1, 1), (2, 2), (3, 3)),
                                [*decay, *dephasing]))
    for offset_mhz in (0., 6.):
        dp = ac.DELTA_P + 2 * np.pi * offset_mhz
        dc, drf = ac.DELTA_C, ac.DELTA_RF
        h, jumps, _ = create_sensor_model(
            *omegas, dp, dc, drf, decay, dephasing, transit_rate_per_us=transit,
            rf_orientation=ac.get_rf_orientation())
        rho = steadystate(h, jumps)
        validate_steady_state(rho, liouvillian(h, jumps),
                              context='prepared-channel regression', backend='qutip')

        # Independent matrix-unit construction and trace-constrained solve.
        reference_h = np.diag([0., dp, dp + dc, dp + dc - drf])
        reference_h -= (np.diag(omegas, 1) + np.diag(omegas, -1)) / 2
        columns = []
        for unit in basis:
            derivative = -1j * (reference_h @ unit - unit @ reference_h)
            derivative += transit * (ground * np.trace(unit) - unit)
            for (target, source), rate in jumps_with_rates:
                derivative[target, target] += rate * unit[source, source]
                derivative[source, :] -= rate * unit[source, :] / 2
                derivative[:, source] -= rate * unit[:, source] / 2
            columns.append(derivative.ravel())
        system = np.column_stack(columns)
        system[0] = np.eye(4).ravel()
        expected = np.linalg.solve(system, ground.ravel()).reshape(4, 4)
        assert rho.full() == pytest.approx(expected, rel=1e-6, abs=1e-12)
