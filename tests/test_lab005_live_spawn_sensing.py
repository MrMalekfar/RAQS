"""Bounded real ARC -> spawn worker -> IF -> DoA test, no synthetic outputs."""

import copy
import numpy as np

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.constants import C_LIGHT
from rydberg_aoa.config.rf_configuration import build_configured_lab_request
from rydberg_aoa.simulation.runner import SimulationRunner
from rydberg_aoa.simulation.laboratory_execution import LaboratoryRunConfig
from tests.test_lab004_live_laboratory_runtime import live_request


def test_real_arc_spawn_workers_and_beamforming_with_explicit_optical_setpoints(live_request):
    # Explicit test-profile preparation. Production performs no laser tuning.
    ac.initialize_atomic_physics(resolution_request=live_request)
    selection = ac.get_laboratory_runtime().selection
    probe_nm = selection.optical_selection.probe.proposed_wavelength_nm
    coupling_nm = selection.winner.manifold.coupling.proposed_wavelength_nm
    ac.reset_atomic_physics()
    ac.LASER_PARAMS['probe']['wavelength_nm'] = probe_nm
    ac.LASER_PARAMS['coupling']['wavelength_nm'] = coupling_nm
    req = build_configured_lab_request(preparation=live_request.preparation, bounds=live_request.bounds, atom=ac.ATOM)
    spacing = C_LIGHT / req.signal.frequency_hz / 8
    cfg = LaboratoryRunConfig(tuple(np.linspace(0., 60., 601)),
                              tuple((i*spacing, 0.) for i in range(3)), -30., 20.,
                              (.02, .03, .03), .5, processes=1)
    applied = copy.deepcopy((ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS))
    delay = params.ATOMIC_TRANSIENT_DELAY_US
    assert delay != cfg.transient_delay_us  # Child default must not win.
    runner = SimulationRunner()
    result = runner.run_laboratory_simulation(req, cfg)
    assert result['estimated_aoa'] == -30.
    assert result['laboratory_runtime'].selection.rf_orientation.direction == -1
    assert abs(ac.DELTA_P) < 1e-6 and abs(ac.DELTA_C) < 1e-6
    assert result['sensor_detunings_rad_per_us'] == ac.get_cached_detunings()
    assert all(len(record[1]) == 401 for record in result['worker_records'])
    assert all(len(record[4]) == 601 for record in result['worker_records'])
    expected_phase = 2*np.pi*req.signal.frequency_hz/C_LIGHT * np.arange(3)*spacing * np.sin(np.deg2rad(-30.))
    recovered = np.unwrap(np.angle(result['complex_responses'] / result['complex_responses'][0]))
    np.testing.assert_allclose(recovered, expected_phase, atol=2e-5, rtol=0)
    assert params.ATOMIC_TRANSIENT_DELAY_US == delay
    assert applied == (ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS)
    assert result['if_snapshots'].diagnostics['start_us'] == 20.
    print('\nLAB-005 real spawn:', 'AoA=', result['estimated_aoa'], 'phase=', recovered,
          'expected=', expected_phase, 'IF=', result['signed_if_mhz'], 'detunings=', ac.get_cached_detunings(),
          'Rabi=', result['laboratory_runtime'].selection.rabi)
