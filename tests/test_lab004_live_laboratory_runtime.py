"""Actual ARC initialization, independent angular/energy/rate checks; no skip."""

import copy
import math
from dataclasses import replace

import pytest

from rydberg_aoa.config import atomic_config as ac, parameters as params
from rydberg_aoa.config.constants import C_LIGHT, HBAR
from rydberg_aoa.config.laboratory import OpticalEnergyConvention
from rydberg_aoa.config.rf_configuration import build_configured_lab_request
from rydberg_aoa.arc.dipole import EA0_TO_CM
from tests.atomic_init_support import snapshot_published_atomic_state, restore_published_atomic_state
from tests.test_auto001b_live_prepared_channel import predicted_elements
from tests.test_lab002_laboratory_request import reference_request


@pytest.fixture
def live_request(configured_atom):
    published = snapshot_published_atomic_state()
    pipeline = ac._snapshot_pipeline_transaction()
    template = reference_request()
    ac.reset_atomic_physics()
    try:
        yield build_configured_lab_request(preparation=template.preparation, bounds=template.bounds, atom=ac.ATOM)
    finally:
        ac._rollback_pipeline_transaction(pipeline)
        restore_published_atomic_state(published)


def test_live_canonical_runtime_matches_independent_arc_quantities(live_request):
    req, atom = live_request, ac.ATOM
    applied = copy.deepcopy((ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS))
    assert ac.initialize_atomic_physics(resolution_request=req)
    runtime = ac.get_laboratory_runtime()
    fs = runtime.selection.winner.manifold
    a, b = fs.rydberg1, fs.rydberg2
    assert (a.n, a.l, a.j, b.n, b.l, b.j) == (26, 2, 2.5, 27, 1, 1.5)
    probe, coupling, rf = predicted_elements(atom, req, fs)
    observed = ac.get_cached_dipole_matrix()
    assert tuple(observed[name]['dipole_ea0'] for name in ('probe', 'coupling', 'rf')) == pytest.approx(
        (probe, coupling, rf), rel=2e-13, abs=0)
    assert ac.OMEGA_RF_BASE == abs(rf * EA0_TO_CM) * req.lo.electric_field_vm / HBAR / 1e6
    assert ac.OMEGA_SIGNAL_RF_BASE / ac.OMEGA_RF_BASE == pytest.approx(
        req.signal.electric_field_vm / req.lo.electric_field_vm, rel=2e-15)
    g, e = req.preparation.ground, req.preparation.excited
    fp = atom.getTransitionFrequency(g.n, g.l, g.j, e.n, e.l, e.j)
    fc = atom.getTransitionFrequency(e.n, e.l, e.j, a.n, a.l, a.j)
    signed = atom.getTransitionFrequency(a.n, a.l, a.j, b.n, b.l, b.j)
    deltas = ((C_LIGHT / (req.probe.wavelength_nm * 1e-9) + req.probe.detuning_mhz * 1e6 - fp) * 2*math.pi / 1e6,
              (C_LIGHT / (req.coupling.wavelength_nm * 1e-9) + req.coupling.detuning_mhz * 1e6 - fc) * 2*math.pi / 1e6,
              (req.lo.frequency_hz - abs(signed)) * 2*math.pi / 1e6)
    assert ac.get_cached_detunings() == pytest.approx(deltas, rel=2e-15)
    assert ac.get_rf_orientation().signed_frequency_hz == signed < 0
    rates = (atom.getTransitionRate(e.n, e.l, e.j, g.n, g.l, g.j),
             atom.getTransitionRate(a.n, a.l, a.j, e.n, e.l, e.j),
             atom.getTransitionRate(a.n, a.l, a.j, b.n, b.l, b.j))
    assert (ac.GAMMA_21, ac.GAMMA_32, ac.GAMMA_43) == tuple(rate / 1e6 for rate in rates)
    assert applied == (ac.LASER_PARAMS, ac.RF_PARAMS, ac.LO_RF_PARAMS, ac.SIGNAL_RF_PARAMS)
    assert abs(ac.DELTA_C) > 1e6  # Applied coupling stays far detuned; no hidden tuning.
    print('\nLAB-004 live:', ac.ATOMIC_LEVELS, 'detunings:', ac.get_cached_detunings(),
          'rates:', runtime.rates, 'Rabi:', runtime.selection.rabi)


def test_live_hfs_and_applied_offsets_reach_runtime_exactly_once(live_request):
    # Explicit user input change before initialization, then capture the request.
    ac.LASER_PARAMS['probe']['detuning_mhz'] = .125
    ac.LASER_PARAMS['coupling']['detuning_mhz'] = -.25
    req = build_configured_lab_request(preparation=live_request.preparation,
                                      bounds=live_request.bounds, atom=ac.ATOM)
    ac.initialize_atomic_physics(resolution_request=req)
    centroid = ac.get_laboratory_runtime()
    hfs_req = replace(req, optical_energy_convention=OpticalEnergyConvention.LOW_LEVEL_HFS)
    ac.initialize_atomic_physics(resolution_request=hfs_req, force=True)
    hfs = ac.get_laboratory_runtime()
    atom, g, e = ac.ATOM, req.preparation.ground, req.preparation.excited
    dg = atom.getHFSEnergyShift(g.j, g.f, *atom.getHFSCoefficients(g.n, g.l, g.j))
    de = atom.getHFSEnergyShift(e.j, e.f, *atom.getHFSCoefficients(e.n, e.l, e.j))
    fs = hfs.selection.winner.manifold
    a = fs.rydberg1
    fp = atom.getTransitionFrequency(g.n, g.l, g.j, e.n, e.l, e.j) + de - dg
    fc = atom.getTransitionFrequency(e.n, e.l, e.j, a.n, a.l, a.j) - de
    expected = ((C_LIGHT / (req.probe.wavelength_nm * 1e-9) + .125e6 - fp) * 2*math.pi / 1e6,
                (C_LIGHT / (req.coupling.wavelength_nm * 1e-9) - .25e6 - fc) * 2*math.pi / 1e6)
    assert hfs.identity != centroid.identity
    assert ac.get_cached_detunings()[:2] == pytest.approx(expected, rel=2e-15)
    assert hfs.detunings[2] == centroid.detunings[2]
    assert hfs.rates == centroid.rates and hfs.selection.rabi == centroid.selection.rabi
    assert ac.LASER_PARAMS['coupling']['wavelength_nm'] == req.coupling.wavelength_nm
