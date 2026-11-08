"""LAB-002 declarations validate inputs without ARC, tuning or mutable aliases."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rydberg_aoa.config.laboratory import (
    AppliedLaser, AppliedRF, PreparedHyperfineState, PreparedChannel,
    LabScanBounds, ExperimentalLabRequest,
)
from rydberg_aoa.exceptions import PhysicsConfigurationError


def reference_request():
    return ExperimentalLabRequest(
        probe=AppliedLaser(780.241476, .0001, .0798, 0., 0),
        coupling=AppliedLaser(483.4, 20., .1223, 0., 0),
        lo=AppliedRF(132.649e9, 1.5, 0), signal=AppliedRF(132.6491e9, .005, 0),
        preparation=PreparedChannel(PreparedHyperfineState(5, 0, .5, 3, 0),
                                    PreparedHyperfineState(5, 1, 1.5, 4, 0),
                                    -.5, -.5, (0., 0., 1.)),
        bounds=LabScanBounds(10, 110, 3, .3e9, (780., 781.), (480., 490.), (130e9, 135e9)),
    )


BAD_REAL = [True, np.bool_(False), '1', b'1', 1j, np.complex128(1),
            np.array(1.), np.array([1.]), np.nan, np.inf, 10**1000]


@pytest.mark.parametrize('bad', BAD_REAL)
@pytest.mark.parametrize('field', ['frequency_hz', 'electric_field_vm'])
def test_rf_input_requires_real_finite_scalar(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().lo, **{field: bad})


@pytest.mark.parametrize('field,bad', [('frequency_hz', 0), ('frequency_hz', -1),
                                     ('electric_field_vm', -1), ('polarization', True),
                                     ('polarization', 0.0), ('polarization', 2)])
def test_rf_invalid_domain(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().lo, **{field: bad})


@pytest.mark.parametrize('bad', BAD_REAL)
def test_laser_offset_has_strict_real_type(bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().probe, detuning_mhz=bad)


@pytest.mark.parametrize('field,bad', [('wavelength_nm', 0), ('power_mw', -1),
                                     ('beam_waist_mm', 0), ('polarization', '0')])
def test_laser_domain(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().probe, **{field: bad})


@pytest.mark.parametrize('field,bad', [('n_min', 1.5), ('n_max', True), ('n_max', 9),
                                     ('max_l', 4), ('rf_tolerance_hz', -1),
                                     ('probe_window_nm', (781., 780.)),
                                     ('coupling_window_nm', [480., 490.]),
                                     ('rf_window_hz', (0., 135e9))])
def test_hardware_domain(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().bounds, **{field: bad})


@pytest.mark.parametrize('field,bad', [('n', 5.5), ('l', 5), ('j', .75),
                                     ('f', -.5), ('mf', .5), ('mf', 4)])
def test_prepared_hfs_state_requires_physical_quantum_numbers(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().preparation.ground, **{field: bad})


def test_large_invalid_orbital_is_rejected_without_raw_float_overflow():
    with pytest.raises(PhysicsConfigurationError, match='n,l,j'):
        PreparedHyperfineState(10**1000 + 1, 10**1000, .5, 3, 0)


@pytest.mark.parametrize('field,bad', [('rydberg1_mj', 0), ('rydberg2_mj', .3),
                                     ('quantization_axis', (0., 0., 2.)),
                                     ('quantization_axis', [0., 0., 1.]),
                                     ('ground', None)])
def test_preparation_has_no_implicit_channel_or_axis(field, bad):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request().preparation, **{field: bad})


def test_request_identity_is_immutable_exact_and_contains_all_inputs():
    request = reference_request()
    assert request.validate_for_atom(SimpleNamespace(I=2.5, groundStateN=5)) is request
    assert request == reference_request()
    assert hash(request) == hash(reference_request())
    for field, replacement in (
        ('lo', replace(request.lo, frequency_hz=request.lo.frequency_hz + .001)),
        ('probe', replace(request.probe, power_mw=.0002)),
        ('coupling', replace(request.coupling, beam_waist_mm=.15)),
        ('bounds', replace(request.bounds, n_min=11)),
        ('preparation', replace(request.preparation, quantization_axis=(1., 0., 0.))),
    ):
        assert replace(request, **{field: replacement}) != request
    with pytest.raises(FrozenInstanceError):
        request.lo.frequency_hz = 0
    assert request.selection_carrier_hz == 132.649e9
    assert request.signed_if_mhz == .1


def test_signed_if_is_not_absolute_and_zero_rf_field_is_preserved():
    request = reference_request()
    request = replace(request, signal=replace(request.signal, frequency_hz=132.6489e9, electric_field_vm=0))
    assert request.signed_if_mhz == -.1
    assert request.signal.electric_field_vm == 0


@pytest.mark.parametrize('field', ['probe', 'coupling', 'lo', 'signal', 'preparation', 'bounds'])
def test_incomplete_request_is_rejected(field):
    with pytest.raises(PhysicsConfigurationError):
        replace(reference_request(), **{field: None})


@pytest.mark.parametrize('field,value', [('probe', 782.), ('coupling', 500.),
                                       ('lo', 136e9), ('signal', 129e9)])
def test_applied_fields_must_fit_explicit_hardware_windows(field, value):
    request = reference_request()
    name = 'wavelength_nm' if field in ('probe', 'coupling') else 'frequency_hz'
    with pytest.raises(PhysicsConfigurationError, match='hardware window'):
        replace(request, **{field: replace(getattr(request, field), **{name: value})})


def test_probe_and_rf_polarizations_must_match_prepared_channel():
    request = reference_request()
    with pytest.raises(PhysicsConfigurationError, match='prepared probe'):
        replace(request, probe=replace(request.probe, polarization=1))
    with pytest.raises(PhysicsConfigurationError, match='same prepared RF'):
        replace(request, signal=replace(request.signal, polarization=1))
    with pytest.raises(PhysicsConfigurationError, match='projections'):
        replace(request, preparation=replace(request.preparation, rydberg2_mj=.5))


@pytest.mark.parametrize('atom', [None, SimpleNamespace(I=2.5, groundStateN=6),
                                 SimpleNamespace(I=1.5, groundStateN=5)])
def test_actual_atom_controls_preparation(atom):
    with pytest.raises(PhysicsConfigurationError):
        reference_request().validate_for_atom(atom)


def test_coupling_nuclear_projection_is_not_inferred_from_mf():
    request = reference_request()
    preparation = replace(request.preparation, rydberg1_mj=3.5, rydberg2_mj=3.5)
    with pytest.raises(PhysicsConfigurationError, match='nuclear projection'):
        replace(request, preparation=preparation).validate_for_atom(SimpleNamespace(I=2.5, groundStateN=5))


def test_construction_never_tunes_applied_frequency_or_wavelength():
    request = reference_request()
    assert request.coupling.wavelength_nm == 483.4
    assert request.lo.frequency_hz == 132.649e9
    assert request.preparation.rydberg1_mj == -.5
    assert not hasattr(request.preparation, 'rydberg1_n')
