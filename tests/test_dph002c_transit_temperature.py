"""
DPH-002C — transit-rate temperature-ownership regression tests.

Independent oracles: expected values are computed IN THIS FILE from the
first-principles formula using the PRODUCTION temperature owner
(``parameters.VAPOR_CELL_TEMPERATURE_K``), never from the duplicated
``constants.VAPOR_CELL_TEMPERATURE_K`` and never by calling the function
under test to generate expectations.
"""
import numpy as np
import pytest

from rydberg_aoa.config import parameters as params
from rydberg_aoa.config.constants import k_B  # project k_B (production constant; NOT the temperature)
from rydberg_aoa.config import atomic_config as acfg
from rydberg_aoa.config.laser_rf_config import LASER_PARAMS
from rydberg_aoa.physics.lindblad.calculate_transit_rate import (
    calculate_transit_rate_per_us,
)


def _oracle(T):
    """Independent Gamma_t oracle [1/us] at temperature T."""
    w0 = LASER_PARAMS['probe']['beam_waist_mm'] * 1e-3
    return (np.sqrt(2 / np.pi)
            * np.sqrt(k_B * T / acfg.ATOM.mass) / w0 * 1e-6)


# ---------------------------------------------------------------------------
# TEST 1 — default follows the production temperature
# ---------------------------------------------------------------------------
def test_default_follows_production_temperature(configured_atom):
    expected = _oracle(params.VAPOR_CELL_TEMPERATURE_K)
    assert calculate_transit_rate_per_us() == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# TEST 2 — explicit temperature overrides the default
# ---------------------------------------------------------------------------
def test_explicit_temperature_override(configured_atom):
    T_test = 320.0
    assert calculate_transit_rate_per_us(temperature_k=T_test) == \
        pytest.approx(_oracle(T_test), rel=1e-12)


# ---------------------------------------------------------------------------
# TEST 3 — production-temperature ownership (main regression guard)
# ---------------------------------------------------------------------------
def test_default_tracks_patched_production_temperature(configured_atom, monkeypatch):
    # Patch ONLY the production owner. The duplicated constants value (300.15)
    # is deliberately left untouched; if the helper ever reverted to it, the
    # default rate would stay at the 300.15 value and could NOT match the
    # 320 K oracle below — so this positive check is the regression guard.
    monkeypatch.setattr(params, "VAPOR_CELL_TEMPERATURE_K", 320.0)
    expected = _oracle(320.0)
    assert calculate_transit_rate_per_us() == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# TEST 4 — the oracle is NOT the duplicated constant
# ---------------------------------------------------------------------------
def test_oracle_is_not_duplicated_constant(configured_atom):
    """Structural guard: this module derives expectations exclusively from
    the PRODUCTION temperature owner (parameters), never from the duplicated
    constants.VAPOR_CELL_TEMPERATURE_K. (The project k_B constant is a
    different quantity and is explicitly allowed.)"""
    import inspect
    src = inspect.getsource(__import__(__name__))
    assert "const.VAPOR_CELL_TEMPERATURE_K" not in src
    assert "constants.VAPOR_CELL_TEMPERATURE_K" not in src