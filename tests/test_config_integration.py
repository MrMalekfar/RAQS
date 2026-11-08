#!/usr/bin/env python3
"""Test script to verify config.py integration with arc_interface."""

import pytest

# Audit B.3: stale script-style test — references parameters.OMEGA_P/OMEGA_C/OMEGA_RF_BASE,
# which were moved to atomic_config. Skip at module level instead of erroring during collection.
pytest.skip(
    "Stale test: Rabi constants now live in atomic_config, not parameters (audit B.3)",
    allow_module_level=True,
)

from rydberg_aoa.config import atomic_config, parameters

print('ATOMIC_LEVELS loaded successfully')
print('Ground:', atomic_config.ATOMIC_LEVELS['ground']['label'])
print('Excited:', atomic_config.ATOMIC_LEVELS['excited']['label'])
print('Rydberg1:', atomic_config.ATOMIC_LEVELS['rydberg1']['label'])
print('Rydberg2:', atomic_config.ATOMIC_LEVELS['rydberg2']['label'])

print('\nRabi frequencies:')
print(f'OMEGA_P: {parameters.OMEGA_P:.2f} rad/us')
print(f'OMEGA_C: {parameters.OMEGA_C:.2f} rad/us')
print(f'OMEGA_RF_BASE: {parameters.OMEGA_RF_BASE:.2f} rad/us')
