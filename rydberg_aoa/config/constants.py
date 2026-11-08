"""
Shared constants for config modules.
"""

import numpy as np
from scipy import constants

# Unit Conversion
US_TO_S_FACTOR = 1e6
VAPOR_CELL_TEMPERATURE_K = 300.15  # 27°C (typical heated vapor cell)

# Physical constants from scipy (NIST CODATA values)
C_LIGHT = constants.c  # Speed of light, m/s
HBAR = constants.hbar  # Reduced Planck's constant, J s
H = constants.h  # Planck's constant, J s
EPSILON_0 = constants.epsilon_0
k_B = 1.38e-23  # J/K (Boltzmann constant)
