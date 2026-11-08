#!/usr/bin/env python3
from rydberg_aoa.arc import ARCInterface
from arc import Rubidium85

# Test the arc_interface search
interface = ARCInterface(Rubidium85())

# Find excited state first
excited = interface.find_excited_state(780.0)
print(f'Excited state: {excited}')

if excited:
    transitions = interface.find_rydberg_transitions(
        excited,  # excited state
        480.0,   # target wavelength
        1.0,     # tolerance
        (40, 80) # n range
    )
    print(f'arc_interface found {len(transitions)} transitions')
    for t in transitions[:3]:  # Show first 3
        print(f'  {t.level1.label} -> {t.level2.label} at {t.wavelength_nm:.2f} nm, dipole={t.dipole_moment:.2e}')
else:
    print('No excited state found')
