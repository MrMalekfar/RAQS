#!/usr/bin/env python3
from arc import Rubidium85

atom = Rubidium85()
print('Searching for Rb 5P3/2 transitions around 480 nm:')
target_wl = 480.0
tolerance = 5.0

found_transitions = []
for n in range(45, 80):
    for l in range(min(4, n)):
        for j in [l-0.5, l+0.5] if l > 0 else [0.5]:
            if j < 0: continue
            try:
                wl = atom.getTransitionWavelength(5, 1, 1.5, n, l, j) * 1e9
                if abs(wl - target_wl) <= tolerance:
                    found_transitions.append((n, l, j, wl))
                    print(f'Found: 5P3/2 -> {n}{"SPDF"[l]}{int(2*j+1)}/2 at {wl:.2f} nm')
            except:
                pass

print(f'Total transitions found within {tolerance} nm of {target_wl} nm: {len(found_transitions)}')

if not found_transitions:
    print('No transitions found! Checking broader range...')
    for n in range(40, 90):
        try:
            wl = atom.getTransitionWavelength(5, 1, 1.5, n, 0, 0.5) * 1e9  # S states
            if 400 < wl < 600:
                print(f'{n}S1/2: {wl:.1f} nm')
        except:
            pass
