#!/usr/bin/env python3
"""Debug script for dipole moment calculations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rydberg_aoa import arc
from arc import Rubidium85

def debug_dipole_calculations():
    """Debug why coupling and RF Rabi frequencies are 0."""

    print("=== Debugging Dipole Calculations ===")

    # Initialize ARC
    atom = Rubidium85()
    interface = arc.ARCInterface(atom)

    # Define the levels from the current output
    ground = interface.ground_state
    excited = arc.AtomicLevel(n=5, l=1, j=1.5, f=3, mf=0, label="5P3/2 F=3")
    rydberg1 = arc.AtomicLevel(n=50, l=2, j=2.5, f=5.0, mf=0, label="50D5/2 F=5")
    rydberg2 = arc.AtomicLevel(n=51, l=1, j=1.5, f=6.0, mf=0, label="51P3/2 F=6")

    print(f"Ground: n={ground.n}, l={ground.l}, j={ground.j}, f={ground.f}")
    print(f"Excited: n={excited.n}, l={excited.l}, j={excited.j}, f={excited.f}")
    print(f"Rydberg1: n={rydberg1.n}, l={rydberg1.l}, j={rydberg1.j}, f={rydberg1.f}")
    print(f"Rydberg2: n={rydberg2.n}, l={rydberg2.l}, j={rydberg2.j}, f={rydberg2.f}")
    print()

    # Test dipole calculations with different methods and polarizations
    transitions = [
        ("Ground->Excited", ground, excited),
        ("Excited->Rydberg1", excited, rydberg1),
        ("Rydberg1->Rydberg2", rydberg1, rydberg2),
    ]

    for name, level1, level2 in transitions:
        print(f"=== {name} Transition ===")

        # Test different polarizations
        for q in [0, 1]:
            print(f"Polarization q={q}:")

            # Test HFS method
            try:
                if level1.f is not None and level2.f is not None:
                    dipole_hfs = atom.getDipoleMatrixElementHFS(
                        level1.n, level1.l, level1.j, level1.f, level1.mf,
                        level2.n, level2.l, level2.j, level2.f, level2.mf,
                        q=q
                    )
                    dipole_hfs_si = abs(dipole_hfs) * 8.478e-30
                    print(f"  HFS method: {dipole_hfs:.2e} (ea0) -> {dipole_hfs_si:.2e} C*m")
                else:
                    print("  HFS method: N/A (missing f values)")
            except Exception as e:
                print(f"  HFS method failed: {e}")

            # Test regular method
            try:
                dipole_reg = atom.getDipoleMatrixElement(
                    level1.n, level1.l, level1.j, 0,  # mj1
                    level2.n, level2.l, level2.j, 0,  # mj2
                    q=q
                )
                dipole_reg_si = abs(dipole_reg) * 8.478e-30
                print(f"  Regular method: {dipole_reg:.2e} (ea0) -> {dipole_reg_si:.2e} C*m")
            except Exception as e:
                print(f"  Regular method failed: {e}")

        print()

    # Test wavelength calculations
    print("=== Wavelength Calculations ===")
    try:
        wl_ge = atom.getTransitionWavelength(ground.n, ground.l, ground.j, excited.n, excited.l, excited.j) * 1e9
        print(f"Ground->Excited wavelength: {wl_ge:.2f} nm")
    except Exception as e:
        print(f"Ground->Excited wavelength failed: {e}")

    try:
        wl_er = atom.getTransitionWavelength(excited.n, excited.l, excited.j, rydberg1.n, rydberg1.l, rydberg1.j) * 1e9
        print(f"Excited->Rydberg1 wavelength: {wl_er:.2f} nm")
    except Exception as e:
        print(f"Excited->Rydberg1 wavelength failed: {e}")

    try:
        wl_rr = atom.getTransitionWavelength(rydberg1.n, rydberg1.l, rydberg1.j, rydberg2.n, rydberg2.l, rydberg2.j) * 1e9
        print(f"Rydberg1->Rydberg2 wavelength: {wl_rr:.2f} nm")
    except Exception as e:
        print(f"Rydberg1->Rydberg2 wavelength failed: {e}")

if __name__ == "__main__":
    debug_dipole_calculations()
