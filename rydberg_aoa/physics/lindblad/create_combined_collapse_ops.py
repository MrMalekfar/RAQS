import qutip as qt
from .create_natural_decay_collapse_ops import create_natural_decay_collapse_ops
from .calculate_dephasing_rates import create_laser_linewidth_collapse_ops

def create_combined_collapse_ops():
    """
    Aggregate all collapse operators for the Lindblad Master Equation.

    Returns:
        list: A single list containing both natural decay (jump) operators 
              and laser linewidth (dephasing) operators.
    """
    # 1. Get natural decay operators (Spontaneous Emission)
    # These are of form sqrt(Gamma) * |lower><upper|
    collapse_natural = create_natural_decay_collapse_ops()

    # 2. Get laser linewidth operators (Dephasing)
    # These are of form sqrt(gamma) * |state><state|
    collapse_laser = create_laser_linewidth_collapse_ops()

    # 3. Combine them into a single list
    # We do NOT sum them element-wise. They are distinct physical processes.
    all_collapse_ops = collapse_natural + collapse_laser

    return all_collapse_ops