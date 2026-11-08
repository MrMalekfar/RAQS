"""
I/O utilities for logging and result saving.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO,
                  format_str: str = '%(asctime)s - %(levelname)s - %(message)s',
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level
        format_str: Log format string
        log_file: Optional log file path

    Returns:
        Root logger
    """
    logging.basicConfig(
        level=level,
        format=format_str
    )

    # Redirect stdout/stderr to log file if specified
    if log_file:
        log_file_handle = open(log_file, 'w')
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle

    return logging.getLogger()


def save_results_to_file(results: dict, filename: str):
    """
    Save simulation results to a file.

    Args:
        results: Dictionary of results
        filename: Output filename
    """
    import numpy as np

    with open(filename, 'w') as f:
        f.write("Rydberg AoA Simulation Results\n")
        f.write("=" * 40 + "\n\n")

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                f.write(f"{key}: array of shape {value.shape}\n")
            else:
                f.write(f"{key}: {value}\n")
