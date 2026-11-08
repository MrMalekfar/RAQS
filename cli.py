"""
Command-line interface for the Rydberg AoA Simulation.
Allows configuration overrides and different execution modes.
"""

import argparse
import logging
import sys
from typing import Dict, Any

from rydberg_aoa.config.laboratory import AppliedRF
from rydberg_aoa.config.rf_configuration import configure_rf_fields, read_configured_rf_fields


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Rydberg AoA Simulation - Modular Quantum Sensor Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration overrides
    parser.add_argument(
        '--n-elements',
        type=int,
        default=None,
        help='Number of antenna elements'
    )

    parser.add_argument(
        '--target-aoa',
        type=float,
        default=None,
        help='Target angle of arrival in degrees'
    )

    parser.add_argument(
        '--rf-frequency',
        type=float,
        default=None,
        help='Applied signal RF frequency in Hz (also used for AoA geometry)'
    )

    parser.add_argument('--lo-rf-frequency', type=float, default=None,
                        help='Applied LO RF frequency in Hz (RF-first selection carrier)')

    parser.add_argument(
        '--spacing',
        type=float,
        default=None,
        help='Element spacing in meters'
    )

    parser.add_argument(
        '--cell-length',
        type=float,
        default=None,
        help='Vapor cell length in meters'
    )

    # Execution options
    parser.add_argument('--profile', help='Explicit LAB-006 JSON profile; required for public execution')
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Resolve/report declared physics without workers, DoA or plots')
    parser.add_argument('--output-dir', help='Optional directory for run_report.json and sensing.png')
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plotting and only run simulation'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )

    return parser.parse_args()


def override_config(args: argparse.Namespace) -> None:
    """
    Override configuration parameters with command-line arguments.

    Args:
        args: Parsed command-line arguments
    """
    from rydberg_aoa.config import parameters as cfg
    # LAB-002: validate RF views before writes; an explicit override updates
    # signal/LO, geometry frequency and signed IF coherently. No RF tuning.
    lo, signal = read_configured_rf_fields()
    lo_frequency = getattr(args, 'lo_rf_frequency', None)
    new_lo = AppliedRF(lo_frequency if lo_frequency is not None else lo.frequency_hz,
                       lo.electric_field_vm, lo.polarization)
    new_signal = AppliedRF(args.rf_frequency if args.rf_frequency is not None else signal.frequency_hz,
                           signal.electric_field_vm, signal.polarization)
    configure_rf_fields(lo=new_lo, signal=new_signal)
    # Override independent configuration values
    for name, argument in (('N_ELEMENTS', 'n_elements'), ('TARGET_AOA_DEG', 'target_aoa'),
                           ('SPACING', 'spacing'), ('CELL_LENGTH', 'cell_length')):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, name, value)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging
        quiet: Suppress all output
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Save original stdout before redirection
    original_stdout = sys.stdout

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('terminal_log.txt'),        # Log file
            logging.StreamHandler(original_stdout)         # Terminal
        ]
    )



def run_simulation_with_config(show_plots_flag: bool = True) -> Dict[str, Any]:
    """
    Run the simulation with current configuration.

    Args:
        show_plots_flag: Whether to display plots

    Returns:
        Simulation results dictionary
    """
    # Legacy programmatic API; public entry points use explicit LAB-006 profiles.
    from rydberg_aoa.config import atomic_config as acfg
    from rydberg_aoa.config import parameters as cfg
    from rydberg_aoa.simulation.runner import SimulationRunner
    from rydberg_aoa.visualization.plotting import create_summary_plot, plot_analytical_spectrum, show_plots
    logger = logging.getLogger(__name__)

    try:
        # ARCH-001: explicit atomic initialization (no import-time physics).
        acfg.initialize_atomic_physics()
        logger.info("Starting Rydberg AoA Simulation")
        logger.info(f"Configuration: N_ELEMENTS={cfg.N_ELEMENTS}, "
                   f"TARGET_AOA={cfg.TARGET_AOA_DEG}°, "
                   f"RF_FREQ={cfg.RF_FREQUENCY/1e9:.1f} GHz, "
                   f"CELL_LENGTH={cfg.CELL_LENGTH*100:.1f} cm")

        # Initialize and run simulation
        runner = SimulationRunner()
        results = runner.run_complete_simulation()

        if show_plots_flag:
            logger.info("Creating numerical simulation plots")
            import numpy as np
            # DOA-INTEGRATE-001: search-free estimators (esprit,
            # root_music) produce no angular spectrum; the summary panel
            # then renders the estimator result (never a fabricated
            # spectrum).
            _power = results['power']
            _power_norm = (
                _power / np.max(_power) if _power is not None else None
            )
            _doa = results.get('doa_result')
            fig_numerical = create_summary_plot(
                dp_scan=results['dp_scan'],
                trans_no_rf=results['trans_no_rf'],
                trans_with_rf=results['trans_with_rf'],
                omega_rf_base=acfg.OMEGA_RF_BASE,   # OMEGA_* live in atomic_config (RF-005)
                xf=results['xf_0'],
                yf=results['yf_0'],
                if_freq=cfg.IF_FREQ,
                tlist=results['tlist'],
                time_traces=results['time_traces'],
                array_size=cfg.N_ELEMENTS,
                rf_frequency=cfg.RF_FREQUENCY,
                angles=results['angles'],
                power=_power_norm,
                true_aoa=results['true_aoa'],
                estimated_aoa=results['estimated_aoa'],
                doa_algorithm=getattr(_doa, 'algorithm', None),
                doa_result=_doa
            )

            logger.info("Creating analytical spectrum plot")
            import matplotlib.pyplot as plt
            fig_analytical = plt.figure(figsize=(10, 6))
            plot_analytical_spectrum(
                results['dp_scan'],
                results['trans_no_rf_analytic'],
                results['trans_with_rf_analytic'],
                acfg.OMEGA_RF_BASE   # OMEGA_* live in atomic_config (RF-005)
            )
            show_plots()

        logger.info("Simulation completed successfully")
        return results

    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        raise


def main():
    """Explicit profile entry point; no implicit legacy or magnetic fallback."""
    # Parse help/errors before opening the run log. A read-only CLI request
    # must not truncate a previous diagnostic session.
    args = parse_arguments()
    from rydberg_aoa.exceptions import PhysicsConfigurationError
    if not args.profile:
        raise PhysicsConfigurationError('LAB-006: --profile is required; declare prepared mJ/q and hardware bounds explicitly. '
                                        'See examples/lab006/ and --help. No legacy fallback.')
    overrides = ('n_elements', 'target_aoa', 'rf_frequency', 'lo_rf_frequency', 'spacing', 'cell_length')
    if any(getattr(args, name) is not None for name in overrides):
        raise PhysicsConfigurationError('LAB-006: edit the explicit profile instead of combining legacy field/geometry overrides.')
    from rydberg_aoa.config.run_profile import load_run_profile
    from rydberg_aoa.simulation.profile_execution import execute_profile, json_default, save_profile_results
    import json
    from pathlib import Path
    profile = load_run_profile(args.profile)
    logging.basicConfig(level=logging.ERROR if args.quiet else (logging.DEBUG if args.verbose else logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(message)s')
    if profile.study is not None:
        if args.diagnostics_only:
            raise PhysicsConfigurationError('NOISE-008: diagnostics-only cannot execute a study; validate through the study API.')
        from rydberg_aoa.simulation.study_execution import execute_study
        study = execute_study(profile)
        report = study.report()
        if args.output_dir:
            from rydberg_aoa.simulation.study_output import save_study_results
            save_study_results(args.output_dir,study)
        print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        if not args.no_plots:
            from rydberg_aoa.visualization.study_figures import create_study_figures
            from rydberg_aoa.visualization.profile_figures import create_profile_figures, save_profile_figures
            figures = create_study_figures(study)
            if args.output_dir:
                save_profile_figures(args.output_dir,figures)
            # Keep the same standalone representative figures, explicitly trial 0.
            for point in study.points[-1:]:
                point_figures = create_profile_figures(point.report,point.result)
                if args.output_dir:
                    save_profile_figures(Path(args.output_dir)/f'point_{point.index:04d}',point_figures)
            import matplotlib.pyplot as plt
            plt.show()
        return report,study
    report, result = execute_profile(profile, diagnostics_only=args.diagnostics_only)
    encoded = json.dumps(report, default=json_default, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output_dir:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'run_report.json').write_text(encoded + '\n', encoding='utf-8')
        if result is not None:
            save_profile_results(directory, result)
    print(encoded)
    if result is not None and not args.no_plots:
        from rydberg_aoa.visualization.profile_figures import create_profile_figures, save_profile_figures
        figures = create_profile_figures(report, result)
        if args.output_dir:
            save_profile_figures(directory, figures)
        import matplotlib.pyplot as plt
        plt.show()
    return report, result


if __name__ == "__main__":
    main()
