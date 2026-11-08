"""Rydberg AoA entry point: share the explicit LAB-006 CLI contract.

Importing this file creates no logs, plots, ARC atom or initialized physics.
Run --help, or supply --profile with a prepared-channel declaration.
"""


def main():
    from cli import main as cli_main
    return cli_main()


if __name__ == '__main__':
    main()
