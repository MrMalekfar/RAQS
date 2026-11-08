"""Real public profile entrypoints produce standalone publication artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['prescribed', 'experimental'])
def test_actual_profile_cli_saves_all_standalone_figures(tmp_path, mode):
    data = json.loads((ROOT / 'examples/lab007' / f'{mode}_full.json').read_text())
    data['spectrum']['probe_offsets_mhz'] = [-.5, .5]
    source = tmp_path / 'profile.json'
    source.write_text(json.dumps(data))
    process = subprocess.run([sys.executable, '-B', 'main.py', '--profile', str(source),
                              '--output-dir', str(tmp_path)], cwd=ROOT,
                             env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'},
                             capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((tmp_path / 'run_report.json').read_text(encoding='utf-8'))
    assert report['mode'] == mode
    assert report['sensing']['true_aoa_deg'] == report['sensing']['estimated_aoa_deg'] == -30.
    for name in ('energy_levels', 'fig_time_domain', 'fig_phase_progression', 'fig_aoa_spectrum',
                 'fig_spectrum', 'fig_analytical'):
        assert (tmp_path / f'{name}.png').read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / f'{name}.pdf').read_bytes().startswith(b'%PDF')
    with np.load(tmp_path / 'simulation_arrays.npz', allow_pickle=False) as arrays:
        assert arrays['time_traces'].shape == (3,601)
        assert np.iscomplexobj(arrays['complex_responses'])
        assert np.isfinite(arrays['spectrum_rf_on']).all()
