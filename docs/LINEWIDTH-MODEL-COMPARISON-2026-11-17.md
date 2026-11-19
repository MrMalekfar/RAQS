# Laser linewidth model comparison — 2026-11-17

This historical comparison branch depends on the review fixes, using refined DoA peaks and tighter temporal defaults to resolve small differences. Profile defaults, schema and the main execution path retain their existing model.

## Models

- **Per-level model.** Probe and coupling linewidths each enter one level's pure-dephasing budget, `[g_probe, g_coupling, g_rf] + collision`, through `√g_k |k⟩⟨k|` on levels 2, 3 and 4.
- **Field phase diffusion.** Optical linewidth enters once through the existing `create_laser_phase_diffusion_ops`. Probe phase spans levels 2–4 and coupling phase spans levels 3–4. The remaining per-level budget contains collisions and RF linewidth.

The coherence contributions below use `g = 2π·FWHM` and are checked by `tests/test_linewidth_model_comparison.py`.

| Coherence | Per-level | Field phase diffusion |
| --- | --- | --- |
| ρ₁₂ | g_probe/2 | g_probe/2 |
| ρ₁₃, dark EIT coherence | g_coupling/2 | (g_probe + g_coupling)/2 |
| ρ₂₃ | (g_probe + g_coupling)/2 | g_coupling/2 |
| ρ₃₄ | g_coupling/2 | 0 |

## Method

`rydberg_aoa/scripts/compare_linewidth_models.py` performs three steps.

1. Run the ordinary pipeline at the requested linewidths to obtain the per-level result.
2. Run the same pipeline with zero optical linewidth and record its exact spectrum calls and sensing tasks. Those calls carry only collision/RF dephasing.
3. Replay those calls through `simulation/noise_seam.py` with field phase-diffusion operators.

Replaying step 1 without a noise model must reproduce the pipeline arrays. Spectrum and IF replay errors were exactly zero in every reported case.

```
python -m rydberg_aoa.scripts.compare_linewidth_models \
    --profile examples/lab007/prescribed_full.json \
    --fwhm-mhz profile 0.5 --output-dir out/linewidth
```

## Results

The two inputs were `examples/lab007/prescribed_full.json` and `experimental_full.json`, using a 41-point ±10 MHz scan specifically for this comparison. The stationary spectrum uses adaptive Doppler. IF excludes temporal Doppler, following the profiles. Both inputs produced the same comparison values. Full data are in `docs/linewidth_comparison/`.

Profile rates are `linewidth_rates_per_us = [0.006283, 0.006283, 0]`, corresponding to 1 kHz optical FWHM. The reviewed examples with linewidth enabled used that setting at the time of the comparison.

| Quantity | 1 kHz per-level | 1 kHz field | 0.5 MHz per-level | 0.5 MHz field |
| --- | --- | --- | --- | --- |
| RF-off EIT transmission peak | 0.87661 | 0.87658 | 0.85937 | 0.84944 |
| RF-on center transmission | 0.74520 | 0.74520 | 0.73800 | 0.74224 |
| Maximum transmission difference, RF off/on | — | `2.3e-5` / `4.0e-5` | — | `9.9e-3` / `1.5e-2` |
| IF amplitude ratio, field/per-level | — | 1.00025 | — | 1.0867 |
| Common IF phase shift | — | `0.0003°` | — | `0.183°` |
| Relative phase error, paths 2 and 3 | `-0.000199°`, `-0.000051°` | `-0.000199°`, `-0.000051°` | `-0.000203°`, `-0.000049°` | `-0.000172°`, `-0.000042°` |
| Phase-beamforming angle, target `-30°` | `-30.0000374°` | `-30.0000374°` | `-30.0000359°` | `-30.0000309°` |

Both table estimates use phase beamforming to isolate the atomic-model change. The experimental profile's own MUSIC estimate is recorded separately in JSON.

## Interpretation and limits

At 1 kHz, maximum transmission difference is below `4e-5`, IF amplitude differs by about 0.025%, and the displayed angle is unchanged. At 0.5 MHz, the EIT peak decreases by about 0.01, IF amplitude rises by about 8.7%, and common phase shifts by about 0.18°. Angle differences remain below `1e-5°` in these cases, where amplitude/phase changes are common across paths.

The phase-diffusion model assumes independent white laser-phase noise. Correlated laser noise and amplitude noise need additional modeling. RF linewidth remains on level 4 in both cases. The harness supports plain sensing tasks, excluding temporal Doppler, RF-drive realizations and physical readout, and rejects unsupported replay. The field-model spectrum uses the reference solver because the noise model disables the affine kernel.

The choice of `lab007` as a thesis reference was an author-independent assumption in the external review. Match it to the actual thesis inputs before extending the conclusion. Both profiles and both linewidth settings were reproduced on Windows on 2026-11-19, with maximum spectral difference `1.1e-16` from the supplied data and zero IF replay error. [Verification record](../tests/README.md#verified-review-branches-2026-11-19)
