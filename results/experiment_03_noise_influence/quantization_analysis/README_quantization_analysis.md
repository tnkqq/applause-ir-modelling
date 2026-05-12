# Quantization analysis for experiment 03

## Purpose

This folder adds a separate analysis of the non-standard behavior of quantization in
`experiment_03_noise_influence`. The original experiment files were not modified.

## Source files used

- `../config.json`
- `../metrics.csv`
- `../noise_type_comparison.csv`
- `../summary.json`
- existing experiment 03 plots were inspected as baseline context

## Fixed parameters

- seed: `3303`
- frame size: `128x96`
- num_frames: `10`
- background_k: `300.0`
- delta_t: `4.0`
- threshold_k: `3.0`
- scene: `circle`, `size=16`, `weak_bg=True`
- reference noise for this isolated quantization analysis: Gaussian sigma `1.0` ADC, no FPN, no defects

## Quantization parameter

The new analysis uses an explicit ADC-code quantization step:

`I_quantized = round(I_reference / quant_step) * quant_step`.

Checked values: `[1, 2, 4, 8, 16, 32, 64]` ADC codes. Equivalent numbers of levels are approximately:
`{step: floor(1023 / step) + 1}`.

This is more explicit than the original experiment 03 `noise_level` abstraction, where the
quantization branch used `quant_bits=max(5, 10-noise_level)`.

## New files

- `quantization_analysis_metrics.csv`
- `quantization_analysis_summary_by_step.csv`
- `quantization_analysis_histogram_stats.csv`
- `quantization_analysis_summary.json`
- `quantization_analysis_snr_like_vs_quant_step.png`
- `quantization_analysis_sigma_bg_vs_quant_step.png`
- `quantization_analysis_contrast_vs_quant_step.png`
- `quantization_analysis_iou_vs_quant_step.png`
- `quantization_analysis_tpr_fpr_vs_quant_step.png`
- `quantization_analysis_precision_f1_vs_quant_step.png`
- `quantization_analysis_unique_levels_vs_quant_step.png`
- `quantization_analysis_histograms_by_quant_step.png`
- `quantization_analysis_frames_comparison.png`
- `quantization_analysis_error_maps.png`
- `quantization_analysis_masks_comparison.png`
- `quantization_analysis_profiles.png`
- `quantization_analysis_background_zoom.png`

## Why quantization can look non-standard

SNR-like is calculated as `(mu_anom - mu_bg) / sigma_bg`. Under coarse quantization, the
background can collapse into a smaller number of discrete digital levels. This can reduce
the estimated `sigma_bg`. If the anomaly-background mean contrast is preserved or decreases
more slowly than `sigma_bg`, SNR-like can artificially increase.

This does not mean that the image or detector becomes better. Coarse quantization removes
radiometric detail, creates stair-step structures, changes local gradients, and can change
the geometry of the detected mask. Therefore quantization must be interpreted using SNR-like
together with IoU, TPR, FPR, Precision, F1-score, histograms, profiles, and visual masks.

## Gaussian noise versus quantization

Gaussian noise is additive random variation and usually increases the background scatter,
which tends to reduce SNR-like. Quantization is not ordinary additive noise: it replaces
smooth values by nearest discrete levels. In some cases it makes the background appear more
uniform in terms of standard deviation, while the image loses radiometric detail.

## Key result

- Maximum SNR-like: `quant_step=16`, SNR-like `infinite/undefined because sigma_bg=0`.
- Maximum IoU: `quant_step=1`, IoU `1.000`.

If these steps differ, the analysis demonstrates why SNR-like alone is not sufficient for
judging quantization quality.
