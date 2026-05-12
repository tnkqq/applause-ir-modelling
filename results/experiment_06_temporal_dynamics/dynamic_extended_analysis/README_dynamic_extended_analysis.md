# Dynamic extended analysis for experiment 06

## Why this analysis was added

This folder extends experiment 06 without modifying its fixed results. The goal is to
show temporal dynamics, sensor inertia, and frame averaging more clearly, especially for
moving, appearing, fading, intermittent, and drifting-background scenes.

## Fixed color scale

The original montage-style images can be visually misleading if every subplot uses its own
automatic color normalization. In that case the same background ADC value may appear with
different colors in different frames. All new sequence figures use a fixed color scale:
`vmin` and `vmax` are computed once for the whole compared set, then passed to every
`imshow`. A shared colorbar is added to every fixed-scale montage.

## Scenarios

The generated scenarios are:

- `static`
- `appearing`
- `moving`
- `appearing_and_growing`
- `appearing_and_fading`
- `moving_fast`
- `moving_diagonal`
- `two_anomalies_static_dynamic`
- `intermittent`
- `background_drift`
- `moving_with_noise_burst`
- `small_moving_target`

## Inertia and averaging models

The inertia model is:

`I_alpha[t] = (1 - alpha) * I[t] + alpha * I_alpha[t-1]`.

The frame averaging model is:

`I_avg[t] = mean(I_alpha[max(0,t-W+1):t+1])`.

At the beginning of the sequence the window is truncated to available frames.

## Metrics

For every frame the script calculates SNR-like, TP, FP, TN, FN, TPR, FPR, Precision,
F1-score, IoU, detection flag, target presence, target center, peak amplitude, background
mean and background standard deviation. For moving scenarios it also calculates center
error in pixels.

## New files

- `tables/dynamic_extended_metrics_per_frame.csv`
- `tables/dynamic_extended_summary_by_scenario_alpha_window.csv`
- `tables/dynamic_extended_summary_by_scenario.csv`
- `tables/dynamic_extended_delay_table.csv`
- `tables/dynamic_extended_tracking_error.csv`
- `dynamic_extended_summary.json`
- compressed source, inertial, processed and mask sequences in `frames/` and `masks/`
- all PNG figures in `figures/`

## Recommended figures for the diploma

- `figures/dynamic_extended_appearing_frames_fixed_scale.png`
- `figures/dynamic_extended_moving_frames_fixed_scale.png`
- `figures/dynamic_extended_alpha_comparison_appearing.png`
- `figures/dynamic_extended_window_comparison_moving.png`
- `figures/dynamic_extended_overlay_moving_window10.png`
- `figures/dynamic_extended_tradeoff_snr_delay.png`
- `figures/dynamic_extended_heatmap_iou_scenario_alpha_window.png`

## Engineering interpretation

For a static object, frame averaging is usually useful because it reduces random noise and
raises SNR-like. For an appearing object, a large averaging window and high inertia increase
detection delay. For a moving object, large windows and high inertia degrade localization,
reduce IoU, and increase center error. Background drift tests whether the global threshold
remains stable when the mean level slowly changes. Intermittent and noise-burst scenarios
show that maximum SNR-like is not enough for dynamic tasks: delay and mask coincidence are
equally important.

The hardest scenario by best IoU in this run was `appearing_and_growing`
with best IoU `0.930`. The largest measured
delay occurred for `moving_with_noise_burst` at alpha `0.9`
and window `3`.

## Files intentionally not modified

The script does not modify the original experiment 06 `config.json`, `summary.json`,
`metrics.csv`, PNG figures, README, masks, or reports. All new results are isolated in
`dynamic_extended_analysis/`.
