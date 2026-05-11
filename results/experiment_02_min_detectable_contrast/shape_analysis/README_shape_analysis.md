# Shape analysis for experiment 02

## Purpose

This folder adds a separate analysis of how anomaly shape affects detection quality in `experiment_02_min_detectable_contrast`.
The original experiment files were not modified. All new artifacts are written only to `shape_analysis/`.

## Source files used

- `../config.json`
- `../summary.json`
- `../metrics.csv`
- existing masks in `../masks/` were inspected as fixed ground-truth products

## Fixed parameters

- seed: `3202`
- frame size: `128x96`
- num_frames: `12`
- background_k: `300.0`
- noise_sigma: `2.2`
- fpn_std: `0.7`
- threshold_k: `3.0`
- iou_success: `0.3`
- detector: `global median + 3.0 * robust_sigma`
- delta_t_K values: `[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]`
- shapes: `['circle', 'rectangle', 'gaussian']`

## New files

- `shape_analysis_metrics_by_shape_delta_t.csv`
- `shape_analysis_detection_thresholds_by_shape.csv`
- `shape_analysis_summary.json`
- `shape_analysis_iou_vs_delta_t_by_shape.png`
- `shape_analysis_detection_probability_vs_delta_t_by_shape.png`
- `shape_analysis_tpr_vs_delta_t_by_shape.png`
- `shape_analysis_f1_vs_delta_t_by_shape.png`
- `shape_analysis_snr_like_vs_delta_t_by_shape.png`
- `shape_analysis_iou_heatmap_shape_delta_t.png`
- `shape_analysis_detection_probability_heatmap_shape_delta_t.png`
- `figures/shape_analysis_ground_truth_temperature_fields.png`
- `figures/shape_analysis_ground_truth_masks.png`
- `figures/shape_analysis_prediction_comparison_delta_t_1p0.png`
- `figures/shape_analysis_prediction_comparison_delta_t_1p5.png`
- `figures/shape_analysis_prediction_comparison_delta_t_2p0.png`
- `figures/shape_analysis_prediction_comparison_delta_t_3p0.png`
- `figures/shape_analysis_gaussian_profile_explanation.png`

## Gaussian interpretation

For `gaussian`, the temperature field is smooth:

`T(x,y)=T_bg + delta_t * exp(-((x-x0)^2+(y-y0)^2)/(2*sigma_g^2))`.

The binary ground-truth mask is not the same as the temperature field. The project logic uses `gaussian_hotspot()`, where
`sigma_g=max(1.0, size/3)`, and the mask is defined as `weights >= 0.5`. In experiment 02, `size=14`.

## Detection thresholds by shape

The threshold criterion is `detection_probability >= 0.9` and mean `IoU > 0.3`.

- `circle`: `1.5`
- `rectangle`: `1.5`
- `gaussian`: `2.0`

## Diploma interpretation

Circle and rectangle anomalies have sharper boundaries. The gaussian anomaly has a smooth temperature profile, so its peripheral
pixels have lower local contrast than the center. This makes threshold segmentation harder near the detection limit: at low
`delta_t_K`, the detector tends to recover only the central high-contrast part. Shape therefore affects `IoU`, `TPR`, `F1-score`,
and detection probability even when all other sensor parameters are fixed.
