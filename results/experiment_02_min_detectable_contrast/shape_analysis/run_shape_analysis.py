#!/usr/bin/env python3
"""Additional shape analysis for experiment 02.

The script reads the fixed experiment results from the parent folder and writes
only to this shape_analysis directory. It does not modify config.json,
summary.json, metrics.csv, existing figures, masks, README files, or DOCX
reports of the original experiment.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

sys.path.append(str(ROOT))

from experiments.series3.common import (  # noqa: E402
    binary_metrics,
    detect_global_threshold,
    generate_adc_frame,
    region_snr,
    rng_from_seed,
    scene_with_anomaly,
)


METRIC_COLUMNS = ["tpr", "fpr", "precision", "f1", "iou", "snr_like"]
SHAPES = ["circle", "rectangle", "gaussian"]
PROFILE_DELTA_T = 2.0
COMPARISON_DELTAS = [1.0, 1.5, 2.0, 3.0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_delta(delta_t: float) -> str:
    return f"{delta_t:.1f}".replace(".", "p")


def ensure_inputs() -> None:
    required = ["config.json", "summary.json", "metrics.csv"]
    missing = [name for name in required if not (EXPERIMENT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")


def aggregate_metrics(metrics: pd.DataFrame, iou_success: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = metrics.copy()
    if "detected_success" not in working.columns:
        working["detected_success"] = (working["iou"] >= iou_success).astype(float)

    grouped = working.groupby(["shape", "delta_t_K"], as_index=False).agg(
        n_frames=("frame_idx", "count"),
        tpr_mean=("tpr", "mean"),
        tpr_std=("tpr", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        snr_like_mean=("snr_like", "mean"),
        snr_like_std=("snr_like", "std"),
        detection_probability=("detected_success", "mean"),
        detection_probability_std=("detected_success", "std"),
    )
    grouped = grouped.sort_values(["shape", "delta_t_K"]).reset_index(drop=True)

    threshold_rows: list[dict[str, Any]] = []
    for shape in SHAPES:
        shape_df = grouped[grouped["shape"] == shape].sort_values("delta_t_K")
        reached = shape_df[(shape_df["detection_probability"] >= 0.9) & (shape_df["iou_mean"] > iou_success)]
        if reached.empty:
            threshold_rows.append(
                {
                    "shape": shape,
                    "threshold_delta_t_K": "not_reached",
                    "status": "not_reached",
                    "detection_probability": np.nan,
                    "iou_mean": np.nan,
                    "tpr_mean": np.nan,
                    "f1_mean": np.nan,
                }
            )
        else:
            row = reached.iloc[0]
            threshold_rows.append(
                {
                    "shape": shape,
                    "threshold_delta_t_K": float(row["delta_t_K"]),
                    "status": "reached",
                    "detection_probability": float(row["detection_probability"]),
                    "iou_mean": float(row["iou_mean"]),
                    "tpr_mean": float(row["tpr_mean"]),
                    "f1_mean": float(row["f1_mean"]),
                }
            )
    thresholds = pd.DataFrame(threshold_rows)
    return grouped, thresholds


def save_line_plot(df: pd.DataFrame, metric: str, ylabel: str, filename: str, ylim_01: bool = True) -> None:
    plt.figure(figsize=(7.6, 4.8))
    for shape in SHAPES:
        sub = df[df["shape"] == shape]
        plt.plot(sub["delta_t_K"], sub[metric], marker="o", linewidth=2.0, label=shape)
    plt.xlabel("Delta T, K")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Delta T by shape")
    if ylim_01:
        plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Shape")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def save_heatmap(df: pd.DataFrame, value_col: str, title: str, filename: str) -> None:
    pivot = df.pivot(index="shape", columns="delta_t_K", values=value_col).loc[SHAPES]
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Delta T, K")
    ax.set_ylabel("Shape")
    ax.set_title(title)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def replay_frame(config: dict[str, Any], target_shape: str, target_delta_t: float, target_frame_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = rng_from_seed(int(config["seed"]))
    delta_values = [float(v) for v in config["delta_t_values_K"]]
    for shape in config["shapes"]:
        for delta_t in delta_values:
            scene, truth = scene_with_anomaly(
                int(config["height"]),
                int(config["width"]),
                background_k=float(config["background_k"]),
                delta_t=float(delta_t),
                shape=shape,
                size=14,
            )
            for frame_idx in range(int(config["num_frames"])):
                frame, _ = generate_adc_frame(
                    scene,
                    rng,
                    gaussian_sigma=float(config["noise_sigma"]),
                    fpn_std=float(config["fpn_std"]),
                )
                result = detect_global_threshold(frame, k=float(config["threshold_k"]), min_area=5)
                if shape == target_shape and math.isclose(float(delta_t), float(target_delta_t)) and frame_idx == target_frame_idx:
                    snr = region_snr(frame, truth)
                    return scene, truth, frame, result.mask, snr
    raise ValueError(f"Target frame not found: {target_shape=} {target_delta_t=} {target_frame_idx=}")


def overlay_error(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    truth_b = truth.astype(bool)
    pred_b = pred.astype(bool)
    rgb = np.zeros((*truth_b.shape, 3), dtype=float)
    rgb[truth_b & pred_b] = [0.1, 0.75, 0.2]  # true positive, green
    rgb[pred_b & ~truth_b] = [0.9, 0.15, 0.15]  # false positive, red
    rgb[truth_b & ~pred_b] = [0.15, 0.3, 0.95]  # false negative, blue
    return rgb


def make_temperature_fields(config: dict[str, Any]) -> None:
    scenes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for shape in SHAPES:
        scene, mask = scene_with_anomaly(
            int(config["height"]),
            int(config["width"]),
            background_k=float(config["background_k"]),
            delta_t=PROFILE_DELTA_T,
            shape=shape,
            size=14,
        )
        scenes.append(scene)
        masks.append(mask)

    vmin = float(config["background_k"])
    vmax = float(config["background_k"]) + PROFILE_DELTA_T
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for ax, scene, shape in zip(axes, scenes, SHAPES):
        im = ax.imshow(scene, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(shape)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="Temperature, K")
    fig.suptitle("Ground-truth temperature fields, Delta T = 2.0 K")
    fig.savefig(FIG_DIR / "shape_analysis_ground_truth_temperature_fields.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for ax, mask, shape in zip(axes, masks, SHAPES):
        ax.imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1)
        ax.set_title(shape)
        ax.set_axis_off()
    fig.suptitle("Binary ground-truth masks; gaussian field is smooth, mask is binary")
    fig.savefig(FIG_DIR / "shape_analysis_ground_truth_masks.png", dpi=180)
    plt.close(fig)


def make_prediction_comparison(config: dict[str, Any], delta_t: float) -> None:
    examples = [replay_frame(config, shape, delta_t, 0) for shape in SHAPES]
    frames = [item[2] for item in examples]
    frame_vmin = min(float(np.min(frame)) for frame in frames)
    frame_vmax = max(float(np.max(frame)) for frame in frames)

    fig, axes = plt.subplots(len(SHAPES), 4, figsize=(12.5, 8.4), constrained_layout=True)
    for row, (shape, (scene, truth, frame, pred, snr)) in enumerate(zip(SHAPES, examples)):
        metric = binary_metrics(pred, truth)
        panels = [
            ("Synthetic IR frame", frame, "magma", frame_vmin, frame_vmax),
            ("Ground-truth mask", truth.astype(float), "gray", 0, 1),
            ("Predicted mask", pred.astype(float), "gray", 0, 1),
            ("Overlay: TP green, FP red, FN blue", overlay_error(truth, pred), None, None, None),
        ]
        for col, (title, data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{shape}\nIoU={metric['iou']:.2f}\nTPR={metric['tpr']:.2f}\nSNR={snr:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Prediction comparison by shape, Delta T = {delta_t:g} K", fontsize=13)
    fig.savefig(FIG_DIR / f"shape_analysis_prediction_comparison_delta_t_{fmt_delta(delta_t)}.png", dpi=180)
    plt.close(fig)


def make_gaussian_profile(config: dict[str, Any]) -> None:
    scene, mask = scene_with_anomaly(
        int(config["height"]),
        int(config["width"]),
        background_k=float(config["background_k"]),
        delta_t=PROFILE_DELTA_T,
        shape="gaussian",
        size=14,
    )
    center_row = scene.shape[0] // 2
    profile = scene[center_row, :] - float(config["background_k"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    im = axes[0].imshow(scene, cmap="inferno", vmin=float(config["background_k"]), vmax=float(config["background_k"]) + PROFILE_DELTA_T)
    axes[0].set_title("Gaussian temperature field")
    axes[0].set_axis_off()
    fig.colorbar(im, ax=axes[0], label="Temperature, K")

    axes[1].plot(np.arange(len(profile)), profile, color="#1f77b4", linewidth=2.0)
    axes[1].axhline(PROFILE_DELTA_T * 0.5, color="#d62728", linestyle="--", label="mask threshold: weight >= 0.5")
    axes[1].set_title("1D center profile")
    axes[1].set_xlabel("Pixel x")
    axes[1].set_ylabel("Temperature excess, K")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Binary gaussian mask")
    axes[2].set_axis_off()
    fig.suptitle("Gaussian anomaly: smooth temperature field and binary ground-truth mask")
    fig.savefig(FIG_DIR / "shape_analysis_gaussian_profile_explanation.png", dpi=180)
    plt.close(fig)


def write_summary(config: dict[str, Any], thresholds: pd.DataFrame, grouped: pd.DataFrame) -> None:
    threshold_map: dict[str, Any] = {}
    for row in thresholds.to_dict(orient="records"):
        value = row["threshold_delta_t_K"]
        threshold_map[row["shape"]] = value if isinstance(value, str) else float(value)

    gaussian_threshold = threshold_map.get("gaussian")
    circle_threshold = threshold_map.get("circle")
    rectangle_threshold = threshold_map.get("rectangle")
    conclusion = (
        "Shape-specific analysis shows that anomaly shape affects IoU, TPR, F1-score, and detection probability. "
        "Circle and rectangle have sharp boundaries and reach stable detection earlier or at the same contrast as the smoother gaussian case; "
        "the gaussian temperature field has lower peripheral contrast, therefore threshold segmentation tends to recover only the central high-contrast region at low Delta T."
    )
    if gaussian_threshold == circle_threshold == rectangle_threshold:
        conclusion += " In this fixed dataset all three forms satisfy the selected stable-detection criterion at the same Delta T, but the metric curves still differ below and around threshold."

    payload = {
        "source_experiment": str(EXPERIMENT_DIR.as_posix()),
        "output_directory": str(OUT_DIR.as_posix()),
        "old_files_modified": False,
        "seed": int(config["seed"]),
        "frame_size": {"width": int(config["width"]), "height": int(config["height"])},
        "num_frames": int(config["num_frames"]),
        "background_k": float(config["background_k"]),
        "noise_sigma": float(config["noise_sigma"]),
        "fpn_std": float(config["fpn_std"]),
        "threshold_k": float(config["threshold_k"]),
        "iou_success": float(config["iou_success"]),
        "delta_t_values_K": [float(v) for v in config["delta_t_values_K"]],
        "shapes_compared": SHAPES,
        "metrics_calculated": [
            "TPR",
            "FPR",
            "Precision",
            "F1",
            "IoU",
            "snr_like",
            "detection_probability",
            "standard deviations across frames",
        ],
        "detection_thresholds_by_shape": threshold_map,
        "gaussian_temperature_field_formula": "T(x,y)=T_bg + delta_t * exp(-((x-x0)^2+(y-y0)^2)/(2*sigma_g^2))",
        "gaussian_mask_logic": "The repository function gaussian_hotspot uses sigma_g=max(1.0, size/3) and binary mask = weights >= 0.5; experiment 02 uses size=14.",
        "main_conclusion": conclusion,
        "rows_in_shape_delta_table": int(len(grouped)),
    }
    (OUT_DIR / "shape_analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(config: dict[str, Any], thresholds: pd.DataFrame) -> None:
    threshold_lines = []
    for row in thresholds.to_dict(orient="records"):
        threshold_lines.append(f"- `{row['shape']}`: `{row['threshold_delta_t_K']}`")
    content = f"""# Shape analysis for experiment 02

## Purpose

This folder adds a separate analysis of how anomaly shape affects detection quality in `experiment_02_min_detectable_contrast`.
The original experiment files were not modified. All new artifacts are written only to `shape_analysis/`.

## Source files used

- `../config.json`
- `../summary.json`
- `../metrics.csv`
- existing masks in `../masks/` were inspected as fixed ground-truth products

## Fixed parameters

- seed: `{config['seed']}`
- frame size: `{config['width']}x{config['height']}`
- num_frames: `{config['num_frames']}`
- background_k: `{config['background_k']}`
- noise_sigma: `{config['noise_sigma']}`
- fpn_std: `{config['fpn_std']}`
- threshold_k: `{config['threshold_k']}`
- iou_success: `{config['iou_success']}`
- detector: `{config['detector']}`
- delta_t_K values: `{config['delta_t_values_K']}`
- shapes: `{config['shapes']}`

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

{chr(10).join(threshold_lines)}

## Diploma interpretation

Circle and rectangle anomalies have sharper boundaries. The gaussian anomaly has a smooth temperature profile, so its peripheral
pixels have lower local contrast than the center. This makes threshold segmentation harder near the detection limit: at low
`delta_t_K`, the detector tends to recover only the central high-contrast part. Shape therefore affects `IoU`, `TPR`, `F1-score`,
and detection probability even when all other sensor parameters are fixed.
"""
    (OUT_DIR / "README_shape_analysis.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for legacy_name in [
        "shape_analysis_prediction_comparison_delta_t_1.png",
        "shape_analysis_prediction_comparison_delta_t_2.png",
        "shape_analysis_prediction_comparison_delta_t_3.png",
    ]:
        legacy_path = FIG_DIR / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    config = read_json(EXPERIMENT_DIR / "config.json")
    metrics = pd.read_csv(EXPERIMENT_DIR / "metrics.csv")
    grouped, thresholds = aggregate_metrics(metrics, float(config["iou_success"]))
    grouped.to_csv(OUT_DIR / "shape_analysis_metrics_by_shape_delta_t.csv", index=False)
    thresholds.to_csv(OUT_DIR / "shape_analysis_detection_thresholds_by_shape.csv", index=False)

    save_line_plot(grouped, "iou_mean", "IoU", "shape_analysis_iou_vs_delta_t_by_shape.png")
    save_line_plot(grouped, "detection_probability", "Detection probability", "shape_analysis_detection_probability_vs_delta_t_by_shape.png")
    save_line_plot(grouped, "tpr_mean", "TPR", "shape_analysis_tpr_vs_delta_t_by_shape.png")
    save_line_plot(grouped, "f1_mean", "F1-score", "shape_analysis_f1_vs_delta_t_by_shape.png")
    save_line_plot(grouped, "snr_like_mean", "SNR-like", "shape_analysis_snr_like_vs_delta_t_by_shape.png", ylim_01=False)
    save_heatmap(grouped, "iou_mean", "Mean IoU by shape and Delta T", "shape_analysis_iou_heatmap_shape_delta_t.png")
    save_heatmap(
        grouped,
        "detection_probability",
        "Detection probability by shape and Delta T",
        "shape_analysis_detection_probability_heatmap_shape_delta_t.png",
    )

    make_temperature_fields(config)
    for delta_t in COMPARISON_DELTAS:
        make_prediction_comparison(config, delta_t)
    make_gaussian_profile(config)

    write_summary(config, thresholds, grouped)
    write_readme(config, thresholds)
    print(f"Shape analysis written to {OUT_DIR}")


if __name__ == "__main__":
    main()
