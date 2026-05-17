#!/usr/bin/env python3
"""Additional quantization analysis for experiment 03.

The script writes only to results/experiment_03_noise_influence/quantization_analysis
and does not modify the fixed outputs of experiment 03.
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

sys.path.append(str(ROOT))

from experiments.series3.common import (  # noqa: E402
    ADC_MAX,
    binary_metrics,
    detect_global_threshold,
    rng_from_seed,
    scene_with_anomaly,
    temperature_to_adc,
)


QUANT_STEPS = [1, 2, 4, 8, 16, 32, 64]
SELECTED_STEPS = [1, 4, 16, 64]
DETECTION_IOU_THRESHOLD = 0.3
GAUSSIAN_SIGMA_ADC = 1.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_inputs() -> None:
    required = [
        "config.json",
        "metrics.csv",
        "noise_type_comparison.csv",
        "summary.json",
        "snr_vs_noise_level.png",
        "iou_vs_noise_level.png",
        "tpr_fpr_vs_noise_level.png",
        "example_noise_types.png",
    ]
    missing = [name for name in required if not (EXPERIMENT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment 03 inputs: {missing}")


def quantize_by_step(frame: np.ndarray, quant_step: int) -> np.ndarray:
    step = float(quant_step)
    quantized = np.round(frame / step) * step
    return np.clip(quantized, 0.0, ADC_MAX)


def region_values(frame: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_bool = mask.astype(bool)
    anomaly = frame[mask_bool]
    background = frame[~mask_bool]
    mu_anom = float(np.mean(anomaly))
    mu_bg = float(np.mean(background))
    sigma_bg = float(np.std(background))
    contrast = mu_anom - mu_bg
    snr_like = contrast / sigma_bg if sigma_bg > 1e-12 else float("inf")
    return {
        "mu_anom": mu_anom,
        "mu_bg": mu_bg,
        "sigma_bg": sigma_bg,
        "contrast": contrast,
        "snr_like": snr_like,
    }


def histogram_stats(frame: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_bool = mask.astype(bool)
    background = frame[~mask_bool]
    anomaly = frame[mask_bool]
    return {
        "unique_levels_frame": int(np.unique(frame).size),
        "unique_levels_bg": int(np.unique(background).size),
        "unique_levels_anom": int(np.unique(anomaly).size),
        "min_value": float(np.min(frame)),
        "max_value": float(np.max(frame)),
        "dynamic_range": float(np.max(frame) - np.min(frame)),
        "sigma_bg": float(np.std(background)),
    }


def make_scene_and_reference_frames(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    scene, truth = scene_with_anomaly(
        int(config["height"]),
        int(config["width"]),
        background_k=float(config["background_k"]),
        delta_t=float(config["delta_t"]),
        shape="circle",
        size=16,
        weak_bg=True,
    )
    ideal = temperature_to_adc(scene)
    rng = rng_from_seed(int(config["seed"]))
    reference_frames: list[np.ndarray] = []
    for _ in range(int(config["num_frames"])):
        noisy = ideal + rng.normal(0.0, GAUSSIAN_SIGMA_ADC, size=ideal.shape)
        reference_frames.append(np.clip(noisy, 0.0, ADC_MAX))
    return scene, truth, reference_frames


def calculate_metrics(config: dict[str, Any], truth: np.ndarray, reference_frames: list[np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    for quant_step in QUANT_STEPS:
        num_levels = int(math.floor(ADC_MAX / quant_step) + 1)
        for frame_idx, reference in enumerate(reference_frames):
            frame = quantize_by_step(reference, quant_step)
            detection = detect_global_threshold(frame, k=float(config["threshold_k"]), min_area=5)
            metrics = binary_metrics(detection.mask, truth)
            regions = region_values(frame, truth)
            detected_success = float(metrics["iou"] >= DETECTION_IOU_THRESHOLD)
            records.append(
                {
                    "quant_step": quant_step,
                    "num_levels": num_levels,
                    "frame_idx": frame_idx,
                    "mu_anom": regions["mu_anom"],
                    "mu_bg": regions["mu_bg"],
                    "sigma_bg": regions["sigma_bg"],
                    "snr_like": regions["snr_like"],
                    "contrast": regions["contrast"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"],
                    "tpr": metrics["tpr"],
                    "fpr": metrics["fpr"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "iou": metrics["iou"],
                    "detected_success": detected_success,
                    "threshold": detection.threshold,
                }
            )
            hist_rows.append(
                {
                    "quant_step": quant_step,
                    "num_levels": num_levels,
                    "frame_idx": frame_idx,
                    **histogram_stats(frame, truth),
                }
            )

    metrics_df = pd.DataFrame(records)
    hist_df = pd.DataFrame(hist_rows)
    return metrics_df, hist_df


def aggregate(metrics_df: pd.DataFrame, hist_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = metrics_df.groupby(["quant_step", "num_levels"], as_index=False).agg(
        snr_like_mean=("snr_like", "mean"),
        snr_like_std=("snr_like", "std"),
        sigma_bg_mean=("sigma_bg", "mean"),
        sigma_bg_std=("sigma_bg", "std"),
        contrast_mean=("contrast", "mean"),
        contrast_std=("contrast", "std"),
        tpr_mean=("tpr", "mean"),
        fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"),
        iou_mean=("iou", "mean"),
        detection_probability=("detected_success", "mean"),
    )
    hist_summary = hist_df.groupby(["quant_step", "num_levels"], as_index=False).agg(
        unique_levels_frame_mean=("unique_levels_frame", "mean"),
        unique_levels_frame_std=("unique_levels_frame", "std"),
        unique_levels_bg_mean=("unique_levels_bg", "mean"),
        unique_levels_bg_std=("unique_levels_bg", "std"),
        unique_levels_anom_mean=("unique_levels_anom", "mean"),
        unique_levels_anom_std=("unique_levels_anom", "std"),
        min_value_mean=("min_value", "mean"),
        max_value_mean=("max_value", "mean"),
        dynamic_range_mean=("dynamic_range", "mean"),
        sigma_bg_mean=("sigma_bg", "mean"),
        sigma_bg_std=("sigma_bg", "std"),
    )
    return summary, hist_summary


def save_metric_plot(summary: pd.DataFrame, ycols: list[str], labels: list[str], ylabel: str, title: str, filename: str, ylim_01: bool = False) -> None:
    plt.figure(figsize=(7.6, 4.8))
    for ycol, label in zip(ycols, labels):
        values = summary[ycol].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        plot_values = values.copy()
        if finite.size and np.any(~np.isfinite(values)):
            cap = float(np.max(finite) * 1.15 if np.max(finite) > 0 else 1.0)
            plot_values[~np.isfinite(values)] = cap
        plt.plot(summary["quant_step"], plot_values, marker="o", linewidth=2.0, label=label)
        if finite.size and np.any(~np.isfinite(values)):
            for step, raw_value, plot_value in zip(summary["quant_step"], values, plot_values):
                if not np.isfinite(raw_value):
                    plt.annotate("inf\nsigma_bg=0", (step, plot_value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xscale("log", base=2)
    plt.xticks(QUANT_STEPS, [str(v) for v in QUANT_STEPS])
    plt.xlabel("Шаг квантования, коды ADC")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim_01:
        plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    if len(ycols) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def save_unique_levels_plot(hist_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7.6, 4.8))
    for col, label in [
        ("unique_levels_frame_mean", "Весь кадр"),
        ("unique_levels_bg_mean", "Фон"),
        ("unique_levels_anom_mean", "Аномалия"),
    ]:
        plt.plot(hist_summary["quant_step"], hist_summary[col], marker="o", linewidth=2.0, label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(QUANT_STEPS, [str(v) for v in QUANT_STEPS])
    plt.xlabel("Шаг квантования, коды ADC")
    plt.ylabel("Число уникальных цифровых уровней")
    plt.title("Число уникальных уровней ADC от шага квантования")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quantization_analysis_unique_levels_vs_quant_step.png", dpi=180)
    plt.close()


def save_histograms(reference_frames: list[np.ndarray], truth: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, quant_step in zip(axes_flat, SELECTED_STEPS):
        bg_values: list[np.ndarray] = []
        anom_values: list[np.ndarray] = []
        for reference in reference_frames:
            frame = quantize_by_step(reference, quant_step)
            bg_values.append(frame[~truth])
            anom_values.append(frame[truth])
        bg = np.concatenate(bg_values)
        anom = np.concatenate(anom_values)
        bins = np.arange(min(bg.min(), anom.min()) - quant_step, max(bg.max(), anom.max()) + 2 * quant_step, max(quant_step, 1))
        ax.hist(bg, bins=bins, alpha=0.65, label="Фон", color="#4c78a8")
        ax.hist(anom, bins=bins, alpha=0.65, label="Аномалия", color="#f58518")
        ax.set_title(f"шаг={quant_step}")
        ax.set_xlabel("Код ADC")
        ax.set_ylabel("Число пикселей")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Гистограммы фона и аномалии для разных шагов квантования")
    fig.savefig(OUT_DIR / "quantization_analysis_histograms_by_quant_step.png", dpi=180)
    plt.close(fig)


def overlay_error(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    truth_b = truth.astype(bool)
    pred_b = pred.astype(bool)
    rgb = np.zeros((*truth_b.shape, 3), dtype=float)
    rgb[truth_b & pred_b] = [0.1, 0.75, 0.2]
    rgb[pred_b & ~truth_b] = [0.9, 0.15, 0.15]
    rgb[truth_b & ~pred_b] = [0.15, 0.3, 0.95]
    return rgb


def save_frames_comparison(reference: np.ndarray) -> None:
    steps = [1, 2, 4, 8, 16, 32, 64]
    images = [reference] + [quantize_by_step(reference, step) for step in steps]
    titles = ["Опорный кадр"] + [f"шаг={step}" for step in steps]
    vmin = min(float(np.min(img)) for img in images)
    vmax = max(float(np.max(img)) for img in images)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4), constrained_layout=True)
    for ax, image, title in zip(axes.ravel(), images, titles):
        im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="Код ADC")
    fig.suptitle("Один кадр при разных шагах квантования")
    fig.savefig(OUT_DIR / "quantization_analysis_frames_comparison.png", dpi=180)
    plt.close(fig)


def save_error_maps(reference: np.ndarray) -> None:
    steps = [2, 4, 8, 16, 32, 64]
    errors = [quantize_by_step(reference, step) - reference for step in steps]
    vmax = max(float(np.max(np.abs(err))) for err in errors)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    for ax, error, step in zip(axes.ravel(), errors, steps):
        im = ax.imshow(error, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"шаг={step}")
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="I_quantized - I_reference, ADC")
    fig.suptitle("Карты ошибки квантования")
    fig.savefig(OUT_DIR / "quantization_analysis_error_maps.png", dpi=180)
    plt.close(fig)


def save_masks_comparison(config: dict[str, Any], reference: np.ndarray, truth: np.ndarray) -> None:
    steps = [1, 4, 16, 64]
    fig, axes = plt.subplots(len(steps), 3, figsize=(9.5, 9.8), constrained_layout=True)
    for row, step in enumerate(steps):
        frame = quantize_by_step(reference, step)
        pred = detect_global_threshold(frame, k=float(config["threshold_k"]), min_area=5).mask
        metrics = binary_metrics(pred, truth)
        panels = [
            ("Эталонная маска", truth.astype(float), "gray"),
            ("Найденная маска", pred.astype(float), "gray"),
            ("Наложение: TP зеленый, FP красный, FN синий", overlay_error(truth, pred), None),
        ]
        for col, (title, data, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(data)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"шаг={step}\nIoU={metrics['iou']:.2f}\nTPR={metrics['tpr']:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Маски обнаружения при разных шагах квантования")
    fig.savefig(OUT_DIR / "quantization_analysis_masks_comparison.png", dpi=180)
    plt.close(fig)


def save_profiles(reference: np.ndarray, truth: np.ndarray) -> None:
    row = reference.shape[0] // 2
    xs = np.arange(reference.shape[1])
    plt.figure(figsize=(9.5, 5.2))
    plt.plot(xs, reference[row, :], color="black", linewidth=2.0, label="Опорный кадр")
    for step in [1, 4, 16, 64]:
        plt.step(xs, quantize_by_step(reference, step)[row, :], where="mid", linewidth=1.6, label=f"шаг={step}")
    cols = np.where(truth[row, :])[0]
    if len(cols):
        plt.axvspan(cols.min(), cols.max(), color="#f58518", alpha=0.14, label="Область аномалии")
    plt.xlabel("Пиксель x через центр аномалии")
    plt.ylabel("Код ADC")
    plt.title("Центральный 1D-профиль: ступенчатость из-за квантования")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quantization_analysis_profiles.png", dpi=180)
    plt.close()


def save_background_zoom(reference: np.ndarray) -> None:
    steps = [1, 4, 16, 64]
    crop = (slice(8, 32), slice(8, 40))
    images = [quantize_by_step(reference, step)[crop] for step in steps]
    vmin = min(float(np.min(img)) for img in images)
    vmax = max(float(np.max(img)) for img in images)
    fig, axes = plt.subplots(1, len(steps), figsize=(11.5, 3.2), constrained_layout=True)
    for ax, image, step in zip(axes, images, steps):
        im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"шаг={step}")
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="Код ADC")
    fig.suptitle("Фрагмент фона: квантование делает поле более ступенчатым")
    fig.savefig(OUT_DIR / "quantization_analysis_background_zoom.png", dpi=180)
    plt.close(fig)


def json_value(value: Any) -> Any:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(num):
        return None
    if math.isinf(num):
        return "infinite_or_undefined_due_to_zero_sigma_bg"
    return num


def write_summary_json(config: dict[str, Any], summary: pd.DataFrame, old_quantization: dict[str, float]) -> None:
    max_snr = summary.loc[summary["snr_like_mean"].idxmax()]
    max_iou = summary.loc[summary["iou_mean"].idxmax()]
    min_sigma = summary.loc[summary["sigma_bg_mean"].idxmin()]
    infinite_snr_steps = [int(row["quant_step"]) for _, row in summary.iterrows() if not np.isfinite(float(row["snr_like_mean"]))]
    payload = {
        "source_experiment": str(EXPERIMENT_DIR.as_posix()),
        "output_directory": str(OUT_DIR.as_posix()),
        "old_files_modified": False,
        "seed": int(config["seed"]),
        "frame_size": {"width": int(config["width"]), "height": int(config["height"])},
        "num_frames": int(config["num_frames"]),
        "background_k": float(config["background_k"]),
        "delta_t": float(config["delta_t"]),
        "threshold_k": float(config["threshold_k"]),
        "base_scene": "circle anomaly, size=16, weak_bg=True, same as experiment 03",
        "reference_noise_model": f"ideal ADC frame plus Gaussian noise with sigma={GAUSSIAN_SIGMA_ADC} ADC, fpn_std=0, no defects",
        "quantization_parameterization": {
            "quant_step_adc_codes": QUANT_STEPS,
            "meaning": "I_quantized = round(I_reference / quant_step) * quant_step, clipped to 10-bit ADC range",
            "equivalent_num_levels": {str(step): int(math.floor(ADC_MAX / step) + 1) for step in QUANT_STEPS},
        },
        "metrics": [
            "mu_anom",
            "mu_bg",
            "sigma_bg",
            "contrast",
            "snr_like",
            "TP/FP/TN/FN",
            "TPR",
            "FPR",
            "Precision",
            "F1",
            "IoU",
            "unique digital levels",
        ],
        "max_snr_like": {
            "quant_step": int(max_snr["quant_step"]),
            "snr_like_mean": json_value(max_snr["snr_like_mean"]),
            "sigma_bg_mean": json_value(max_snr["sigma_bg_mean"]),
            "iou_mean": json_value(max_snr["iou_mean"]),
        },
        "infinite_snr_like_steps": infinite_snr_steps,
        "max_iou": {
            "quant_step": int(max_iou["quant_step"]),
            "iou_mean": json_value(max_iou["iou_mean"]),
            "snr_like_mean": json_value(max_iou["snr_like_mean"]),
        },
        "min_sigma_bg": {
            "quant_step": int(min_sigma["quant_step"]),
            "sigma_bg_mean": json_value(min_sigma["sigma_bg_mean"]),
            "snr_like_mean": json_value(min_sigma["snr_like_mean"]),
        },
        "old_experiment_quantization_baseline": {key: json_value(value) for key, value in old_quantization.items()},
        "snr_like_interpretation": (
            "SNR-like=(mu_anom-mu_bg)/sigma_bg can increase under coarse quantization when the background collapses into fewer digital levels "
            "and sigma_bg decreases faster than the anomaly-background contrast. This is a metric artifact, not automatic image improvement."
        ),
        "main_conclusion": (
            "Quantization is not additive random noise. It discretizes the radiometric signal, can reduce background standard deviation, and may inflate SNR-like, "
            "while simultaneously producing stair-step structure, changing masks, and affecting IoU/TPR/F1. Therefore quantization must be evaluated with both "
            "radiometric statistics and segmentation metrics."
        ),
    }
    (OUT_DIR / "quantization_analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_readme(config: dict[str, Any], summary: pd.DataFrame) -> None:
    max_snr = summary.loc[summary["snr_like_mean"].idxmax()]
    max_iou = summary.loc[summary["iou_mean"].idxmax()]
    max_snr_text = "infinite/undefined because sigma_bg=0" if not np.isfinite(float(max_snr["snr_like_mean"])) else f"{max_snr['snr_like_mean']:.3f}"
    content = f"""# Quantization analysis for experiment 03

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

- seed: `{config['seed']}`
- frame size: `{config['width']}x{config['height']}`
- num_frames: `{config['num_frames']}`
- background_k: `{config['background_k']}`
- delta_t: `{config['delta_t']}`
- threshold_k: `{config['threshold_k']}`
- scene: `circle`, `size=16`, `weak_bg=True`
- reference noise for this isolated quantization analysis: Gaussian sigma `{GAUSSIAN_SIGMA_ADC}` ADC, no FPN, no defects

## Quantization parameter

The new analysis uses an explicit ADC-code quantization step:

`I_quantized = round(I_reference / quant_step) * quant_step`.

Checked values: `{QUANT_STEPS}` ADC codes. Equivalent numbers of levels are approximately:
`{{step: floor(1023 / step) + 1}}`.

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

- Maximum SNR-like: `quant_step={int(max_snr['quant_step'])}`, SNR-like `{max_snr_text}`.
- Maximum IoU: `quant_step={int(max_iou['quant_step'])}`, IoU `{max_iou['iou_mean']:.3f}`.

If these steps differ, the analysis demonstrates why SNR-like alone is not sufficient for
judging quantization quality.
"""
    (OUT_DIR / "README_quantization_analysis.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_json(EXPERIMENT_DIR / "config.json")
    old_comparison = pd.read_csv(EXPERIMENT_DIR / "noise_type_comparison.csv")
    old_quant = old_comparison[old_comparison["noise_type"] == "quantization"].iloc[0].to_dict()

    _, truth, reference_frames = make_scene_and_reference_frames(config)
    metrics_df, hist_df = calculate_metrics(config, truth, reference_frames)
    summary, hist_summary = aggregate(metrics_df, hist_df)

    metrics_df.to_csv(OUT_DIR / "quantization_analysis_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "quantization_analysis_summary_by_step.csv", index=False)
    hist_summary.to_csv(OUT_DIR / "quantization_analysis_histogram_stats.csv", index=False)

    save_metric_plot(summary, ["snr_like_mean"], ["SNR-like"], "SNR-like", "SNR-like от шага квантования", "quantization_analysis_snr_like_vs_quant_step.png")
    save_metric_plot(summary, ["sigma_bg_mean"], ["sigma_bg"], "sigma_bg, ADC", "STD фона от шага квантования", "quantization_analysis_sigma_bg_vs_quant_step.png")
    save_metric_plot(summary, ["contrast_mean"], ["mu_anom - mu_bg"], "Контраст, ADC", "Средний контраст аномалия-фон от шага квантования", "quantization_analysis_contrast_vs_quant_step.png")
    save_metric_plot(summary, ["iou_mean"], ["IoU"], "IoU", "IoU от шага квантования", "quantization_analysis_iou_vs_quant_step.png", ylim_01=True)
    save_metric_plot(summary, ["tpr_mean", "fpr_mean"], ["TPR", "FPR"], "Значение метрики", "TPR/FPR от шага квантования", "quantization_analysis_tpr_fpr_vs_quant_step.png", ylim_01=True)
    save_metric_plot(summary, ["precision_mean", "f1_mean"], ["Precision", "F1-score"], "Значение метрики", "Precision/F1 от шага квантования", "quantization_analysis_precision_f1_vs_quant_step.png", ylim_01=True)
    save_unique_levels_plot(hist_summary)
    save_histograms(reference_frames, truth)

    reference0 = reference_frames[0]
    save_frames_comparison(reference0)
    save_error_maps(reference0)
    save_masks_comparison(config, reference0, truth)
    save_profiles(reference0, truth)
    save_background_zoom(reference0)

    write_summary_json(config, summary, old_quant)
    write_readme(config, summary)

    max_snr = summary.loc[summary["snr_like_mean"].idxmax()]
    max_iou = summary.loc[summary["iou_mean"].idxmax()]
    created = sorted(path.name for path in OUT_DIR.glob("quantization_analysis_*")) + ["README_quantization_analysis.md", "run_quantization_analysis.py"]
    print(f"Quantization analysis written to {OUT_DIR}")
    print(f"Created files: {', '.join(created)}")
    max_snr_value = "inf/undefined (sigma_bg=0)" if not np.isfinite(float(max_snr["snr_like_mean"])) else f"{max_snr['snr_like_mean']:.3f}"
    print(f"Max SNR-like: quant_step={int(max_snr['quant_step'])}, value={max_snr_value}")
    print(f"Max IoU: quant_step={int(max_iou['quant_step'])}, value={max_iou['iou_mean']:.3f}")
    print(f"SNR/IoU maxima differ: {int(max_snr['quant_step']) != int(max_iou['quant_step'])}")


if __name__ == "__main__":
    main()
