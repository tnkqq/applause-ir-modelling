#!/usr/bin/env python3
"""Extended temporal dynamics analysis for experiment 06.

All outputs are written to dynamic_extended_analysis only. The fixed files of
results/experiment_06_temporal_dynamics are used as inputs and are not modified.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
FRAMES_DIR = OUT_DIR / "frames"
MASKS_DIR = OUT_DIR / "masks"
TABLES_DIR = OUT_DIR / "tables"

sys.path.append(str(ROOT))

from experiments.series3.common import (  # noqa: E402
    binary_metrics,
    detect_global_threshold,
    generate_adc_frame,
    inertia_sequence,
    local_contrast,
    moving_average_sequence,
    rng_from_seed,
    scene_with_anomaly,
)


SCENARIOS = [
    "static",
    "appearing",
    "moving",
    "appearing_and_growing",
    "appearing_and_fading",
    "moving_fast",
    "moving_diagonal",
    "two_anomalies_static_dynamic",
    "intermittent",
    "background_drift",
    "moving_with_noise_burst",
    "small_moving_target",
]
APPEARING_SCENARIOS = ["appearing", "appearing_and_growing", "intermittent", "moving_with_noise_burst"]
MOVING_SCENARIOS = ["moving", "moving_fast", "moving_diagonal", "two_anomalies_static_dynamic", "moving_with_noise_burst", "small_moving_target"]
DISPLAY_TIMES = {
    "static": [0, 20, 40, 79],
    "appearing": [0, 19, 20, 25, 40, 79],
    "moving": [0, 20, 40, 79],
    "moving_fast": [0, 12, 24, 40],
    "moving_diagonal": [0, 20, 40, 79],
    "appearing_and_growing": [0, 20, 25, 32, 45, 79],
    "intermittent": [0, 20, 29, 35, 45, 65],
    "background_drift": [0, 20, 40, 79],
    "two_anomalies_static_dynamic": [0, 20, 40, 79],
    "moving_with_noise_burst": [0, 30, 36, 42, 60, 79],
}


@dataclass
class ScenarioSequence:
    frames: np.ndarray
    masks: np.ndarray
    centers_x: np.ndarray
    centers_y: np.ndarray
    amplitudes: np.ndarray
    background_means: np.ndarray
    background_stds: np.ndarray


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("background_k", 300.0)
    config.setdefault("alphas", [0.0, 0.3, 0.6, 0.9])
    config.setdefault("windows", [1, 3, 5, 10])
    return config


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def ensure_inputs() -> None:
    required = [
        "config.json",
        "metrics.csv",
        "summary.json",
        "README.md",
        "detection_delay_vs_alpha.png",
        "snr_vs_window_size.png",
        "example_sequence_frames.png",
    ]
    missing = [name for name in required if not (EXPERIMENT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment 06 inputs: {missing}")


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, FRAMES_DIR, MASKS_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def empty_scene(height: int, width: int, background_k: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full((height, width), background_k, dtype=float), np.zeros((height, width), dtype=bool)


def add_anomaly(scene: np.ndarray, mask: np.ndarray, *, delta_t: float, shape: str, size: int, center: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    extra_scene, extra_mask = scene_with_anomaly(
        scene.shape[0],
        scene.shape[1],
        background_k=0.0,
        delta_t=delta_t,
        shape=shape,
        size=size,
        center=center,
    )
    scene = scene + extra_scene
    mask = mask | extra_mask
    return scene, mask


def scenario_scene(config: dict[str, Any], scenario: str, t: int) -> tuple[np.ndarray, np.ndarray, float]:
    height = int(config["height"])
    width = int(config["width"])
    num_frames = int(config["num_frames"])
    event_start = int(config["event_start"])
    delta_t = float(config["delta_t"])
    background_k = float(config["background_k"])
    scene, mask = empty_scene(height, width, background_k)
    amplitude = 0.0

    if scenario == "static":
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "appearing":
        amplitude = delta_t if t >= event_start else 0.0
        if amplitude > 0:
            scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "moving":
        progress = t / max(num_frames - 1, 1)
        center = (height // 2, int(16 + progress * (width - 32)))
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=12, center=center)

    elif scenario == "appearing_and_growing":
        ramp = np.clip((t - event_start + 1) / 20.0, 0.0, 1.0)
        amplitude = delta_t * float(ramp)
        if amplitude > 0:
            scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "appearing_and_fading":
        if t < event_start:
            amplitude = 0.0
        elif t < event_start + 24:
            amplitude = delta_t
        elif t < event_start + 48:
            amplitude = delta_t * float(np.clip(1.0 - (t - (event_start + 24)) / 24.0, 0.0, 1.0))
        else:
            amplitude = 0.0
        if amplitude > 0:
            scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "moving_fast":
        progress = min(t / max((num_frames - 1) * 0.52, 1), 1.0)
        center = (height // 2, int(10 + progress * (width - 20)))
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=12, center=center)

    elif scenario == "moving_diagonal":
        progress = t / max(num_frames - 1, 1)
        center = (int(12 + progress * (height - 24)), int(14 + progress * (width - 28)))
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=12, center=center)

    elif scenario == "two_anomalies_static_dynamic":
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=delta_t * 0.85, shape="circle", size=12, center=(height // 3, width // 3))
        progress = t / max(num_frames - 1, 1)
        center = (int(height * 0.68), int(14 + progress * (width - 28)))
        scene, mask = add_anomaly(scene, mask, delta_t=delta_t, shape="circle", size=10, center=center)

    elif scenario == "intermittent":
        active = (event_start <= t < event_start + 10) or (event_start + 20 <= t < event_start + 32) or (event_start + 42 <= t < event_start + 52)
        amplitude = delta_t if active else 0.0
        if amplitude > 0:
            scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "background_drift":
        drift_k = 1.2 * t / max(num_frames - 1, 1)
        scene = np.full((height, width), background_k + drift_k, dtype=float)
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=14)

    elif scenario == "moving_with_noise_burst":
        progress = t / max(num_frames - 1, 1)
        center = (height // 2, int(16 + progress * (width - 32)))
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=12, center=center)

    elif scenario == "small_moving_target":
        progress = t / max(num_frames - 1, 1)
        center = (height // 2, int(16 + progress * (width - 32)))
        amplitude = delta_t
        scene, mask = add_anomaly(scene, mask, delta_t=amplitude, shape="circle", size=6, center=center)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return scene, mask, amplitude


def center_of_mask(mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return float("nan"), float("nan")
    yy, xx = np.nonzero(mask)
    return float(np.mean(xx)), float(np.mean(yy))


def generate_sequence(config: dict[str, Any], scenario: str, seed_offset: int) -> ScenarioSequence:
    rng = rng_from_seed(int(config["seed"]) + 1000 + seed_offset)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    amplitudes: list[float] = []
    background_means: list[float] = []
    background_stds: list[float] = []
    for t in range(int(config["num_frames"])):
        scene, mask, amplitude = scenario_scene(config, scenario, t)
        gaussian_sigma = 3.0
        if scenario == "moving_with_noise_burst" and 34 <= t <= 44:
            gaussian_sigma = 8.0
        frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=gaussian_sigma, fpn_std=0.9)
        cx, cy = center_of_mask(mask)
        background = frame[~mask] if np.any(mask) else frame.ravel()
        frames.append(frame.astype(np.float32))
        masks.append(mask.astype(bool))
        centers_x.append(cx)
        centers_y.append(cy)
        amplitudes.append(float(amplitude))
        background_means.append(float(np.mean(background)))
        background_stds.append(float(np.std(background)))
    return ScenarioSequence(
        frames=np.stack(frames).astype(np.float32),
        masks=np.stack(masks).astype(bool),
        centers_x=np.asarray(centers_x, dtype=float),
        centers_y=np.asarray(centers_y, dtype=float),
        amplitudes=np.asarray(amplitudes, dtype=float),
        background_means=np.asarray(background_means, dtype=float),
        background_stds=np.asarray(background_stds, dtype=float),
    )


def target_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not np.any(truth):
        pred = pred.astype(bool)
        fp = int(np.sum(pred))
        tn = int(pred.size - fp)
        return {
            "tp": 0,
            "fp": fp,
            "tn": tn,
            "fn": 0,
            "tpr": 0.0,
            "recall": 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "iou": 0.0,
        }
    return binary_metrics(pred, truth)


def snr_like(frame: np.ndarray, truth: np.ndarray) -> float:
    if not np.any(truth):
        return 0.0
    anomaly = frame[truth]
    background = frame[~truth]
    sigma = float(np.std(background))
    return float((np.mean(anomaly) - np.mean(background)) / max(sigma, 1.0))


def center_error(pred: np.ndarray, truth: np.ndarray) -> float:
    if not np.any(pred) or not np.any(truth):
        return float("nan")
    pred_x, pred_y = center_of_mask(pred)
    truth_x, truth_y = center_of_mask(truth)
    return float(math.hypot(pred_x - truth_x, pred_y - truth_y))


def save_sequences(scenario: str, sequence: ScenarioSequence, inertial_by_alpha: dict[float, np.ndarray], processed_by_combo: dict[tuple[float, int], np.ndarray], masks_by_combo: dict[tuple[float, int], np.ndarray]) -> None:
    np.savez_compressed(FRAMES_DIR / f"{scenario}_raw_sequence.npz", frames=sequence.frames)
    np.savez_compressed(
        MASKS_DIR / f"{scenario}_truth_masks.npz",
        masks=sequence.masks.astype(np.uint8),
        center_x=sequence.centers_x,
        center_y=sequence.centers_y,
        amplitudes=sequence.amplitudes,
    )
    np.savez_compressed(
        FRAMES_DIR / f"{scenario}_inertia_sequences.npz",
        **{f"alpha_{token(alpha)}": arr.astype(np.float32) for alpha, arr in inertial_by_alpha.items()},
    )
    np.savez_compressed(
        FRAMES_DIR / f"{scenario}_processed_sequences.npz",
        **{f"alpha_{token(alpha)}_window_{window}": arr.astype(np.float32) for (alpha, window), arr in processed_by_combo.items()},
    )
    np.savez_compressed(
        MASKS_DIR / f"{scenario}_predicted_masks.npz",
        **{f"alpha_{token(alpha)}_window_{window}": arr.astype(np.uint8) for (alpha, window), arr in masks_by_combo.items()},
    )


def process_all(config: dict[str, Any]) -> tuple[dict[str, ScenarioSequence], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alphas = [float(v) for v in config["alphas"]]
    windows = [int(v) for v in config["windows"]]
    per_frame_rows: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    scenario_sequences: dict[str, ScenarioSequence] = {}

    for scenario_idx, scenario in enumerate(SCENARIOS):
        sequence = generate_sequence(config, scenario, scenario_idx)
        scenario_sequences[scenario] = sequence
        raw_peak = max(abs(local_contrast(sequence.frames[t], sequence.masks[t])) for t in range(sequence.frames.shape[0]))
        inertial_by_alpha: dict[float, np.ndarray] = {}
        processed_by_combo: dict[tuple[float, int], np.ndarray] = {}
        masks_by_combo: dict[tuple[float, int], np.ndarray] = {}

        for alpha in alphas:
            inertial = inertia_sequence(sequence.frames, alpha).astype(np.float32)
            inertial_by_alpha[alpha] = inertial
            for window in windows:
                processed = moving_average_sequence(inertial, window).astype(np.float32)
                pred_masks: list[np.ndarray] = []
                contrasts = []
                for t in range(processed.shape[0]):
                    truth = sequence.masks[t]
                    pred = detect_global_threshold(processed[t], k=float(config["threshold_k"]), min_area=5).mask
                    pred_masks.append(pred)
                    metrics = target_metrics(pred, truth)
                    present = bool(np.any(truth))
                    detected = int(present and metrics["iou"] > 0.3)
                    c_err = center_error(pred, truth)
                    contrast = local_contrast(processed[t], truth) if present else 0.0
                    contrasts.append(contrast)
                    bg = processed[t][~truth] if present else processed[t].ravel()
                    per_frame_rows.append(
                        {
                            "scenario": scenario,
                            "alpha": alpha,
                            "window": window,
                            "frame_idx": t,
                            "snr_like": snr_like(processed[t], truth),
                            "tp": metrics["tp"],
                            "fp": metrics["fp"],
                            "tn": metrics["tn"],
                            "fn": metrics["fn"],
                            "tpr": metrics["tpr"],
                            "fpr": metrics["fpr"],
                            "precision": metrics["precision"],
                            "f1": metrics["f1"],
                            "iou": metrics["iou"],
                            "detected": detected,
                            "target_present": int(present),
                            "target_center_x": sequence.centers_x[t],
                            "target_center_y": sequence.centers_y[t],
                            "peak_amplitude": contrast,
                            "raw_peak_reference": raw_peak,
                            "background_mean": float(np.mean(bg)),
                            "background_std": float(np.std(bg)),
                            "center_error_px": c_err,
                        }
                    )
                    if scenario in MOVING_SCENARIOS:
                        tracking_rows.append(
                            {
                                "scenario": scenario,
                                "alpha": alpha,
                                "window": window,
                                "frame_idx": t,
                                "target_present": int(present),
                                "detected": detected,
                                "center_error_px": c_err,
                            }
                        )
                processed_by_combo[(alpha, window)] = processed
                masks_by_combo[(alpha, window)] = np.stack(pred_masks).astype(bool)

        save_sequences(scenario, sequence, inertial_by_alpha, processed_by_combo, masks_by_combo)

    per_frame = pd.DataFrame(per_frame_rows)
    tracking = pd.DataFrame(tracking_rows)
    return scenario_sequences, per_frame, tracking, aggregate_metrics(per_frame, tracking, config)


def first_detection_delay(group: pd.DataFrame, event_start: int) -> float:
    present = group[(group["frame_idx"] >= event_start) & (group["target_present"] == 1)]
    detected = present[present["detected"] == 1]
    if detected.empty:
        return float("nan")
    return float(detected["frame_idx"].iloc[0] - event_start)


def aggregate_metrics(per_frame: pd.DataFrame, tracking: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_start = int(config["event_start"])
    present = per_frame[per_frame["target_present"] == 1].copy()
    grouped_rows: list[dict[str, Any]] = []
    for (scenario, alpha, window), group in per_frame.groupby(["scenario", "alpha", "window"]):
        present_group = group[group["target_present"] == 1]
        use = present_group if not present_group.empty else group
        raw_peak = max(float(group["raw_peak_reference"].iloc[0]) if "raw_peak_reference" in group else 0.0, 1e-9)
        grouped_rows.append(
            {
                "scenario": scenario,
                "alpha": alpha,
                "window": window,
                "snr_like": float(use["snr_like"].mean()),
                "tpr": float(use["tpr"].mean()),
                "fpr": float(use["fpr"].mean()),
                "precision": float(use["precision"].mean()),
                "f1": float(use["f1"].mean()),
                "iou": float(use["iou"].mean()),
                "detection_delay_frames": first_detection_delay(group, event_start),
                "peak_amplitude_ratio": float(np.nanmax(np.abs(use["peak_amplitude"])) / raw_peak),
                "detection_probability": float(use["detected"].mean()),
                "mean_center_error_px": float(use["center_error_px"].mean(skipna=True)),
            }
        )
    summary_combo = pd.DataFrame(grouped_rows)

    scenario_rows: list[dict[str, Any]] = []
    for scenario, group in summary_combo.groupby("scenario"):
        best_iou = group.loc[group["iou"].idxmax()]
        valid_delay = group.dropna(subset=["detection_delay_frames"])
        best_delay = valid_delay.loc[valid_delay["detection_delay_frames"].idxmin()] if not valid_delay.empty else best_iou
        norm_iou = group["iou"] / max(float(group["iou"].max()), 1e-9)
        delay_filled = group["detection_delay_frames"].fillna(group["detection_delay_frames"].max(skipna=True) if group["detection_delay_frames"].notna().any() else 20.0)
        norm_delay = 1.0 - delay_filled / max(float(delay_filled.max()), 1.0)
        center = group["mean_center_error_px"].fillna(group["mean_center_error_px"].max(skipna=True) if group["mean_center_error_px"].notna().any() else 0.0)
        norm_center = 1.0 - center / max(float(center.max()), 1.0)
        score = 0.55 * norm_iou + 0.25 * norm_delay + 0.20 * norm_center
        compromise = group.loc[score.idxmax()]
        scenario_rows.append(
            {
                "scenario": scenario,
                "best_iou_alpha": best_iou["alpha"],
                "best_iou_window": int(best_iou["window"]),
                "best_iou": best_iou["iou"],
                "min_delay_alpha": best_delay["alpha"],
                "min_delay_window": int(best_delay["window"]),
                "min_delay_frames": best_delay["detection_delay_frames"],
                "compromise_alpha": compromise["alpha"],
                "compromise_window": int(compromise["window"]),
                "compromise_iou": compromise["iou"],
                "compromise_delay": compromise["detection_delay_frames"],
                "compromise_snr_like": compromise["snr_like"],
            }
        )
    summary_scenario = pd.DataFrame(scenario_rows)

    delay_rows = summary_combo[summary_combo["scenario"].isin(APPEARING_SCENARIOS)][
        ["scenario", "alpha", "window", "detection_delay_frames", "detection_probability", "iou", "snr_like"]
    ].copy()

    tracking_summary = (
        tracking[tracking["target_present"] == 1]
        .groupby(["scenario", "alpha", "window"], as_index=False)
        .agg(
            center_error_px_mean=("center_error_px", "mean"),
            center_error_px_std=("center_error_px", "std"),
            tracking_detection_probability=("detected", "mean"),
        )
    )
    return summary_combo, summary_scenario, delay_rows.merge(tracking_summary, on=["scenario", "alpha", "window"], how="left")


def write_tables(per_frame: pd.DataFrame, summary_combo: pd.DataFrame, summary_scenario: pd.DataFrame, delay_tracking: pd.DataFrame) -> pd.DataFrame:
    per_frame.to_csv(TABLES_DIR / "dynamic_extended_metrics_per_frame.csv", index=False)
    summary_combo.to_csv(TABLES_DIR / "dynamic_extended_summary_by_scenario_alpha_window.csv", index=False)
    summary_scenario.to_csv(TABLES_DIR / "dynamic_extended_summary_by_scenario.csv", index=False)
    delay_tracking[delay_tracking["scenario"].isin(APPEARING_SCENARIOS)][
        ["scenario", "alpha", "window", "detection_delay_frames", "detection_probability", "iou", "snr_like"]
    ].to_csv(TABLES_DIR / "dynamic_extended_delay_table.csv", index=False)
    tracking_cols = ["scenario", "alpha", "window", "center_error_px_mean", "center_error_px_std", "tracking_detection_probability"]
    delay_tracking[delay_tracking["scenario"].isin(MOVING_SCENARIOS)][tracking_cols].to_csv(TABLES_DIR / "dynamic_extended_tracking_error.csv", index=False)
    return delay_tracking


def fixed_scale_montage(images: list[np.ndarray], titles: list[str], path: Path, suptitle: str, *, cols: int | None = None, cmap: str = "inferno") -> None:
    if cols is None:
        cols = len(images)
    rows = int(math.ceil(len(images) / cols))
    vmin = float(min(np.min(image) for image in images))
    vmax = float(max(np.max(image) for image in images))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.7), squeeze=False, constrained_layout=True)
    last = None
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(images):
            last = ax.imshow(images[idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(titles[idx], fontsize=9)
        ax.set_axis_off()
    if last is not None:
        fig.colorbar(last, ax=axes, label="ADC code")
    fig.suptitle(f"{suptitle} (fixed color scale)", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mask_montage(images: list[np.ndarray], titles: list[str], path: Path, suptitle: str, *, cols: int | None = None) -> None:
    if cols is None:
        cols = len(images)
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4), squeeze=False, constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(images):
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[idx], fontsize=9)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lines(summary: pd.DataFrame, x: str, y: str, group: str, path: Path, title: str, xlabel: str, ylabel: str, *, ylim_01: bool = False, scenarios: list[str] | None = None) -> None:
    data = summary if scenarios is None else summary[summary["scenario"].isin(scenarios)]
    plt.figure(figsize=(8.4, 5.2))
    for label, sub in data.groupby(group):
        curve = sub.groupby(x, as_index=False)[y].mean().sort_values(x)
        plt.plot(curve[x], curve[y], marker="o", linewidth=2.0, label=str(label))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim_01:
        plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_required_graphs(summary: pd.DataFrame, delay_tracking: pd.DataFrame) -> None:
    delay_alpha = summary[summary["scenario"].isin(APPEARING_SCENARIOS)].groupby(["scenario", "alpha"], as_index=False)["detection_delay_frames"].mean()
    plot_lines(delay_alpha, "alpha", "detection_delay_frames", "scenario", FIG_DIR / "dynamic_extended_detection_delay_vs_alpha.png", "Detection delay vs alpha", "Alpha", "Delay, frames")

    snr_window = summary.groupby(["scenario", "window"], as_index=False)["snr_like"].mean()
    plot_lines(snr_window, "window", "snr_like", "scenario", FIG_DIR / "dynamic_extended_snr_vs_window.png", "SNR-like vs averaging window", "Window, frames", "SNR-like")

    iou_window = summary[summary["scenario"].isin(MOVING_SCENARIOS)].groupby(["scenario", "window"], as_index=False)["iou"].mean()
    plot_lines(iou_window, "window", "iou", "scenario", FIG_DIR / "dynamic_extended_iou_vs_window_moving_scenarios.png", "IoU vs averaging window for moving scenarios", "Window, frames", "IoU", ylim_01=True)

    iou_alpha = summary.groupby(["scenario", "alpha"], as_index=False)["iou"].mean()
    plot_lines(iou_alpha, "alpha", "iou", "scenario", FIG_DIR / "dynamic_extended_iou_vs_alpha.png", "IoU vs alpha", "Alpha", "IoU", ylim_01=True)

    peak_alpha = summary.groupby(["scenario", "alpha"], as_index=False)["peak_amplitude_ratio"].mean()
    plot_lines(peak_alpha, "alpha", "peak_amplitude_ratio", "scenario", FIG_DIR / "dynamic_extended_peak_amplitude_ratio_vs_alpha.png", "Peak amplitude ratio vs alpha", "Alpha", "Peak amplitude ratio")

    center = delay_tracking[delay_tracking["scenario"].isin(MOVING_SCENARIOS)].copy()
    center["alpha_window"] = center["alpha"].map(lambda x: f"a={x:g}") + ", w=" + center["window"].astype(str)
    pivot = center.pivot_table(index="scenario", columns="alpha_window", values="center_error_px_mean", aggfunc="mean")
    save_heatmap(pivot, FIG_DIR / "dynamic_extended_center_error_vs_alpha_window.png", "Center error by scenario and alpha/window", "Center error, px")

    tpr_fpr = summary.groupby("scenario", as_index=False).agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"))
    x = np.arange(len(tpr_fpr))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 0.18, tpr_fpr["tpr"], width=0.36, label="TPR")
    ax.bar(x + 0.18, tpr_fpr["fpr"], width=0.36, label="FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(tpr_fpr["scenario"], rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("TPR/FPR by scenario")
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dynamic_extended_tpr_fpr_by_scenario.png", dpi=180)
    plt.close(fig)

    heat_data = summary.copy()
    heat_data["alpha_window"] = heat_data["alpha"].map(lambda x: f"a={x:g}") + ", w=" + heat_data["window"].astype(str)
    save_heatmap(heat_data.pivot_table(index="scenario", columns="alpha_window", values="iou", aggfunc="mean"), FIG_DIR / "dynamic_extended_heatmap_iou_scenario_alpha_window.png", "IoU heatmap by scenario and alpha/window", "IoU", vmin=0, vmax=1)

    delay_heat = heat_data[heat_data["scenario"].isin(APPEARING_SCENARIOS)].pivot_table(index="scenario", columns="alpha_window", values="detection_delay_frames", aggfunc="mean")
    save_heatmap(delay_heat, FIG_DIR / "dynamic_extended_heatmap_delay_scenario_alpha_window.png", "Delay heatmap by scenario and alpha/window", "Delay, frames")

    trade = summary[summary["scenario"].isin(APPEARING_SCENARIOS)].dropna(subset=["detection_delay_frames"])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for scenario, sub in trade.groupby("scenario"):
        ax.scatter(sub["snr_like"], sub["detection_delay_frames"], label=scenario, s=45)
    ax.set_xlabel("SNR-like")
    ax.set_ylabel("Detection delay, frames")
    ax.set_title("Trade-off: SNR-like vs detection delay")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dynamic_extended_tradeoff_snr_delay.png", dpi=180)
    plt.close(fig)


def save_heatmap(pivot: pd.DataFrame, path: Path, title: str, cbar_label: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
    pivot = pivot.sort_index()
    fig_w = max(8.0, 0.55 * len(pivot.columns))
    fig_h = max(3.8, 0.35 * len(pivot.index) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    values = pivot.to_numpy(dtype=float)
    if vmax is None and np.isfinite(values).any():
        vmax = float(np.nanmax(values))
    if vmin is None:
        vmin = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            text = "-" if np.isnan(value) else f"{value:.2f}"
            ax.text(x, y, text, ha="center", va="center", color="white" if np.isfinite(value) and value > (vmin + (vmax - vmin) * 0.55 if vmax != vmin else 0.5) else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_processed(scenario: str, alpha: float, window: int) -> np.ndarray:
    key = f"alpha_{token(alpha)}_window_{window}"
    with np.load(FRAMES_DIR / f"{scenario}_processed_sequences.npz") as data:
        return data[key]


def load_pred_masks(scenario: str, alpha: float, window: int) -> np.ndarray:
    key = f"alpha_{token(alpha)}_window_{window}"
    with np.load(MASKS_DIR / f"{scenario}_predicted_masks.npz") as data:
        return data[key].astype(bool)


def plot_sequence_figures(sequences: dict[str, ScenarioSequence]) -> None:
    mapping = [
        ("static", "dynamic_extended_static_frames_fixed_scale.png", "Static scenario"),
        ("appearing", "dynamic_extended_appearing_frames_fixed_scale.png", "Appearing scenario"),
        ("moving", "dynamic_extended_moving_frames_fixed_scale.png", "Moving scenario"),
        ("moving_fast", "dynamic_extended_moving_fast_frames_fixed_scale.png", "Fast moving scenario"),
        ("moving_diagonal", "dynamic_extended_moving_diagonal_frames_fixed_scale.png", "Diagonal moving scenario"),
        ("appearing_and_growing", "dynamic_extended_appearing_growing_frames_fixed_scale.png", "Appearing and growing scenario"),
        ("intermittent", "dynamic_extended_intermittent_frames_fixed_scale.png", "Intermittent scenario"),
        ("background_drift", "dynamic_extended_background_drift_frames_fixed_scale.png", "Background drift scenario"),
        ("two_anomalies_static_dynamic", "dynamic_extended_two_anomalies_frames_fixed_scale.png", "Two anomalies scenario"),
        ("moving_with_noise_burst", "dynamic_extended_noise_burst_frames_fixed_scale.png", "Moving with noise burst scenario"),
    ]
    for scenario, filename, title in mapping:
        times = DISPLAY_TIMES[scenario]
        images = [sequences[scenario].frames[t] for t in times]
        titles = [f"t={t}" for t in times]
        fixed_scale_montage(images, titles, FIG_DIR / filename, title)


def plot_alpha_window_comparisons(sequences: dict[str, ScenarioSequence], config: dict[str, Any]) -> None:
    times = [19, 20, 25, 40]
    images = []
    titles = []
    for alpha in [0.0, 0.3, 0.6, 0.9]:
        processed = load_processed("appearing", alpha, 1)
        for t in times:
            images.append(processed[t])
            titles.append(f"alpha={alpha:g}, t={t}")
    fixed_scale_montage(images, titles, FIG_DIR / "dynamic_extended_alpha_comparison_appearing.png", "Appearing object: alpha comparison", cols=len(times))

    times = [0, 20, 40, 79]
    images = []
    titles = []
    for window in [1, 3, 5, 10]:
        processed = load_processed("moving", 0.6, window)
        for t in times:
            images.append(processed[t])
            titles.append(f"w={window}, t={t}")
    fixed_scale_montage(images, titles, FIG_DIR / "dynamic_extended_window_comparison_moving.png", "Moving object: window comparison, alpha=0.6", cols=len(times))

    mask_images = []
    mask_titles = []
    for alpha in [0.0, 0.3, 0.6, 0.9]:
        masks = load_pred_masks("appearing", alpha, 1)
        for t in [20, 25, 40]:
            mask_images.append(masks[t].astype(float))
            mask_titles.append(f"alpha={alpha:g}, t={t}")
    mask_montage(mask_images, mask_titles, FIG_DIR / "dynamic_extended_masks_alpha_comparison_appearing.png", "Predicted masks: appearing alpha comparison", cols=3)

    mask_images = []
    mask_titles = []
    for window in [1, 3, 5, 10]:
        masks = load_pred_masks("moving", 0.6, window)
        for t in [20, 40, 79]:
            mask_images.append(masks[t].astype(float))
            mask_titles.append(f"w={window}, t={t}")
    mask_montage(mask_images, mask_titles, FIG_DIR / "dynamic_extended_masks_window_comparison_moving.png", "Predicted masks: moving window comparison, alpha=0.6", cols=3)

    truth = sequences["moving"].masks
    pred = load_pred_masks("moving", 0.6, 10)
    overlays = [overlay(truth[t], pred[t]) for t in [20, 40, 60, 79]]
    titles = [f"t={t}" for t in [20, 40, 60, 79]]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), constrained_layout=True)
    for ax, image, title in zip(axes, overlays, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Moving object overlay, alpha=0.6, window=10: TP green, FP red, FN blue")
    fig.savefig(FIG_DIR / "dynamic_extended_overlay_moving_window10.png", dpi=180)
    plt.close(fig)


def overlay(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*truth.shape, 3), dtype=float)
    truth_b = truth.astype(bool)
    pred_b = pred.astype(bool)
    rgb[truth_b & pred_b] = [0.1, 0.75, 0.2]
    rgb[pred_b & ~truth_b] = [0.9, 0.15, 0.15]
    rgb[truth_b & ~pred_b] = [0.15, 0.3, 0.95]
    return rgb


def write_summary_json(summary: pd.DataFrame, scenario_summary: pd.DataFrame, delay_tracking: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    hardest = scenario_summary.loc[scenario_summary["best_iou"].idxmin()]
    delay_scope = summary[summary["scenario"].isin(APPEARING_SCENARIOS)].dropna(subset=["detection_delay_frames"])
    max_delay = delay_scope.sort_values("detection_delay_frames", ascending=False).iloc[0]
    moving = summary[summary["scenario"].isin(MOVING_SCENARIOS)]
    best_moving = moving.loc[moving["iou"].idxmax()]
    payload = {
        "source_experiment": str(EXPERIMENT_DIR.as_posix()),
        "output_directory": str(OUT_DIR.as_posix()),
        "old_files_modified": False,
        "seed": int(config["seed"]),
        "frame_size": {"height": int(config["height"]), "width": int(config["width"])},
        "num_frames": int(config["num_frames"]),
        "delta_t": float(config["delta_t"]),
        "event_start": int(config["event_start"]),
        "threshold_k": float(config["threshold_k"]),
        "alphas": [float(v) for v in config["alphas"]],
        "windows": [int(v) for v in config["windows"]],
        "scenarios": SCENARIOS,
        "inertia_model": "I_alpha[t] = (1 - alpha) * I[t] + alpha * I_alpha[t-1]",
        "averaging_model": "I_avg[t] = mean(I_alpha[max(0,t-W+1):t+1])",
        "fixed_color_scale_policy": "Every montage computes vmin/vmax once across all frames in the compared set and uses those values for every imshow.",
        "hardest_scenario_by_best_iou": {
            "scenario": hardest["scenario"],
            "best_iou": float(hardest["best_iou"]),
            "best_alpha": float(hardest["best_iou_alpha"]),
            "best_window": int(hardest["best_iou_window"]),
        },
        "largest_delay": {
            "scenario": max_delay["scenario"],
            "alpha": float(max_delay["alpha"]),
            "window": int(max_delay["window"]),
            "detection_delay_frames": None if pd.isna(max_delay["detection_delay_frames"]) else float(max_delay["detection_delay_frames"]),
        },
        "best_moving_iou": {
            "scenario": best_moving["scenario"],
            "alpha": float(best_moving["alpha"]),
            "window": int(best_moving["window"]),
            "iou": float(best_moving["iou"]),
        },
        "recommended_figures_for_diploma": [
            "figures/dynamic_extended_appearing_frames_fixed_scale.png",
            "figures/dynamic_extended_moving_frames_fixed_scale.png",
            "figures/dynamic_extended_alpha_comparison_appearing.png",
            "figures/dynamic_extended_window_comparison_moving.png",
            "figures/dynamic_extended_overlay_moving_window10.png",
            "figures/dynamic_extended_tradeoff_snr_delay.png",
            "figures/dynamic_extended_heatmap_iou_scenario_alpha_window.png",
        ],
    }
    (OUT_DIR / "dynamic_extended_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_readme(payload: dict[str, Any], summary: pd.DataFrame) -> None:
    content = f"""# Dynamic extended analysis for experiment 06

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

{chr(10).join(f'- `{scenario}`' for scenario in SCENARIOS)}

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

{chr(10).join(f'- `{figure}`' for figure in payload['recommended_figures_for_diploma'])}

## Engineering interpretation

For a static object, frame averaging is usually useful because it reduces random noise and
raises SNR-like. For an appearing object, a large averaging window and high inertia increase
detection delay. For a moving object, large windows and high inertia degrade localization,
reduce IoU, and increase center error. Background drift tests whether the global threshold
remains stable when the mean level slowly changes. Intermittent and noise-burst scenarios
show that maximum SNR-like is not enough for dynamic tasks: delay and mask coincidence are
equally important.

The hardest scenario by best IoU in this run was `{payload['hardest_scenario_by_best_iou']['scenario']}`
with best IoU `{payload['hardest_scenario_by_best_iou']['best_iou']:.3f}`. The largest measured
delay occurred for `{payload['largest_delay']['scenario']}` at alpha `{payload['largest_delay']['alpha']}`
and window `{payload['largest_delay']['window']}`.

## Files intentionally not modified

The script does not modify the original experiment 06 `config.json`, `summary.json`,
`metrics.csv`, PNG figures, README, masks, or reports. All new results are isolated in
`dynamic_extended_analysis/`.
"""
    (OUT_DIR / "README_dynamic_extended_analysis.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_inputs()
    ensure_dirs()
    config = normalize_config(read_json(EXPERIMENT_DIR / "config.json"))
    sequences, per_frame, tracking, aggregates = process_all(config)
    summary_combo, summary_scenario, delay_tracking = aggregates
    write_tables(per_frame, summary_combo, summary_scenario, delay_tracking)
    plot_required_graphs(summary_combo, delay_tracking)
    plot_sequence_figures(sequences)
    plot_alpha_window_comparisons(sequences, config)
    payload = write_summary_json(summary_combo, summary_scenario, delay_tracking, config)
    write_readme(payload, summary_combo)

    created_png = sorted(path.relative_to(OUT_DIR).as_posix() for path in FIG_DIR.glob("*.png"))
    created_csv = sorted(path.relative_to(OUT_DIR).as_posix() for path in TABLES_DIR.glob("*.csv"))
    print(f"Dynamic extended analysis written to {OUT_DIR}")
    print(f"Scenarios processed: {len(SCENARIOS)}")
    print(f"CSV files: {', '.join(created_csv)}")
    print(f"PNG files: {len(created_png)} generated")
    print("Old experiment 06 files were not modified; outputs are isolated in dynamic_extended_analysis/.")
    print(f"Hardest scenario: {payload['hardest_scenario_by_best_iou']['scenario']} (best IoU={payload['hardest_scenario_by_best_iou']['best_iou']:.3f})")
    print(
        "Largest delay: "
        f"{payload['largest_delay']['scenario']} alpha={payload['largest_delay']['alpha']} "
        f"window={payload['largest_delay']['window']} delay={payload['largest_delay']['detection_delay_frames']}"
    )
    print(
        "Best moving IoU: "
        f"{payload['best_moving_iou']['scenario']} alpha={payload['best_moving_iou']['alpha']} "
        f"window={payload['best_moving_iou']['window']} IoU={payload['best_moving_iou']['iou']:.3f}"
    )
    print("Recommended figures: " + ", ".join(payload["recommended_figures_for_diploma"]))


if __name__ == "__main__":
    main()
