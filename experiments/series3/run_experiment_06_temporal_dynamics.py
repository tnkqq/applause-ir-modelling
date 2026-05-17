#!/usr/bin/env python3
"""Experiment 06: temporal inertia and frame averaging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    binary_metrics,
    detect_global_threshold,
    experiment_dir,
    generate_adc_frame,
    inertia_sequence,
    local_contrast,
    moving_average_sequence,
    region_snr,
    rng_from_seed,
    save_line_plot,
    save_mask_png,
    save_montage,
    save_multi_line_plot,
    scene_with_anomaly,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal inertia and frame averaging.")
    parser.add_argument("--seed", type=int, default=3606)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--num_frames", type=int, default=80)
    parser.add_argument("--delta_t", type=float, default=4.5)
    parser.add_argument("--event_start", type=int, default=20)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    return parser.parse_args()


def build_sequence(args: argparse.Namespace, rng: np.random.Generator, scenario: str) -> tuple[np.ndarray, list[np.ndarray]]:
    frames = []
    masks = []
    for t in range(args.num_frames):
        if scenario == "static":
            scene, mask = scene_with_anomaly(args.height, args.width, delta_t=args.delta_t, shape="circle", size=14)
        elif scenario == "moving":
            progress = t / max(args.num_frames - 1, 1)
            center = (args.height // 2, int(16 + progress * (args.width - 32)))
            scene, mask = scene_with_anomaly(args.height, args.width, delta_t=args.delta_t, shape="circle", size=12, center=center)
        elif scenario == "appearing":
            ramp = np.clip((t - args.event_start) / 12.0, 0.0, 1.0)
            scene, mask = scene_with_anomaly(args.height, args.width, delta_t=args.delta_t * ramp, shape="circle", size=14)
            if ramp <= 0:
                mask = np.zeros_like(mask)
        else:
            raise ValueError(scenario)
        frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=3.0, fpn_std=0.9)
        frames.append(frame)
        masks.append(mask)
    return np.stack(frames), masks


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(6, "temporal_dynamics")
    rng = rng_from_seed(args.seed)
    alphas = [0.0, 0.3, 0.6, 0.9]
    windows = [1, 3, 5, 10]
    scenarios = ["static", "appearing", "moving"]
    write_json(out_dir / "config.json", vars(args) | {"experiment": "06_temporal_dynamics", "alphas": alphas, "windows": windows, "scenarios": scenarios})

    raw_sequences = {scenario: build_sequence(args, rng, scenario) for scenario in scenarios}
    for scenario, (_, masks) in raw_sequences.items():
        for t in [0, args.event_start, args.num_frames // 2, args.num_frames - 1]:
            save_mask_png(out_dir / "masks" / f"mask_{scenario}_t{t}.png", masks[t])
            np.savetxt(out_dir / "masks" / f"mask_{scenario}_t{t}.csv", masks[t].astype(int), delimiter=",", fmt="%d")
    records: list[dict] = []
    amplitude_records: list[dict] = []
    example_images: list[np.ndarray] = []
    example_titles: list[str] = []

    for scenario, (sequence, masks) in raw_sequences.items():
        raw_peak = max(abs(local_contrast(sequence[t], masks[t])) for t in range(args.num_frames))
        for alpha in alphas:
            inertial = inertia_sequence(sequence, alpha)
            for window in windows:
                processed = moving_average_sequence(inertial, window)
                per_frame_metrics = []
                detection_frame = None
                contrasts = []
                for t in range(args.num_frames):
                    truth = masks[t]
                    result = detect_global_threshold(processed[t], k=args.threshold_k, min_area=5)
                    metrics = binary_metrics(result.mask, truth)
                    metrics["snr_like"] = region_snr(processed[t], truth) if np.any(truth) else 0.0
                    per_frame_metrics.append(metrics)
                    contrasts.append(local_contrast(processed[t], truth) if np.any(truth) else 0.0)
                    if detection_frame is None and t >= args.event_start and metrics["iou"] >= 0.3 and np.any(truth):
                        detection_frame = t
                    if scenario == "appearing" and window == 1:
                        amplitude_records.append({"time": t, "alpha": alpha, "contrast_adc": contrasts[-1]})
                delay = np.nan if detection_frame is None else max(0, detection_frame - args.event_start)
                peak = max(abs(v) for v in contrasts)
                records.append(
                    {
                        "scenario": scenario,
                        "alpha": alpha,
                        "window": window,
                        "snr_like": float(np.mean([m["snr_like"] for m in per_frame_metrics])),
                        "tpr": float(np.mean([m["tpr"] for m in per_frame_metrics])),
                        "fpr": float(np.mean([m["fpr"] for m in per_frame_metrics])),
                        "precision": float(np.mean([m["precision"] for m in per_frame_metrics])),
                        "iou": float(np.mean([m["iou"] for m in per_frame_metrics])),
                        "detection_delay_frames": delay,
                        "peak_amplitude_ratio": float(peak / max(raw_peak, 1e-9)),
                    }
                )
                if scenario == "moving" and alpha == 0.6 and window == 5:
                    for t in [0, args.event_start, args.num_frames // 2, args.num_frames - 1]:
                        example_images.append(processed[t])
                        example_titles.append(f"t={t}")

    df = write_csv(out_dir / "metrics.csv", records)
    amp_df = write_csv(out_dir / "amplitude_over_time.csv", amplitude_records)
    snr_by_window = df.groupby("window", as_index=False)["snr_like"].mean()
    delay_by_alpha = df.groupby("alpha", as_index=False)["detection_delay_frames"].mean()
    tpr_fpr_by_alpha = df.groupby("alpha", as_index=False).agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"))
    save_line_plot(out_dir / "snr_vs_window_size.png", snr_by_window["window"], snr_by_window["snr_like"], "SNR от окна усреднения", "Окно, кадров", "SNR-like")
    save_line_plot(out_dir / "detection_delay_vs_alpha.png", delay_by_alpha["alpha"], delay_by_alpha["detection_delay_frames"], "Задержка обнаружения от alpha", "Alpha", "Задержка, кадров")
    save_multi_line_plot(
        out_dir / "tpr_fpr_vs_alpha.png",
        {"TPR": (tpr_fpr_by_alpha["alpha"], tpr_fpr_by_alpha["tpr"]), "FPR": (tpr_fpr_by_alpha["alpha"], tpr_fpr_by_alpha["fpr"])},
        "TPR/FPR от alpha",
        "Alpha",
        "Значение метрики",
    )
    amp_series = {
        f"alpha={alpha}": (
            amp_df[amp_df.alpha == alpha]["time"],
            amp_df[amp_df.alpha == alpha]["contrast_adc"],
        )
        for alpha in alphas
    }
    save_multi_line_plot(out_dir / "anomaly_amplitude_over_time.png", amp_series, "Амплитуда аномалии во времени", "Кадр", "Контраст, ADC")
    save_montage(out_dir / "example_sequence_frames.png", example_images, example_titles, cols=4)

    best_snr = float(snr_by_window["snr_like"].max())
    max_delay = float(delay_by_alpha["detection_delay_frames"].max())
    conclusion = f"Кадровое усреднение повышало средний SNR до {best_snr:.3f}, но высокая инерционность увеличивала задержку до {max_delay:.2f} кадров."
    write_readme(
        out_dir / "README.md",
        "Эксперимент 06 - временная инерционность и усреднение кадров",
        "Показать компромисс между ростом SNR при временной фильтрации и задержкой обнаружения динамической аномалии.",
        f"Сценарии: {scenarios}; alpha={alphas}; windows={windows}; длина={args.num_frames}; seed: {args.seed}.",
        ["metrics.csv", "amplitude_over_time.csv", "snr_vs_window_size.png", "detection_delay_vs_alpha.png", "tpr_fpr_vs_alpha.png", "anomaly_amplitude_over_time.png", "example_sequence_frames.png"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=6,
        title="Временная инерционность ИК-датчика и усреднение последовательности кадров",
        varied_parameters="alpha фильтра первого порядка и размер окна усреднения",
        main_metrics="SNR-like, TPR, FPR, detection_delay, peak_amplitude_ratio",
        main_result=conclusion,
        main_plot=str(out_dir / "detection_delay_vs_alpha.png"),
        conclusion="Усреднение полезно для статических объектов, но инерционность ухудшает скорость реакции на появляющиеся и движущиеся аномалии.",
    )
    print(f"Experiment 06 completed: {out_dir}")


if __name__ == "__main__":
    main()
