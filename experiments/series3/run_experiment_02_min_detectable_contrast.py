#!/usr/bin/env python3
"""Experiment 02: estimate minimum detectable temperature contrast."""

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
    region_snr,
    rng_from_seed,
    save_heatmap,
    save_line_plot,
    save_mask_png,
    save_montage,
    scene_with_anomaly,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate minimum detectable thermal contrast.")
    parser.add_argument("--seed", type=int, default=3202)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--background_k", type=float, default=300.0)
    parser.add_argument("--noise_sigma", type=float, default=2.2)
    parser.add_argument("--fpn_std", type=float, default=0.7)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    parser.add_argument("--iou_success", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(2, "min_detectable_contrast")
    rng = rng_from_seed(args.seed)
    delta_values = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    shapes = ["circle", "rectangle", "gaussian"]
    config = vars(args) | {
        "experiment": "02_min_detectable_contrast",
        "delta_t_values_K": delta_values.tolist(),
        "shapes": shapes,
        "detector": f"global median + {args.threshold_k} * robust_sigma",
    }
    write_json(out_dir / "config.json", config)

    records: list[dict] = []
    examples: list[np.ndarray] = []
    example_titles: list[str] = []
    example_deltas = {delta_values[0], delta_values[len(delta_values) // 2], delta_values[-1]}

    for shape in shapes:
        for delta_t in delta_values:
            scene, truth = scene_with_anomaly(
                args.height,
                args.width,
                background_k=args.background_k,
                delta_t=float(delta_t),
                shape=shape,
                size=14,
            )
            mask_name = f"mask_{shape}_dT_{delta_t:g}K"
            save_mask_png(out_dir / "masks" / f"{mask_name}.png", truth)
            np.savetxt(out_dir / "masks" / f"{mask_name}.csv", truth.astype(int), delimiter=",", fmt="%d")
            for frame_idx in range(args.num_frames):
                frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=args.noise_sigma, fpn_std=args.fpn_std)
                result = detect_global_threshold(frame, k=args.threshold_k, min_area=5)
                metrics = binary_metrics(result.mask, truth)
                snr = region_snr(frame, truth)
                metrics.update(
                    {
                        "shape": shape,
                        "delta_t_K": float(delta_t),
                        "frame_idx": frame_idx,
                        "snr_like": snr,
                        "detected_success": float(metrics["iou"] >= args.iou_success),
                    }
                )
                records.append(metrics)
                if frame_idx == 0 and delta_t in example_deltas and shape == "circle":
                    save_heatmap(out_dir / "images" / f"frame_{shape}_dT_{delta_t:g}K.png", frame, f"{shape}, dT={delta_t:g} K")
                    save_mask_png(out_dir / "images" / f"prediction_{shape}_dT_{delta_t:g}K.png", result.mask)
                    examples.extend([frame, truth.astype(float), result.mask.astype(float)])
                    example_titles.extend([f"Frame dT={delta_t:g}", "Truth", "Detected"])

    df = write_csv(out_dir / "metrics.csv", records)
    grouped = df.groupby("delta_t_K", as_index=False).agg(
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        precision=("precision", "mean"),
        iou=("iou", "mean"),
        detection_probability=("detected_success", "mean"),
        snr_like=("snr_like", "mean"),
    )
    grouped.to_csv(out_dir / "metrics_by_delta_t.csv", index=False)

    save_line_plot(out_dir / "tpr_vs_delta_t.png", grouped["delta_t_K"], grouped["tpr"], "TPR vs delta T", "Delta T, K", "TPR")
    save_line_plot(out_dir / "fpr_vs_delta_t.png", grouped["delta_t_K"], grouped["fpr"], "FPR vs delta T", "Delta T, K", "FPR")
    save_line_plot(out_dir / "iou_vs_delta_t.png", grouped["delta_t_K"], grouped["iou"], "IoU vs delta T", "Delta T, K", "IoU")
    save_line_plot(
        out_dir / "detection_probability_vs_delta_t.png",
        grouped["delta_t_K"],
        grouped["detection_probability"],
        "Detection probability vs delta T",
        "Delta T, K",
        "P(IoU > threshold)",
    )
    save_montage(out_dir / "examples_low_mid_high_delta_t.png", examples, example_titles, cols=3)

    candidates = grouped[grouped["detection_probability"] >= 0.9]
    d_t_min = float(candidates["delta_t_K"].iloc[0]) if len(candidates) else float("nan")
    conclusion = (
        f"Минимальный устойчиво обнаруживаемый контраст при IoU>{args.iou_success:g} и вероятности >=0.9: "
        f"{d_t_min:.3g} K."
    )
    write_readme(
        out_dir / "README.md",
        "Эксперимент 02 - минимальный обнаруживаемый температурный контраст",
        "Оценить, при каком температурном контрасте простая пороговая сегментация стабильно выделяет аномалию.",
        f"Фон {args.background_k:g} K; dT={delta_values.tolist()}; формы: {', '.join(shapes)}; seed: {args.seed}.",
        ["metrics.csv", "metrics_by_delta_t.csv", "tpr_vs_delta_t.png", "fpr_vs_delta_t.png", "iou_vs_delta_t.png", "detection_probability_vs_delta_t.png", "masks/"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=2,
        title="Оценка минимального обнаруживаемого температурного контраста dTmin",
        varied_parameters="Температурный контраст и форма аномалии",
        main_metrics="TPR, FPR, Precision, IoU, detection_probability",
        main_result=conclusion,
        main_plot=str(out_dir / "detection_probability_vs_delta_t.png"),
        conclusion="При малом dT аномалия маскируется шумом; после порогового контраста вероятность обнаружения резко растет.",
    )
    print(f"Experiment 02 completed: {out_dir}")


if __name__ == "__main__":
    main()
