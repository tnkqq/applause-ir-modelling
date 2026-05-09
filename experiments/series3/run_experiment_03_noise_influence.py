#!/usr/bin/env python3
"""Experiment 03: influence of noise type and level on anomaly detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    ADC_BITS,
    binary_metrics,
    detect_global_threshold,
    experiment_dir,
    generate_adc_frame,
    region_snr,
    rng_from_seed,
    save_heatmap,
    save_mask_png,
    save_line_plot,
    save_montage,
    save_multi_line_plot,
    scene_with_anomaly,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare noise components in synthetic IR frames.")
    parser.add_argument("--seed", type=int, default=3303)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--delta_t", type=float, default=4.0)
    parser.add_argument("--background_k", type=float, default=300.0)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    return parser.parse_args()


def noise_params(noise_type: str, level: float) -> dict:
    if noise_type == "gaussian":
        return {"gaussian_sigma": level, "fpn_std": 0.0, "quant_bits": ADC_BITS, "defect_rate": 0.0}
    if noise_type == "fpn":
        return {"gaussian_sigma": 1.0, "fpn_std": level, "quant_bits": ADC_BITS, "defect_rate": 0.0}
    if noise_type == "quantization":
        bits = max(5, ADC_BITS - int(level))
        return {"gaussian_sigma": 1.0, "fpn_std": 0.0, "quant_bits": bits, "defect_rate": 0.0}
    if noise_type == "defects":
        return {"gaussian_sigma": 1.0, "fpn_std": 0.0, "quant_bits": ADC_BITS, "defect_rate": level * 0.0025}
    if noise_type == "combined":
        bits = max(5, ADC_BITS - int(level))
        return {"gaussian_sigma": 1.0 + level, "fpn_std": 0.75 * level, "quant_bits": bits, "defect_rate": level * 0.0015}
    raise ValueError(noise_type)


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(3, "noise_influence")
    rng = rng_from_seed(args.seed)
    noise_types = ["gaussian", "fpn", "quantization", "defects", "combined"]
    levels = [0, 1, 2, 4, 6, 8]
    scene, truth = scene_with_anomaly(
        args.height,
        args.width,
        background_k=args.background_k,
        delta_t=args.delta_t,
        shape="circle",
        size=16,
        weak_bg=True,
    )
    save_mask_png(out_dir / "masks" / "base_anomaly_mask.png", truth)
    np.savetxt(out_dir / "masks" / "base_anomaly_mask.csv", truth.astype(int), delimiter=",", fmt="%d")
    write_json(
        out_dir / "config.json",
        vars(args) | {"experiment": "03_noise_influence", "noise_types": noise_types, "noise_levels": levels},
    )

    records: list[dict] = []
    example_images: list[np.ndarray] = []
    example_titles: list[str] = []

    for noise_type in noise_types:
        for level in levels:
            params = noise_params(noise_type, float(level))
            for frame_idx in range(args.num_frames):
                frame, _ = generate_adc_frame(scene, rng, **params)
                result = detect_global_threshold(frame, k=args.threshold_k, min_area=5)
                metrics = binary_metrics(result.mask, truth)
                metrics.update(
                    {
                        "noise_type": noise_type,
                        "noise_level": float(level),
                        "frame_idx": frame_idx,
                        "snr_like": region_snr(frame, truth),
                        "gaussian_sigma": params["gaussian_sigma"],
                        "fpn_std": params["fpn_std"],
                        "quant_bits": params["quant_bits"],
                        "defect_rate": params["defect_rate"],
                    }
                )
                records.append(metrics)
                if frame_idx == 0 and level == levels[-1]:
                    save_heatmap(out_dir / "images" / f"example_{noise_type}.png", frame, f"{noise_type}, level {level}")
                    example_images.extend([frame, result.mask.astype(float)])
                    example_titles.extend([noise_type, f"{noise_type} mask"])

    df = write_csv(out_dir / "metrics.csv", records)
    grouped = df.groupby(["noise_type", "noise_level"], as_index=False).agg(
        snr_like=("snr_like", "mean"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        precision=("precision", "mean"),
        iou=("iou", "mean"),
    )
    grouped.to_csv(out_dir / "metrics_by_noise_level.csv", index=False)
    comparison = df.groupby("noise_type", as_index=False).agg(
        snr_like_mean=("snr_like", "mean"),
        tpr_mean=("tpr", "mean"),
        fpr_mean=("fpr", "mean"),
        iou_mean=("iou", "mean"),
    )
    comparison.to_csv(out_dir / "noise_type_comparison.csv", index=False)

    save_multi_line_plot(
        out_dir / "snr_vs_noise_level.png",
        {nt: (grouped[grouped.noise_type == nt]["noise_level"], grouped[grouped.noise_type == nt]["snr_like"]) for nt in noise_types},
        "SNR vs noise level",
        "Noise level",
        "SNR-like",
    )
    save_multi_line_plot(
        out_dir / "tpr_fpr_vs_noise_level.png",
        {
            f"{nt} TPR": (grouped[grouped.noise_type == nt]["noise_level"], grouped[grouped.noise_type == nt]["tpr"])
            for nt in noise_types
        }
        | {
            f"{nt} FPR": (grouped[grouped.noise_type == nt]["noise_level"], grouped[grouped.noise_type == nt]["fpr"])
            for nt in noise_types
        },
        "TPR/FPR vs noise level",
        "Noise level",
        "Metric value",
    )
    save_multi_line_plot(
        out_dir / "iou_vs_noise_level.png",
        {nt: (grouped[grouped.noise_type == nt]["noise_level"], grouped[grouped.noise_type == nt]["iou"]) for nt in noise_types},
        "IoU vs noise level",
        "Noise level",
        "IoU",
    )
    save_montage(out_dir / "example_noise_types.png", example_images, example_titles, cols=2)

    worst = comparison.sort_values("iou_mean").iloc[0]
    conclusion = f"Наиболее сильное среднее ухудшение IoU дал шум `{worst['noise_type']}`: средний IoU={worst['iou_mean']:.3f}."
    write_readme(
        out_dir / "README.md",
        "Эксперимент 03 - влияние типа и уровня шума",
        "Оценить, как разные шумовые составляющие ИК-датчика ухудшают обнаружение температурной аномалии.",
        f"Типы шума: {', '.join(noise_types)}; уровни: {levels}; seed: {args.seed}. "
        "Gaussian - временный шум электроники; FPN - фиксированная неоднородность матрицы; "
        "quantization - дискретизация АЦП; defects - hot/cold пиксели; combined - совместное действие факторов.",
        ["metrics.csv", "noise_type_comparison.csv", "snr_vs_noise_level.png", "tpr_fpr_vs_noise_level.png", "iou_vs_noise_level.png", "example_noise_types.png"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=3,
        title="Влияние уровня и типа шума на обнаружение температурной аномалии",
        varied_parameters="Тип шума и интенсивность шума",
        main_metrics="SNR-like, TPR, FPR, Precision, IoU",
        main_result=conclusion,
        main_plot=str(out_dir / "iou_vs_noise_level.png"),
        conclusion="Рост шума снижает SNR и IoU; разные физические компоненты шума деградируют обнаружение по-разному.",
    )
    print(f"Experiment 03 completed: {out_dir}")


if __name__ == "__main__":
    main()
