#!/usr/bin/env python3
"""Experiment 05: spatial resolution, anomaly size and fill factor."""

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
    save_heatmap_matrix,
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
    parser = argparse.ArgumentParser(description="Evaluate detectability vs anomaly size and fill factor.")
    parser.add_argument("--seed", type=int, default=3505)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--delta_t", type=float, default=6.0)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(5, "spatial_resolution")
    rng = rng_from_seed(args.seed)
    sizes = [1, 2, 4, 8, 16, 32]
    fills = [0.1, 0.25, 0.5, 0.75, 1.0]
    write_json(out_dir / "config.json", vars(args) | {"experiment": "05_spatial_resolution", "sizes_px": sizes, "fill_factors": fills})

    records: list[dict] = []
    example_images: list[np.ndarray] = []
    example_titles: list[str] = []
    for fill in fills:
        for size in sizes:
            for frame_idx in range(args.num_frames):
                scene, truth = scene_with_anomaly(
                    args.height,
                    args.width,
                    background_k=300.0,
                    delta_t=args.delta_t,
                    shape="rectangle",
                    size=size,
                    fill_factor=fill,
                )
                if frame_idx == 0:
                    mask_name = f"mask_size_{size}_fill_{fill:g}"
                    save_mask_png(out_dir / "masks" / f"{mask_name}.png", truth)
                    np.savetxt(out_dir / "masks" / f"{mask_name}.csv", truth.astype(int), delimiter=",", fmt="%d")
                frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=2.5, fpn_std=0.8)
                result = detect_global_threshold(frame, k=args.threshold_k, min_area=max(1, min(4, size * size)))
                metrics = binary_metrics(result.mask, truth)
                metrics.update(
                    {
                        "size_px": size,
                        "fill_factor": fill,
                        "frame_idx": frame_idx,
                        "snr_like": region_snr(frame, truth),
                        "anomaly_area_px": int(np.sum(truth)),
                    }
                )
                records.append(metrics)
                if frame_idx == 0 and fill in {0.25, 1.0} and size in {1, 8, 32}:
                    save_heatmap(out_dir / "images" / f"frame_size_{size}_fill_{fill:g}.png", frame, f"size={size}, fill={fill:g}")
                    example_images.extend([frame, truth.astype(float), result.mask.astype(float)])
                    example_titles.extend([f"{size}px fill {fill:g}", "truth", "detected"])

    df = write_csv(out_dir / "metrics.csv", records)
    grouped = df.groupby(["fill_factor", "size_px"], as_index=False).agg(
        snr_like=("snr_like", "mean"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        precision=("precision", "mean"),
        iou=("iou", "mean"),
    )
    grouped.to_csv(out_dir / "metrics_grid.csv", index=False)

    tpr_matrix = np.zeros((len(fills), len(sizes)))
    iou_matrix = np.zeros_like(tpr_matrix)
    for i, fill in enumerate(fills):
        for j, size in enumerate(sizes):
            row = grouped[(grouped.fill_factor == fill) & (grouped.size_px == size)].iloc[0]
            tpr_matrix[i, j] = row["tpr"]
            iou_matrix[i, j] = row["iou"]
    save_heatmap_matrix(out_dir / "tpr_heatmap.png", tpr_matrix, [str(s) for s in sizes], [str(f) for f in fills], "TPR(size, fill factor)", "TPR")
    save_heatmap_matrix(out_dir / "iou_heatmap.png", iou_matrix, [str(s) for s in sizes], [str(f) for f in fills], "IoU(size, fill factor)", "IoU")

    min_rows = []
    for fill in fills:
        subset = grouped[(grouped.fill_factor == fill) & (grouped.tpr > 0.9) & (grouped.fpr < 0.01)].sort_values("size_px")
        min_size = int(subset["size_px"].iloc[0]) if len(subset) else np.nan
        min_rows.append({"fill_factor": fill, "min_detectable_size_px": min_size})
    min_df = write_csv(out_dir / "min_detectable_size.csv", min_rows)
    save_line_plot(out_dir / "min_detectable_size.png", min_df["fill_factor"], min_df["min_detectable_size_px"], "Minimum detectable size vs fill factor", "Fill factor", "Minimum size, px")
    save_montage(out_dir / "size_examples.png", example_images, example_titles, cols=3)

    valid = min_df.dropna()
    best_text = "не достигнут в заданной сетке" if valid.empty else f"{int(valid['min_detectable_size_px'].min())} px"
    conclusion = f"Минимальный обнаруживаемый размер при TPR>0.9 и FPR<0.01: {best_text}; малый fill_factor заметно ухудшает обнаружение."
    write_readme(
        out_dir / "README.md",
        "Эксперимент 05 - пространственное разрешение и размер аномалии",
        "Показать влияние площади аномалии и коэффициента заполнения пикселя на обнаружимость.",
        f"Размеры: {sizes}; fill_factor: {fills}; dT={args.delta_t:g} K; seed: {args.seed}.",
        ["metrics.csv", "metrics_grid.csv", "tpr_heatmap.png", "iou_heatmap.png", "min_detectable_size.png", "size_examples.png"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=5,
        title="Влияние пространственного разрешения и размера аномалии на обнаружение",
        varied_parameters="Размер аномалии и fill_factor",
        main_metrics="SNR-like, TPR, FPR, IoU, min_detectable_size",
        main_result=conclusion,
        main_plot=str(out_dir / "tpr_heatmap.png"),
        conclusion="Малые аномалии и неполное заполнение пикселя уменьшают локальный контраст и ухудшают обнаружение.",
    )
    print(f"Experiment 05 completed: {out_dir}")


if __name__ == "__main__":
    main()
