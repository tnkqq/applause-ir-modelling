#!/usr/bin/env python3
"""Experiment 04: evaluate preprocessing filters before anomaly detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    apply_filter,
    binary_metrics,
    detect_global_threshold,
    experiment_dir,
    generate_adc_frame,
    local_contrast,
    region_snr,
    rng_from_seed,
    save_bar_plot,
    save_heatmap,
    save_mask_png,
    save_montage,
    scene_with_anomaly,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)

PLOT_TITLE_FONTSIZE = 24
PLOT_LABEL_FONTSIZE = 21
PLOT_TICK_FONTSIZE = 17
PLOT_SMALL_FONTSIZE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple IR image filters before anomaly detection.")
    parser.add_argument("--seed", type=int, default=3404)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--delta_t", type=float, default=3.5)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    return parser.parse_args()


def apply_large_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_LABEL_FONTSIZE,
            "axes.titlesize": PLOT_TITLE_FONTSIZE,
            "axes.labelsize": PLOT_LABEL_FONTSIZE,
            "xtick.labelsize": PLOT_TICK_FONTSIZE,
            "ytick.labelsize": PLOT_TICK_FONTSIZE,
            "legend.fontsize": PLOT_SMALL_FONTSIZE,
            "legend.title_fontsize": PLOT_SMALL_FONTSIZE,
            "figure.titlesize": PLOT_TITLE_FONTSIZE + 2,
        }
    )


def wrap_title(title: str, width: int = 38) -> str:
    return fill(title, width=width)


def save_heatmap(path: Path, array: np.ndarray, title: str, cbar_label: str = "Код ADC") -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.3), constrained_layout=True)
    im = ax.imshow(array, cmap="inferno")
    ax.set_title(wrap_title(title), fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)
    ax.bar(labels, values)
    ax.set_title(wrap_title(title), fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PLOT_SMALL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(28)
        tick.set_ha("right")
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_montage(path: Path, images: list[np.ndarray], titles: list[str], *, cmap: str = "inferno", cols: int = 3) -> None:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.1), squeeze=False, constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(images):
            ax.imshow(images[idx], cmap=cmap)
            ax.set_title(wrap_title(titles[idx], width=24), fontsize=PLOT_TITLE_FONTSIZE - 5)
        ax.set_axis_off()
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def filter_grid() -> list[tuple[str, str, dict]]:
    return [
        ("none", "none", {}),
        ("gaussian", "gaussian_k3_s1", {"ksize": 3, "sigma": 1.0}),
        ("gaussian", "gaussian_k5_s1p4", {"ksize": 5, "sigma": 1.4}),
        ("median", "median_k3", {"ksize": 3}),
        ("median", "median_k5", {"ksize": 5}),
        ("bilateral", "bilateral_d5", {"diameter": 5, "sigma_color": 18, "sigma_space": 5}),
        ("bilateral", "bilateral_d9", {"diameter": 9, "sigma_color": 28, "sigma_space": 7}),
        ("nlm", "nlm_h5", {"h": 5}),
        ("nlm", "nlm_h10", {"h": 10}),
    ]


def filter_title(label: str) -> str:
    titles = {
        "none": "Без фильтра",
        "gaussian_k3_s1": "Гауссов фильтр k=3, sigma=1.0",
        "gaussian_k5_s1p4": "Гауссов фильтр k=5, sigma=1.4",
        "median_k3": "Медианный фильтр k=3",
        "median_k5": "Медианный фильтр k=5",
        "bilateral_d5": "Билатеральный фильтр d=5",
        "bilateral_d9": "Билатеральный фильтр d=9",
        "nlm_h5": "Non-local means h=5",
        "nlm_h10": "Non-local means h=10",
    }
    return titles.get(label, label)


def main() -> None:
    args = parse_args()
    apply_large_plot_style()
    out_dir = experiment_dir(4, "filtering")
    rng = rng_from_seed(args.seed)
    scene, truth = scene_with_anomaly(
        args.height,
        args.width,
        background_k=300.0,
        delta_t=args.delta_t,
        shape="circle",
        size=14,
        weak_bg=True,
    )
    save_mask_png(out_dir / "masks" / "base_anomaly_mask.png", truth)
    np.savetxt(out_dir / "masks" / "base_anomaly_mask.csv", truth.astype(int), delimiter=",", fmt="%d")
    filters = filter_grid()
    write_json(out_dir / "config.json", vars(args) | {"experiment": "04_filtering", "filters": [{"name": n, "label": l, "params": p} for n, l, p in filters]})

    base_frames = []
    for _ in range(args.num_frames):
        frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=4.0, fpn_std=2.0, quant_bits=9, defect_rate=0.002)
        base_frames.append(frame)

    records: list[dict] = []
    example_images: list[np.ndarray] = []
    example_titles: list[str] = []
    for filter_name, label, params in filters:
        for idx, frame in enumerate(base_frames):
            before_snr = region_snr(frame, truth)
            before_contrast = local_contrast(frame, truth)
            filtered = apply_filter(frame, filter_name, params)
            result = detect_global_threshold(filtered, k=args.threshold_k, min_area=5)
            metrics = binary_metrics(result.mask, truth)
            after_snr = region_snr(filtered, truth)
            after_contrast = local_contrast(filtered, truth)
            metrics.update(
                {
                    "filter": filter_name,
                    "label": label,
                    "frame_idx": idx,
                    "snr_before": before_snr,
                    "snr_after": after_snr,
                    "snr_improvement": after_snr - before_snr,
                    "contrast_before": before_contrast,
                    "contrast_after": after_contrast,
                    "contrast_change": after_contrast - before_contrast,
                    "params": str(params),
                }
            )
            records.append(metrics)
            if idx == 0 and label in {"none", "gaussian_k5_s1p4", "median_k3", "bilateral_d5", "nlm_h10"}:
                save_heatmap(out_dir / "images" / f"filtered_{label}.png", filtered, filter_title(label))
                example_images.extend([filtered, result.mask.astype(float)])
                example_titles.extend([filter_title(label), f"{filter_title(label)}: маска"])

    df = write_csv(out_dir / "metrics.csv", records)
    comparison = df.groupby(["filter", "label"], as_index=False).agg(
        snr_before=("snr_before", "mean"),
        snr_after=("snr_after", "mean"),
        snr_improvement=("snr_improvement", "mean"),
        tpr=("tpr", "mean"),
        fpr=("fpr", "mean"),
        precision=("precision", "mean"),
        iou=("iou", "mean"),
        contrast_change=("contrast_change", "mean"),
    )
    comparison.to_csv(out_dir / "filter_comparison.csv", index=False)
    best_rows = comparison.sort_values("iou", ascending=False)
    labels = best_rows["label"].tolist()
    plot_labels = [filter_title(label) for label in labels]
    save_bar_plot(out_dir / "snr_improvement_by_filter.png", plot_labels, best_rows["snr_improvement"].tolist(), "Прирост SNR после фильтрации", "Delta SNR")
    save_bar_plot(out_dir / "iou_by_filter.png", plot_labels, best_rows["iou"].tolist(), "IoU для разных фильтров", "IoU")
    save_bar_plot(out_dir / "tpr_fpr_by_filter.png", plot_labels, (best_rows["tpr"] - best_rows["fpr"]).tolist(), "TPR-FPR для разных фильтров", "TPR - FPR")
    save_montage(out_dir / "filter_examples.png", example_images, example_titles, cols=2)

    best = best_rows.iloc[0]
    conclusion = f"Наиболее устойчивый вариант по среднему IoU: `{best['label']}` с IoU={best['iou']:.3f} и SNR improvement={best['snr_improvement']:.3f}."
    write_readme(
        out_dir / "README.md",
        "Эксперимент 04 - эффективность фильтрации ИК-изображений",
        "Проверить, повышает ли предварительная фильтрация качество обнаружения аномалий на зашумленных кадрах.",
        f"Единый набор из {args.num_frames} кадров; фильтры: none, Gaussian, median, bilateral, non-local means; seed: {args.seed}.",
        ["metrics.csv", "filter_comparison.csv", "snr_improvement_by_filter.png", "iou_by_filter.png", "tpr_fpr_by_filter.png", "filter_examples.png"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=4,
        title="Оценка эффективности фильтрации ИК-изображений перед обнаружением аномалий",
        varied_parameters="Тип фильтра и параметры фильтра",
        main_metrics="SNR before/after, TPR, FPR, Precision, IoU, contrast_change",
        main_result=conclusion,
        main_plot=str(out_dir / "iou_by_filter.png"),
        conclusion="Фильтрация снижает шум, но слишком сильное сглаживание может уменьшать локальный контраст аномалии.",
    )
    print(f"Experiment 04 completed: {out_dir}")


if __name__ == "__main__":
    main()
