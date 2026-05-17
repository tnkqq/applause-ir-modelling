#!/usr/bin/env python3
"""Experiment 07: compare interpretable anomaly detectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    DetectionResult,
    auc_from_points,
    binary_metrics,
    detect_dog,
    detect_global_threshold,
    detect_local_threshold,
    detect_otsu,
    experiment_dir,
    generate_adc_frame,
    morphology_postprocess,
    region_snr,
    rng_from_seed,
    save_bar_plot,
    save_mask_png,
    save_montage,
    save_multi_line_plot,
    scene_with_anomaly,
    time_call,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple anomaly detectors.")
    parser.add_argument("--seed", type=int, default=3707)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no_anomaly_frames", type=int, default=10)
    return parser.parse_args()


def detector_grid() -> list[tuple[str, str, dict]]:
    grid = []
    for k in [2.0, 2.5, 3.0, 3.5, 4.0]:
        grid.append(("global_threshold", f"k={k:g}", {"k": k}))
    for k in [1.5, 2.0, 2.5, 3.0, 3.5]:
        grid.append(("adaptive_local", f"k={k:g}", {"k": k, "sigma": 5.0}))
    for offset in [-30, -15, 0, 15, 30]:
        grid.append(("otsu", f"offset={offset}", {"offset": offset}))
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        grid.append(("dog", f"k={k:g}", {"k": k}))
    for kernel in [1, 3, 5, 7]:
        grid.append(("morph_global", f"kernel={kernel}", {"k": 2.5, "kernel": kernel}))
    return grid


def algorithm_title(algorithm: str) -> str:
    titles = {
        "global_threshold": "Глобальный порог",
        "adaptive_local": "Локальный адаптивный порог",
        "otsu": "Метод Otsu",
        "dog": "DoG",
        "morph_global": "Глобальный порог + морфология",
    }
    return titles.get(algorithm, algorithm)


def run_detector(frame: np.ndarray, algorithm: str, params: dict):
    if algorithm == "global_threshold":
        return detect_global_threshold(frame, k=params["k"], min_area=5)
    if algorithm == "adaptive_local":
        return detect_local_threshold(frame, k=params["k"], sigma=params["sigma"], min_area=5)
    if algorithm == "otsu":
        return detect_otsu(frame, offset=params["offset"], min_area=5)
    if algorithm == "dog":
        return detect_dog(frame, k=params["k"], min_area=5)
    if algorithm == "morph_global":
        base = detect_global_threshold(frame, k=params["k"], min_area=1)
        mask = morphology_postprocess(base.mask, kernel_size=params["kernel"], min_area=5)
        return DetectionResult(mask=mask, score=base.score, threshold=base.threshold)
    raise ValueError(algorithm)


def make_dataset(args: argparse.Namespace, rng: np.random.Generator, out_dir: Path) -> list[dict]:
    dataset = []
    deltas = [2.0, 4.0, 6.0]
    sizes = [8, 16, 24]
    noise_levels = [1.5, 3.0]
    shapes = ["circle", "rectangle"]
    idx = 0
    for delta_t in deltas:
        for size in sizes:
            for noise in noise_levels:
                for shape in shapes:
                    for _ in range(args.repeats):
                        center = (
                            int(rng.integers(size + 2, args.height - size - 2)),
                            int(rng.integers(size + 2, args.width - size - 2)),
                        )
                        scene, truth = scene_with_anomaly(
                            args.height,
                            args.width,
                            background_k=300.0,
                            delta_t=delta_t,
                            shape=shape,
                            size=size,
                            center=center,
                            weak_bg=True,
                        )
                        frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=noise, fpn_std=noise * 0.5, quant_bits=9)
                        mask_path = out_dir / "masks" / f"mask_{idx:04d}.png"
                        save_mask_png(mask_path, truth)
                        dataset.append({"id": idx, "frame": frame, "truth": truth, "delta_t": delta_t, "size": size, "noise": noise, "shape": shape, "has_anomaly": True})
                        idx += 1
    for _ in range(args.no_anomaly_frames):
        scene = np.full((args.height, args.width), 300.0, dtype=float)
        frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=3.0, fpn_std=1.5, quant_bits=9)
        truth = np.zeros_like(frame, dtype=bool)
        save_mask_png(out_dir / "masks" / f"mask_{idx:04d}.png", truth)
        dataset.append({"id": idx, "frame": frame, "truth": truth, "delta_t": 0.0, "size": 0, "noise": 3.0, "shape": "none", "has_anomaly": False})
        idx += 1
    return dataset


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(7, "detector_comparison")
    rng = rng_from_seed(args.seed)
    grid = detector_grid()
    dataset = make_dataset(args, rng, out_dir)
    write_json(
        out_dir / "config.json",
        vars(args) | {"experiment": "07_detector_comparison", "dataset_size": len(dataset), "detector_grid": [{"algorithm": a, "label": l, "params": p} for a, l, p in grid]},
    )

    records: list[dict] = []
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    for algorithm, label, params in grid:
        total_time = 0.0
        for item in dataset:
            result, elapsed = time_call(run_detector, item["frame"], algorithm, params)
            total_time += elapsed
            metrics = binary_metrics(result.mask, item["truth"])
            metrics.update(
                {
                    "algorithm": algorithm,
                    "parameter": label,
                    "frame_id": item["id"],
                    "has_anomaly": item["has_anomaly"],
                    "delta_t": item["delta_t"],
                    "size": item["size"],
                    "noise": item["noise"],
                    "shape": item["shape"],
                    "snr_like": region_snr(item["frame"], item["truth"]) if item["has_anomaly"] else 0.0,
                    "runtime_s": elapsed,
                }
            )
            records.append(metrics)
            predictions[(algorithm, label, item["id"])] = result.mask
        # Дублируем суммарное время через отдельные записи не нужно: среднее считается из runtime_s.

    df = write_csv(out_dir / "metrics.csv", records)
    by_param = df.groupby(["algorithm", "parameter"], as_index=False).agg(
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        fpr=("fpr", "mean"),
        f1=("f1", "mean"),
        iou=("iou", "mean"),
        runtime_s=("runtime_s", "mean"),
    )
    summary_rows = []
    strengths = {
        "global_threshold": "простая интерпретация и минимальное время",
        "adaptive_local": "устойчивее к медленному фону",
        "otsu": "не требует явного k для каждого кадра",
        "dog": "выделяет локальные горячие пятна",
        "morph_global": "уменьшает мелкие ложные области",
    }
    weaknesses = {
        "global_threshold": "чувствителен к неоднородному фону",
        "adaptive_local": "зависит от масштаба локального окна",
        "otsu": "может ошибаться при малой площади аномалии",
        "dog": "чувствителен к размеру объекта",
        "morph_global": "может удалить очень малые аномалии",
    }
    for algorithm in sorted(by_param["algorithm"].unique()):
        part = by_param[by_param.algorithm == algorithm].sort_values("f1", ascending=False)
        best = part.iloc[0]
        auc = auc_from_points(part["fpr"].to_numpy(), part["recall"].to_numpy())
        summary_rows.append(
            {
                "algorithm": algorithm,
                "best_parameter": best["parameter"],
                "precision": best["precision"],
                "recall": best["recall"],
                "fpr": best["fpr"],
                "f1": best["f1"],
                "iou": best["iou"],
                "auc": auc,
                "runtime_s": best["runtime_s"],
                "strength": strengths[algorithm],
                "weakness": weaknesses[algorithm],
            }
        )
    summary = write_csv(out_dir / "algorithm_summary.csv", summary_rows)

    roc_series = {}
    for algorithm in sorted(by_param["algorithm"].unique()):
        part = by_param[by_param.algorithm == algorithm].sort_values("fpr")
        roc_series[algorithm_title(algorithm)] = (part["fpr"], part["recall"])
    save_multi_line_plot(out_dir / "roc_curves.png", roc_series, "ROC-подобные кривые", "FPR", "TPR")
    sorted_summary = summary.sort_values("f1", ascending=False)
    plot_algorithms = [algorithm_title(algorithm) for algorithm in sorted_summary["algorithm"].tolist()]
    save_bar_plot(out_dir / "f1_by_algorithm.png", plot_algorithms, sorted_summary["f1"].tolist(), "F1-score по алгоритмам", "F1-score")
    save_bar_plot(out_dir / "iou_by_algorithm.png", plot_algorithms, sorted_summary["iou"].tolist(), "IoU по алгоритмам", "IoU")
    save_bar_plot(out_dir / "runtime_by_algorithm.png", plot_algorithms, sorted_summary["runtime_s"].tolist(), "Время обработки по алгоритмам", "Секунд на кадр")

    best_algo = sorted_summary.iloc[0]["algorithm"]
    best_param = sorted_summary.iloc[0]["best_parameter"]
    best_df = df[(df.algorithm == best_algo) & (df.parameter == best_param) & (df.has_anomaly)]
    success_id = int(best_df.sort_values("iou", ascending=False).iloc[0]["frame_id"])
    failure_id = int(best_df.sort_values("iou", ascending=True).iloc[0]["frame_id"])
    images = []
    titles = []
    for label, frame_id in [("Успешный пример", success_id), ("Неудачный пример", failure_id)]:
        item = next(item for item in dataset if item["id"] == frame_id)
        pred = predictions[(best_algo, best_param, frame_id)]
        images.extend([item["frame"], item["truth"].astype(float), pred.astype(float)])
        titles.extend([f"{label}: кадр", "Эталонная маска", "Найденная маска"])
    save_montage(out_dir / "success_failure_examples.png", images, titles, cols=3)

    recommendation = f"Рекомендуемый базовый алгоритм: `{best_algo}` с параметром `{best_param}` (F1={sorted_summary.iloc[0]['f1']:.3f}, IoU={sorted_summary.iloc[0]['iou']:.3f})."
    write_readme(
        out_dir / "README.md",
        "Эксперимент 07 - сравнение алгоритмов обнаружения",
        "Сравнить простые интерпретируемые алгоритмы обнаружения температурных аномалий на общем синтетическом наборе.",
        f"Кадров: {len(dataset)}; алгоритмы: global threshold, adaptive local, Otsu, DoG, morphology; seed: {args.seed}.",
        ["metrics.csv", "algorithm_summary.csv", "roc_curves.png", "f1_by_algorithm.png", "iou_by_algorithm.png", "runtime_by_algorithm.png", "success_failure_examples.png"],
        recommendation,
    )
    write_summary_json(
        out_dir,
        number=7,
        title="Сравнение алгоритмов обнаружения температурных аномалий на синтетических ИК-кадрах",
        varied_parameters="Алгоритм обнаружения и его параметры",
        main_metrics="Precision, Recall, FPR, F1-score, IoU, AUC, runtime",
        main_result=recommendation,
        main_plot=str(out_dir / "f1_by_algorithm.png"),
        conclusion="Сравнение показывает компромисс между простотой, устойчивостью к фону, качеством сегментации и временем обработки.",
    )
    print(f"Experiment 07 completed: {out_dir}")


if __name__ == "__main__":
    main()
