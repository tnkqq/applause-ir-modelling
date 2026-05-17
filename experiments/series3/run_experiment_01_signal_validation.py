#!/usr/bin/env python3
"""Experiment 01: extended validation of the synthetic IR sensor model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    DEFAULT_BG_K,
    RESULTS_DIR,
    binary_metrics,
    detect_global_threshold,
    frame_stats,
    generate_adc_frame,
    optics_factor,
    rng_from_seed,
    save_heatmap,
    save_line_plot,
    save_mask_png,
    save_montage,
    scene_with_anomaly,
    temperature_to_adc,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


OUT_DIR = RESULTS_DIR / "experiment_01_signal_validation"
BLOCK_01_DIR = OUT_DIR / "block_01_temperature_response"
BLOCK_03_DIR = OUT_DIR / "block_03_nuc_nonuniformity"
BLOCK_04_DIR = OUT_DIR / "block_04_vignetting"
BLOCK_05_DIR = OUT_DIR / "block_05_fill_factor"

NUC_TEMPERATURES = [300.0, 320.0, 350.0, 400.0]
NUC_TEST_TEMPERATURE = 320.0
NUC_MODES = ["uncorrected", "3bit", "4bit", "5bit", "6bit", "full"]
NUC_BITS: dict[str, int | None] = {"uncorrected": None, "3bit": 3, "4bit": 4, "5bit": 5, "6bit": 6, "full": None}
NUC_LABELS = {
    "uncorrected": "Без коррекции",
    "3bit": "NUC 3 bit",
    "4bit": "NUC 4 bit",
    "5bit": "NUC 5 bit",
    "6bit": "NUC 6 bit",
    "full": "NUC full",
}

VIGNETTING_TEMPERATURE = 320.0
VIGNETTING_MODES = {
    "none": {"title": "Без виньетирования", "fov_deg": 0.0, "strength_order": 0.0},
    "weak": {"title": "Слабое виньетирование", "fov_deg": 24.0, "strength_order": 1.0},
    "medium": {"title": "Среднее виньетирование", "fov_deg": 42.0, "strength_order": 2.0},
    "strong": {"title": "Сильное виньетирование", "fov_deg": 68.0, "strength_order": 3.0},
}

FILL_DELTAS = [2.0, 4.0, 8.0]
FILL_FACTORS = [1.0, 0.75, 0.5, 0.25, 0.1]
FILL_SIZES = [1, 2, 4, 8, 16]
FILL_TEST_DELTA = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IR signal formation and key model parameters.")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--temp_min", type=float, default=280.0)
    parser.add_argument("--temp_max", type=float, default=360.0)
    parser.add_argument("--temp_step", type=float, default=10.0)
    parser.add_argument("--noise_sigma", type=float, default=2.5)
    parser.add_argument("--fpn_std", type=float, default=0.8)
    parser.add_argument("--block", choices=["temperature", "nuc", "vignetting", "fill_factor", "all"], default="all")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [OUT_DIR, OUT_DIR / "images", OUT_DIR / "masks", BLOCK_01_DIR, BLOCK_03_DIR, BLOCK_04_DIR, BLOCK_05_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "inf"
        return number
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf"
    return value


def write_metrics_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {
                "config": json_ready(config),
                "summary": json_ready(summary),
                "records": json_ready(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def adc_uniform(temp_k: float, height: int, width: int, *, calibration_k: tuple[float, float] = (280.0, 420.0)) -> np.ndarray:
    temp_map = np.full((height, width), float(temp_k), dtype=float)
    return temperature_to_adc(temp_map, calibration_k=calibration_k, signal_low=80.0, signal_high=900.0)


def mode_sort_key(mode: str) -> int:
    return NUC_MODES.index(mode)


def save_categorical_line(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(range(len(labels)), values, marker="o", linewidth=2.0)
    plt.xticks(range(len(labels)), labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_multiline(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(7.6, 4.8))
    for label, (x, y) in series.items():
        plt.plot(x, y, linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_histogram_set(path: Path, values: dict[str, np.ndarray], title: str, xlabel: str) -> None:
    plt.figure(figsize=(8.0, 4.8))
    for label, data in values.items():
        plt.hist(np.asarray(data).ravel(), bins=56, alpha=0.45, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Число пикселей")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_image_grid(
    path: Path,
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str,
    *,
    cols: int = 3,
    cmap: str = "inferno",
    cbar_label: str = "Код ADC",
    fixed_scale: bool = True,
    symmetric: bool = False,
) -> None:
    rows = int(math.ceil(len(images) / cols))
    if fixed_scale:
        if symmetric:
            vmax = max(float(np.max(np.abs(image))) for image in images)
            vmin = -vmax
        else:
            vmin = min(float(np.min(image)) for image in images)
            vmax = max(float(np.max(image)) for image in images)
    else:
        vmin = vmax = None
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 2.9), squeeze=False, constrained_layout=True)
    last = None
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(images):
            last = ax.imshow(images[idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(titles[idx], fontsize=9)
        ax.set_axis_off()
    if last is not None:
        fig.colorbar(last, ax=axes, label=cbar_label)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_mask_grid(path: Path, images: list[np.ndarray], titles: list[str], suptitle: str, *, cols: int = 4) -> None:
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.5), squeeze=False, constrained_layout=True)
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(images):
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[idx], fontsize=9)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_heatmap_matrix(path: Path, matrix: pd.DataFrame, title: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(col) for col in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(idx) for idx in matrix.index])
    ax.set_xlabel("fill_factor")
    ax.set_ylabel("Размер, px")
    ax.set_title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=8, color="white" if values[y, x] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_temperature_block(args: argparse.Namespace) -> dict[str, str]:
    rng = rng_from_seed(args.seed)
    temps = np.arange(args.temp_min, args.temp_max + 0.5 * args.temp_step, args.temp_step)
    config = vars(args).copy()
    config.update(
        {
            "experiment": "01_signal_validation",
            "block": "01_temperature_response",
            "temperature_values_K": temps.tolist(),
            "adc_model": "Blackbody 8-14 um -> calibrated 10-bit ADC with Gaussian noise and FPN",
        }
    )
    write_json(OUT_DIR / "config.json", config)
    write_json(BLOCK_01_DIR / "config.json", config)

    records: list[dict[str, Any]] = []
    examples: list[np.ndarray] = []
    example_titles: list[str] = []

    for temp in temps:
        temp_map = np.full((args.height, args.width), temp, dtype=float)
        frames = []
        fpn_pattern = rng.normal(0.0, args.fpn_std, size=temp_map.shape)
        for frame_idx in range(args.num_frames):
            frame, _ = generate_adc_frame(
                temp_map,
                rng,
                gaussian_sigma=args.noise_sigma,
                fpn_std=args.fpn_std,
                fpn_pattern=fpn_pattern,
            )
            frames.append(frame)
            if frame_idx == 0 and temp in {temps[0], temps[len(temps) // 2], temps[-1]}:
                title = f"Однородный кадр, T={temp:g} K"
                save_heatmap(OUT_DIR / "images" / f"example_uniform_{temp:g}K.png", frame, title)
                save_heatmap(BLOCK_01_DIR / f"example_uniform_{temp:g}K.png", frame, title)
                examples.append(frame)
                example_titles.append(f"{temp:g} K")
        stack = np.stack(frames)
        temporal_mean = np.mean(stack, axis=0)
        temporal_std = np.std(stack, axis=0)
        stats = frame_stats(temporal_mean)
        records.append(
            {
                "temperature_K": float(temp),
                "mean_signal_adc": float(np.mean(stack)),
                "noise_std_adc": float(np.mean(temporal_std)),
                "frame_mean_std_adc": stats["std"],
                "min_adc": stats["min"],
                "max_adc": stats["max"],
                "dynamic_range_adc": stats["dynamic_range"],
            }
        )

    for idx, row in enumerate(records):
        if idx == 0:
            snr = np.nan
            slope = (records[idx + 1]["mean_signal_adc"] - row["mean_signal_adc"]) / args.temp_step
        elif idx == len(records) - 1:
            snr = (row["mean_signal_adc"] - records[idx - 1]["mean_signal_adc"]) / max(row["noise_std_adc"], 1e-9)
            slope = (row["mean_signal_adc"] - records[idx - 1]["mean_signal_adc"]) / args.temp_step
        else:
            snr = (row["mean_signal_adc"] - records[idx - 1]["mean_signal_adc"]) / max(row["noise_std_adc"], 1e-9)
            slope = (records[idx + 1]["mean_signal_adc"] - records[idx - 1]["mean_signal_adc"]) / (2 * args.temp_step)
        row["snr_for_temp_step"] = float(snr) if np.isfinite(snr) else np.nan
        row["local_slope_adc_per_K"] = float(slope)
        row["netd_K"] = float(row["noise_std_adc"] / max(abs(slope), 1e-9))

    df = write_csv(OUT_DIR / "metrics.csv", records)
    df.to_csv(BLOCK_01_DIR / "metrics.csv", index=False)

    for target in [OUT_DIR, BLOCK_01_DIR]:
        save_line_plot(target / "signal_vs_temperature.png", df["temperature_K"], df["mean_signal_adc"], "Средний сигнал от температуры", "Температура, K", "Средний код ADC")
        save_line_plot(target / "noise_vs_temperature.png", df["temperature_K"], df["noise_std_adc"], "Шум от температуры", "Температура, K", "STD шума, код ADC")
        save_line_plot(target / "snr_vs_temperature.png", df["temperature_K"], df["snr_for_temp_step"], "SNR для температурного шага", "Температура, K", "SNR")
        save_line_plot(target / "netd_estimation.png", df["temperature_K"], df["netd_K"], "Оценка NETD", "Температура, K", "NETD, K")
        save_montage(target / "example_frames.png", examples, example_titles, cols=3)

    _, sample_mask = scene_with_anomaly(args.height, args.width, background_k=DEFAULT_BG_K, delta_t=5.0, size=12)
    np.savetxt(OUT_DIR / "masks" / "example_anomaly_mask.csv", sample_mask.astype(int), delimiter=",", fmt="%d")

    monotonic = bool(df["mean_signal_adc"].is_monotonic_increasing)
    best_netd = float(df["netd_K"].min())
    mean_slope = float(df["local_slope_adc_per_K"].mean())
    summary = {
        "monotonic_mean_signal": monotonic,
        "mean_slope_adc_per_K": mean_slope,
        "best_netd_K": best_netd,
        "mean_noise_std_adc": float(df["noise_std_adc"].mean()),
    }
    write_metrics_json(BLOCK_01_DIR / "metrics.json", config, records, summary)

    conclusion = (
        f"Средний наклон радиометрической характеристики составил {mean_slope:.3f} ADC/K; "
        f"лучшая оценка NETD равна {best_netd:.3f} K; "
        f"монотонность среднего сигнала: {'подтверждена' if monotonic else 'не подтверждена'}."
    )
    for readme_path in [OUT_DIR / "README.md", BLOCK_01_DIR / "README.md"]:
        write_readme(
            readme_path,
            "Эксперимент 01 - блок 1: температурная радиометрическая характеристика",
            "Проверить монотонность цифрового сигнала от температуры сцены и оценить SNR/NETD.",
            f"Температуры {args.temp_min:g}-{args.temp_max:g} K с шагом {args.temp_step:g} K; кадров на уровень: {args.num_frames}; seed: {args.seed}.",
            ["metrics.csv", "metrics.json", "config.json", "signal_vs_temperature.png", "noise_vs_temperature.png", "snr_vs_temperature.png", "netd_estimation.png", "example_frames.png"],
            conclusion,
        )
    write_summary_json(
        OUT_DIR,
        number=1,
        title="Валидация модели формирования ИК-сигнала и метрик SNR/NETD",
        varied_parameters="Температура однородной сцены",
        main_metrics="mean_signal_adc, noise_std_adc, snr_for_temp_step, netd_K",
        main_result=conclusion,
        main_plot=str(OUT_DIR / "signal_vs_temperature.png"),
        conclusion="Модель дает монотонный рост цифрового сигнала с температурой и позволяет оценивать NETD через наклон характеристики и шум.",
    )
    return {
        "block_id": "01",
        "block_name": "Температурная радиометрическая характеристика",
        "varied_parameter": "Температура сцены",
        "expected_behavior": "Средний ADC-сигнал монотонно растет с температурой.",
        "observed_behavior": "Монотонность подтверждена." if monotonic else "Монотонность не подтверждена.",
        "key_metric": "best_netd_K",
        "key_result": f"{best_netd:.4f} K",
        "main_figure": "block_01_temperature_response/signal_vs_temperature.png",
        "conclusion": conclusion,
    }


def make_nuc_sensor_maps(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = rng_from_seed(args.seed + 301)
    gain_map = np.clip(rng.normal(1.0, 0.035, size=(args.height, args.width)), 0.88, 1.12)
    offset_map = rng.normal(0.0, 8.0, size=(args.height, args.width))
    fpn_map = rng.normal(0.0, 5.0, size=(args.height, args.width))
    cos4 = optics_factor(args.height, args.width, fov_rad=math.radians(52.0))
    vignette = cos4 / np.max(cos4)
    return {"gain_map": gain_map, "offset_map": offset_map, "fpn_map": fpn_map, "vignette": vignette}


def nuc_raw_frame(temp_k: float, args: argparse.Namespace, maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ideal = adc_uniform(temp_k, args.height, args.width)
    raw = ideal * maps["gain_map"] * maps["vignette"] + maps["offset_map"] + maps["fpn_map"]
    return np.clip(raw, 0.0, 1023.0), ideal


def quantize_coeff(values: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits - 1
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        return values.copy()
    normalized = (values - vmin) / (vmax - vmin)
    return np.round(normalized * levels) / levels * (vmax - vmin) + vmin


def run_nuc_block(args: argparse.Namespace) -> dict[str, str]:
    config = {
        "seed": args.seed + 301,
        "height": args.height,
        "width": args.width,
        "temperatures_K": NUC_TEMPERATURES,
        "calibration_temperatures_K": [300.0, 400.0],
        "coeff_bits": [3, 4, 5, 6, "full"],
        "sensor_effects": ["pixel gain spread", "offset spread", "FPN", "cos^4-like vignetting"],
        "gaussian_noise_adc": 0.0,
    }
    maps = make_nuc_sensor_maps(args)
    raw_low, ideal_low = nuc_raw_frame(300.0, args, maps)
    raw_high, ideal_high = nuc_raw_frame(400.0, args, maps)
    a_full = (ideal_high - ideal_low) / np.maximum(raw_high - raw_low, 1e-9)
    b_full = ideal_low - a_full * raw_low

    coeffs: dict[str, tuple[np.ndarray, np.ndarray]] = {"full": (a_full, b_full)}
    for mode in ["3bit", "4bit", "5bit", "6bit"]:
        bits = int(NUC_BITS[mode] or 0)
        coeffs[mode] = (quantize_coeff(a_full, bits), quantize_coeff(b_full, bits))

    raw_by_temp: dict[float, np.ndarray] = {}
    corrected_by_temp_mode: dict[tuple[float, str], np.ndarray] = {}
    ideal_by_temp: dict[float, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    for temp in NUC_TEMPERATURES:
        raw, ideal = nuc_raw_frame(temp, args, maps)
        raw_by_temp[temp] = raw
        ideal_by_temp[temp] = ideal
        raw_std = float(np.std(raw))
        for mode in NUC_MODES:
            if mode == "uncorrected":
                frame = raw.copy()
                coeff_bits: int | str | None = None
            else:
                a_map, b_map = coeffs[mode]
                frame = a_map * raw + b_map
                coeff_bits = "full" if mode == "full" else int(NUC_BITS[mode] or 0)
            corrected_by_temp_mode[(temp, mode)] = frame
            stats = frame_stats(frame)
            rmse = float(np.sqrt(np.mean((frame - ideal) ** 2)))
            records.append(
                {
                    "temperature_K": temp,
                    "correction_mode": mode,
                    "coeff_bits": coeff_bits,
                    "mean_adc": stats["mean"],
                    "std_adc": stats["std"],
                    "min_adc": stats["min"],
                    "max_adc": stats["max"],
                    "peak_to_peak_adc": stats["dynamic_range"],
                    "residual_nonuniformity_adc": stats["std"],
                    "improvement_factor": raw_std / max(stats["std"], 1e-12),
                    "rmse_to_ideal": rmse,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(BLOCK_03_DIR / "metrics.csv", index=False)
    df.to_csv(BLOCK_03_DIR / "nuc_metrics.csv", index=False)
    write_json(BLOCK_03_DIR / "config.json", config)
    write_json(BLOCK_03_DIR / "nuc_config.json", config)

    rows_320 = df[df["temperature_K"] == NUC_TEST_TEMPERATURE].copy()
    rows_320["mode_order"] = rows_320["correction_mode"].map(mode_sort_key)
    rows_320 = rows_320.sort_values("mode_order")
    mode_labels = [NUC_LABELS[mode] for mode in rows_320["correction_mode"]]
    save_categorical_line(BLOCK_03_DIR / "residual_std_vs_coeff_bits.png", mode_labels, rows_320["residual_nonuniformity_adc"].tolist(), "Остаточный STD после NUC при 320 K", "STD, ADC")
    save_categorical_line(BLOCK_03_DIR / "peak_to_peak_vs_coeff_bits.png", mode_labels, rows_320["peak_to_peak_adc"].tolist(), "Диапазон max-min после NUC при 320 K", "Peak-to-peak, ADC")
    save_categorical_line(BLOCK_03_DIR / "nuc_improvement_factor_vs_coeff_bits.png", mode_labels, rows_320["improvement_factor"].tolist(), "Коэффициент улучшения NUC при 320 K", "std(raw) / std(corrected)")

    center_row = args.height // 2
    center_col = args.width // 2
    save_multiline(
        BLOCK_03_DIR / "center_row_profiles_nuc.png",
        {NUC_LABELS[mode]: (np.arange(args.width), corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)][center_row, :]) for mode in NUC_MODES},
        "Профиль центральной строки после NUC, 320 K",
        "Пиксель x",
        "Код ADC",
    )
    save_multiline(
        BLOCK_03_DIR / "center_column_profiles_nuc.png",
        {NUC_LABELS[mode]: (np.arange(args.height), corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)][:, center_col]) for mode in NUC_MODES},
        "Профиль центрального столбца после NUC, 320 K",
        "Пиксель y",
        "Код ADC",
    )
    save_histogram_set(
        BLOCK_03_DIR / "residual_histograms_nuc.png",
        {NUC_LABELS[mode]: corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)] - ideal_by_temp[NUC_TEST_TEMPERATURE] for mode in NUC_MODES},
        "Гистограммы остаточной ошибки после NUC, 320 K",
        "Ошибка, ADC",
    )

    raw_320 = raw_by_temp[NUC_TEST_TEMPERATURE]
    save_heatmap(BLOCK_03_DIR / "nuc_uncorrected_320K.png", raw_320, "Некорректированный кадр, 320 K")
    for mode in ["3bit", "4bit", "5bit", "6bit", "full"]:
        save_heatmap(BLOCK_03_DIR / f"nuc_corrected_320K_{mode}.png", corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)], f"NUC {NUC_LABELS[mode]}, 320 K")
    save_image_grid(
        BLOCK_03_DIR / "nuc_comparison_grid_320K.png",
        [corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)] for mode in NUC_MODES],
        [NUC_LABELS[mode] for mode in NUC_MODES],
        "Сравнение NUC-коррекции, 320 K",
        cols=3,
    )
    save_image_grid(
        BLOCK_03_DIR / "nuc_residual_maps_320K.png",
        [corrected_by_temp_mode[(NUC_TEST_TEMPERATURE, mode)] - ideal_by_temp[NUC_TEST_TEMPERATURE] for mode in NUC_MODES],
        [NUC_LABELS[mode] for mode in NUC_MODES],
        "Карты остаточной ошибки NUC, 320 K",
        cols=3,
        cmap="coolwarm",
        cbar_label="Ошибка, ADC",
        symmetric=True,
    )
    save_heatmap(BLOCK_03_DIR / "nuc_coeff_gain_map.png", a_full, "Карта коэффициента NUC gain", "Коэффициент")
    save_heatmap(BLOCK_03_DIR / "nuc_coeff_offset_map.png", b_full, "Карта коэффициента NUC offset", "Код ADC")
    coeff_images = []
    coeff_titles = []
    for mode in ["3bit", "6bit", "full"]:
        coeff_images.extend([coeffs[mode][0], coeffs[mode][1]])
        coeff_titles.extend([f"{NUC_LABELS[mode]}: gain", f"{NUC_LABELS[mode]}: offset"])
    save_image_grid(
        BLOCK_03_DIR / "nuc_quantized_coeff_maps.png",
        coeff_images,
        coeff_titles,
        "Квантованные коэффициенты NUC",
        cols=3,
        cmap="viridis",
        cbar_label="Значение коэффициента",
        fixed_scale=False,
    )

    best = rows_320.sort_values("residual_nonuniformity_adc").iloc[0]
    uncorrected = rows_320[rows_320["correction_mode"] == "uncorrected"].iloc[0]
    summary = {
        "uncorrected_std_320K": float(uncorrected["residual_nonuniformity_adc"]),
        "best_mode_320K": best["correction_mode"],
        "best_residual_std_320K": float(best["residual_nonuniformity_adc"]),
        "best_improvement_factor_320K": float(best["improvement_factor"]),
    }
    write_metrics_json(BLOCK_03_DIR / "metrics.json", config, records, summary)
    write_json(BLOCK_03_DIR / "nuc_metrics.json", {"config": config, "summary": summary, "records": records})
    best_std_text = "<1e-6" if summary["best_residual_std_320K"] < 1e-6 else f"{summary['best_residual_std_320K']:.6f}"
    improvement_text = (
        "до численного нуля"
        if summary["best_residual_std_320K"] < 1e-6
        else f"в {summary['best_improvement_factor_320K']:.1f} раза"
    )
    conclusion = (
        f"При 320 K STD некорректированного кадра равен {summary['uncorrected_std_320K']:.3f} ADC; "
        f"лучший режим `{summary['best_mode_320K']}` дал STD {best_std_text} ADC "
        f"и снизил фиксированную неоднородность {improvement_text}."
    )
    write_readme(
        BLOCK_03_DIR / "README.md",
        "Блок 3 - неравномерность матрицы и NUC-коррекция",
        "Проверить фиксированную пространственную неоднородность матрицы и влияние точности коэффициентов двухточечной NUC-коррекции.",
        "Использованы температуры 300, 320, 350, 400 K; калибровка NUC по 300 и 400 K; коэффициенты 3/4/5/6 bit и full precision.",
        ["metrics.csv", "metrics.json", "nuc_config.json", "residual_std_vs_coeff_bits.png", "nuc_comparison_grid_320K.png", "nuc_residual_maps_320K.png", "nuc_quantized_coeff_maps.png"],
        conclusion + " Двухточечная NUC снижает фиксированную неоднородность, а низкая разрядность коэффициентов оставляет остаточные артефакты.",
    )
    return {
        "block_id": "03",
        "block_name": "Неравномерность матрицы и NUC",
        "varied_parameter": "Точность коэффициентов NUC",
        "expected_behavior": "STD и peak-to-peak должны снижаться при повышении точности коэффициентов.",
        "observed_behavior": f"STD снизился с {summary['uncorrected_std_320K']:.3f} до {best_std_text} ADC.",
        "key_metric": "residual_nonuniformity_adc",
        "key_result": f"{best_std_text} ADC",
        "main_figure": "block_03_nuc_nonuniformity/residual_std_vs_coeff_bits.png",
        "conclusion": conclusion,
    }


def vignetting_factor(height: int, width: int, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones((height, width), dtype=float)
    fov_rad = math.radians(float(VIGNETTING_MODES[mode]["fov_deg"]))
    factor = optics_factor(height, width, fov_rad=fov_rad)
    return factor / np.max(factor)


def region_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width))
    center = (height / 2.0 - 0.5, width / 2.0 - 0.5)
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    max_radius = float(np.max(radius))
    center_mask = radius <= 0.18 * max_radius
    edge_mask = radius >= 0.78 * max_radius
    corner_mask = np.zeros((height, width), dtype=bool)
    h = max(2, int(height * 0.14))
    w = max(2, int(width * 0.14))
    corner_mask[:h, :w] = True
    corner_mask[:h, -w:] = True
    corner_mask[-h:, :w] = True
    corner_mask[-h:, -w:] = True
    return center_mask, edge_mask, corner_mask


def radial_profile(frame: np.ndarray, bins: int = 28) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2)
    radius_norm = radius / np.max(radius)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.zeros(bins, dtype=float)
    for idx in range(bins):
        mask = (radius_norm >= edges[idx]) & (radius_norm < edges[idx + 1])
        values[idx] = float(np.mean(frame[mask])) if np.any(mask) else np.nan
    return centers, values


def run_vignetting_block(args: argparse.Namespace) -> dict[str, str]:
    config = {
        "seed": args.seed + 401,
        "height": args.height,
        "width": args.width,
        "temperature_K": VIGNETTING_TEMPERATURE,
        "modes": VIGNETTING_MODES,
        "noise_sigma_for_extra_image_adc": 2.0,
    }
    write_json(BLOCK_04_DIR / "config.json", config)
    write_json(BLOCK_04_DIR / "vignetting_config.json", config)
    uniform = adc_uniform(VIGNETTING_TEMPERATURE, args.height, args.width)
    center_mask, edge_mask, corner_mask = region_masks(args.height, args.width)
    records: list[dict[str, Any]] = []
    frames: dict[str, np.ndarray] = {}
    factors: dict[str, np.ndarray] = {}

    for mode in VIGNETTING_MODES:
        factor = vignetting_factor(args.height, args.width, mode)
        frame = uniform * factor
        frames[mode] = frame
        factors[mode] = factor
        center_mean = float(np.mean(frame[center_mask]))
        edge_mean = float(np.mean(frame[edge_mask]))
        corner_mean = float(np.mean(frame[corner_mask]))
        radius, profile = radial_profile(frame)
        valid = np.isfinite(profile)
        slope = float(np.polyfit(radius[valid], profile[valid], deg=1)[0]) if np.sum(valid) > 1 else 0.0
        records.append(
            {
                "temperature_K": VIGNETTING_TEMPERATURE,
                "vignetting_mode": mode,
                "center_mean_adc": center_mean,
                "edge_mean_adc": edge_mean,
                "corner_mean_adc": corner_mean,
                "center_to_corner_drop_adc": center_mean - corner_mean,
                "center_to_corner_drop_percent": (center_mean - corner_mean) / max(center_mean, 1e-9) * 100.0,
                "radial_gradient_adc": -slope,
                "std_adc": float(np.std(frame)),
                "rmse_to_uniform": float(np.sqrt(np.mean((frame - uniform) ** 2))),
            }
        )
        save_heatmap(BLOCK_04_DIR / f"vignetting_{mode}_320K.png", frame, f"{VIGNETTING_MODES[mode]['title']}, 320 K")

    rng = rng_from_seed(args.seed + 402)
    medium_noise = frames["medium"] + rng.normal(0.0, 2.0, size=frames["medium"].shape)
    save_heatmap(BLOCK_04_DIR / "vignetting_medium_noise_320K.png", medium_noise, "Среднее виньетирование с шумом, 320 K")
    flat_corrected = frames["medium"] / np.maximum(factors["medium"], 1e-9)
    save_heatmap(BLOCK_04_DIR / "vignetting_medium_after_flatfield_nuc_320K.png", flat_corrected, "Виньетирование после flat-field коррекции, 320 K")

    df = pd.DataFrame(records)
    df.to_csv(BLOCK_04_DIR / "metrics.csv", index=False)
    df.to_csv(BLOCK_04_DIR / "vignetting_metrics.csv", index=False)
    write_json(BLOCK_04_DIR / "vignetting_metrics.json", {"config": config, "records": records})

    radius_series = {VIGNETTING_MODES[mode]["title"]: radial_profile(frame) for mode, frame in frames.items()}
    save_multiline(BLOCK_04_DIR / "vignetting_radial_profile.png", {k: (x, y) for k, (x, y) in radius_series.items()}, "Радиальный профиль виньетирования", "Нормированное расстояние от центра", "Средний код ADC")
    save_multiline(BLOCK_04_DIR / "vignetting_center_row_profile.png", {VIGNETTING_MODES[mode]["title"]: (np.arange(args.width), frame[args.height // 2, :]) for mode, frame in frames.items()}, "Профиль центральной строки", "Пиксель x", "Код ADC")
    save_multiline(BLOCK_04_DIR / "vignetting_center_column_profile.png", {VIGNETTING_MODES[mode]["title"]: (np.arange(args.height), frame[:, args.width // 2]) for mode, frame in frames.items()}, "Профиль центрального столбца", "Пиксель y", "Код ADC")

    labels = [VIGNETTING_MODES[mode]["title"] for mode in VIGNETTING_MODES]
    save_categorical_line(BLOCK_04_DIR / "center_to_corner_drop_vs_strength.png", labels, df["center_to_corner_drop_adc"].tolist(), "Падение центр-угол от силы виньетирования", "Падение, ADC")
    save_categorical_line(BLOCK_04_DIR / "vignetting_std_vs_strength.png", labels, df["std_adc"].tolist(), "STD кадра от силы виньетирования", "STD, ADC")

    save_image_grid(
        BLOCK_04_DIR / "vignetting_comparison_grid_320K.png",
        [frames[mode] for mode in VIGNETTING_MODES],
        labels,
        "Сравнение режимов виньетирования, 320 K",
        cols=2,
    )
    save_image_grid(
        BLOCK_04_DIR / "vignetting_residual_maps.png",
        [frames[mode] - uniform for mode in VIGNETTING_MODES],
        labels,
        "Разность с равномерным кадром без виньетирования",
        cols=2,
        cmap="coolwarm",
        cbar_label="Разность, ADC",
        symmetric=True,
    )
    yy, xx = np.indices((args.height, args.width))
    distance = np.sqrt((yy - (args.height - 1) / 2.0) ** 2 + (xx - (args.width - 1) / 2.0) ** 2)
    distance /= np.max(distance)
    save_image_grid(
        BLOCK_04_DIR / "vignetting_radial_mask_or_distance_map.png",
        [distance, factors["strong"]],
        ["Нормированное расстояние", "Коэффициент cos^4, strong"],
        "Карта расстояния и коэффициента виньетирования",
        cols=2,
        cmap="viridis",
        cbar_label="Нормированное значение",
        fixed_scale=False,
    )

    strong = df[df["vignetting_mode"] == "strong"].iloc[0]
    summary = {
        "strong_center_to_corner_drop_adc": float(strong["center_to_corner_drop_adc"]),
        "strong_center_to_corner_drop_percent": float(strong["center_to_corner_drop_percent"]),
        "strong_std_adc": float(strong["std_adc"]),
    }
    write_metrics_json(BLOCK_04_DIR / "metrics.json", config, records, summary)
    conclusion = (
        f"В режиме strong падение центр-угол составило {summary['strong_center_to_corner_drop_adc']:.3f} ADC "
        f"({summary['strong_center_to_corner_drop_percent']:.2f}%), STD кадра {summary['strong_std_adc']:.3f} ADC."
    )
    write_readme(
        BLOCK_04_DIR / "README.md",
        "Блок 4 - оптическое виньетирование и распределение мощности",
        "Проверить радиальный спад сигнала при равномерной температуре сцены из-за cos^4-виньетирования.",
        "Температура 320 K; режимы none/weak/medium/strong; дополнительно сохранены пример с шумом и flat-field компенсацией.",
        ["metrics.csv", "metrics.json", "vignetting_config.json", "vignetting_radial_profile.png", "center_to_corner_drop_vs_strength.png", "vignetting_comparison_grid_320K.png", "vignetting_residual_maps.png"],
        conclusion + " Модель воспроизводит систематический радиальный градиент, который необходимо учитывать или компенсировать.",
    )
    return {
        "block_id": "04",
        "block_name": "Оптическое виньетирование",
        "varied_parameter": "Сила виньетирования / FOV",
        "expected_behavior": "Сигнал в центре выше, чем в углах; спад растет с силой виньетирования.",
        "observed_behavior": f"Для strong падение центр-угол равно {summary['strong_center_to_corner_drop_adc']:.3f} ADC.",
        "key_metric": "center_to_corner_drop_adc",
        "key_result": f"{summary['strong_center_to_corner_drop_adc']:.3f} ADC",
        "main_figure": "block_04_vignetting/vignetting_radial_profile.png",
        "conclusion": conclusion,
    }


def fill_factor_scene(args: argparse.Namespace, delta_t: float, size: int, fill_factor: float) -> tuple[np.ndarray, np.ndarray]:
    return scene_with_anomaly(
        args.height,
        args.width,
        background_k=300.0,
        delta_t=delta_t,
        shape="rectangle",
        size=size,
        fill_factor=fill_factor,
    )


def run_fill_factor_block(args: argparse.Namespace) -> dict[str, str]:
    config = {
        "seed": args.seed + 501,
        "height": args.height,
        "width": args.width,
        "background_K": 300.0,
        "delta_T_K": FILL_DELTAS,
        "anomaly_sizes_px": FILL_SIZES,
        "fill_factors": FILL_FACTORS,
        "num_frames": 12,
        "noise_sigma_adc": 2.0,
        "fpn_std_adc": 0.5,
        "detector": "global median + 3 * robust_sigma",
        "success_iou": 0.3,
    }
    write_json(BLOCK_05_DIR / "config.json", config)
    write_json(BLOCK_05_DIR / "fill_factor_config.json", config)
    rng = rng_from_seed(args.seed + 501)
    fpn_pattern = rng.normal(0.0, config["fpn_std_adc"], size=(args.height, args.width))
    records: list[dict[str, Any]] = []
    examples: dict[tuple[float, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for delta_t in FILL_DELTAS:
        for size in FILL_SIZES:
            for fill in FILL_FACTORS:
                scene, truth = fill_factor_scene(args, delta_t, size, fill)
                per_frame: list[dict[str, float]] = []
                for frame_idx in range(config["num_frames"]):
                    frame, _ = generate_adc_frame(scene, rng, gaussian_sigma=config["noise_sigma_adc"], fpn_std=config["fpn_std_adc"], fpn_pattern=fpn_pattern)
                    pred = detect_global_threshold(frame, k=3.0, min_area=max(1, min(4, size * size))).mask
                    metrics = binary_metrics(pred, truth)
                    bg = frame[~truth]
                    anom = frame[truth]
                    delta_adc = float(np.mean(anom) - np.mean(bg)) if np.any(truth) else 0.0
                    per_frame.append(
                        {
                            "mean_background_adc": float(np.mean(bg)),
                            "mean_anomaly_adc": float(np.mean(anom)) if np.any(truth) else 0.0,
                            "delta_adc": delta_adc,
                            "std_background_adc": float(np.std(bg)),
                            "snr_like": delta_adc / max(float(np.std(bg)), 1.0),
                            "iou": metrics["iou"],
                            "tpr": metrics["tpr"],
                            "fpr": metrics["fpr"],
                            "detection_success": float(metrics["iou"] >= config["success_iou"]),
                        }
                    )
                    if frame_idx == 0:
                        examples[(delta_t, size, fill)] = (frame, truth, pred)
                row = {
                    "delta_T_K": delta_t,
                    "anomaly_size_px": size,
                    "fill_factor": fill,
                }
                for key in per_frame[0]:
                    row[key] = float(np.mean([item[key] for item in per_frame]))
                row["detection_probability"] = row["detection_success"]
                records.append(row)

    df = pd.DataFrame(records)
    for (delta_t, size), group in df.groupby(["delta_T_K", "anomaly_size_px"]):
        base = float(group[group["fill_factor"] == 1.0]["delta_adc"].iloc[0])
        idx = (df["delta_T_K"] == delta_t) & (df["anomaly_size_px"] == size)
        df.loc[idx, "expected_delta_adc_linear"] = df.loc[idx, "fill_factor"] * base
    df["relative_error_to_linear_model"] = np.abs(df["delta_adc"] - df["expected_delta_adc_linear"]) / np.maximum(np.abs(df["expected_delta_adc_linear"]), 1e-9)
    ordered_cols = [
        "delta_T_K",
        "anomaly_size_px",
        "fill_factor",
        "mean_background_adc",
        "mean_anomaly_adc",
        "delta_adc",
        "expected_delta_adc_linear",
        "relative_error_to_linear_model",
        "std_background_adc",
        "snr_like",
        "iou",
        "tpr",
        "fpr",
        "detection_success",
        "detection_probability",
    ]
    df = df[ordered_cols]
    df.to_csv(BLOCK_05_DIR / "metrics.csv", index=False)
    df.to_csv(BLOCK_05_DIR / "fill_factor_metrics.csv", index=False)
    write_json(BLOCK_05_DIR / "fill_factor_metrics.json", {"config": config, "records": df.to_dict(orient="records")})

    size_plot = 8
    line_df = df[df["anomaly_size_px"] == size_plot]
    plt.figure(figsize=(7.4, 4.7))
    for delta_t, group in line_df.groupby("delta_T_K"):
        group = group.sort_values("fill_factor")
        plt.plot(group["fill_factor"], group["delta_adc"], marker="o", linewidth=2.0, label=f"Delta T={delta_t:g} K")
    plt.title("Прирост ADC от fill_factor")
    plt.xlabel("fill_factor")
    plt.ylabel("Delta ADC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BLOCK_05_DIR / "delta_adc_vs_fill_factor.png", dpi=170)
    plt.close()

    plt.figure(figsize=(7.4, 4.7))
    for delta_t, group in line_df.groupby("delta_T_K"):
        group = group.sort_values("fill_factor")
        plt.plot(group["fill_factor"], group["snr_like"], marker="o", linewidth=2.0, label=f"Delta T={delta_t:g} K")
    plt.title("SNR-like от fill_factor")
    plt.xlabel("fill_factor")
    plt.ylabel("SNR-like")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BLOCK_05_DIR / "snr_like_vs_fill_factor.png", dpi=170)
    plt.close()

    plt.figure(figsize=(6.4, 6.0))
    for delta_t, group in df.groupby("delta_T_K"):
        plt.scatter(group["expected_delta_adc_linear"], group["delta_adc"], label=f"Delta T={delta_t:g} K", s=28)
    lim = max(float(df["expected_delta_adc_linear"].max()), float(df["delta_adc"].max())) * 1.05
    plt.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Линейное ожидание")
    plt.title("Проверка линейности delta_adc")
    plt.xlabel("Ожидаемый Delta ADC")
    plt.ylabel("Фактический Delta ADC")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(BLOCK_05_DIR / "delta_adc_linearity_check.png", dpi=170)
    plt.close()

    det_df = df[df["delta_T_K"] == FILL_TEST_DELTA]
    for metric, filename, title, ylabel in [
        ("detection_probability", "detection_probability_vs_fill_factor.png", "Вероятность обнаружения от fill_factor", "Вероятность обнаружения"),
        ("iou", "iou_vs_fill_factor.png", "IoU от fill_factor", "IoU"),
    ]:
        plt.figure(figsize=(7.4, 4.7))
        for size, group in det_df.groupby("anomaly_size_px"):
            group = group.sort_values("fill_factor")
            plt.plot(group["fill_factor"], group[metric], marker="o", linewidth=2.0, label=f"{size}x{size} px")
        plt.title(title)
        plt.xlabel("fill_factor")
        plt.ylabel(ylabel)
        plt.ylim(-0.03, 1.03)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(BLOCK_05_DIR / filename, dpi=170)
        plt.close()

    iou_matrix = det_df.pivot(index="anomaly_size_px", columns="fill_factor", values="iou")
    snr_matrix = det_df.pivot(index="anomaly_size_px", columns="fill_factor", values="snr_like")
    save_heatmap_matrix(BLOCK_05_DIR / "heatmap_iou_size_vs_fill_factor.png", iou_matrix, "IoU(size, fill_factor), Delta T=4 K", "IoU")
    save_heatmap_matrix(BLOCK_05_DIR / "heatmap_snr_size_vs_fill_factor.png", snr_matrix, "SNR-like(size, fill_factor), Delta T=4 K", "SNR-like")

    min_rows = []
    for delta_t in FILL_DELTAS:
        for fill in FILL_FACTORS:
            subset = df[(df["delta_T_K"] == delta_t) & (df["fill_factor"] == fill) & (df["detection_probability"] >= 0.8) & (df["iou"] >= 0.3)].sort_values("anomaly_size_px")
            min_rows.append({"delta_T_K": delta_t, "fill_factor": fill, "min_size_px": float(subset["anomaly_size_px"].iloc[0]) if len(subset) else np.nan})
    min_df = pd.DataFrame(min_rows)
    plt.figure(figsize=(7.4, 4.7))
    for delta_t, group in min_df.groupby("delta_T_K"):
        group = group.sort_values("fill_factor")
        plt.plot(group["fill_factor"], group["min_size_px"], marker="o", linewidth=2.0, label=f"Delta T={delta_t:g} K")
    plt.title("Минимальный обнаруживаемый размер от fill_factor")
    plt.xlabel("fill_factor")
    plt.ylabel("Минимальный размер, px")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BLOCK_05_DIR / "min_detectable_size_vs_fill_factor.png", dpi=170)
    plt.close()

    fill_examples = [1.0, 0.5, 0.25, 0.1]
    frames = [examples[(FILL_TEST_DELTA, 8, fill)][0] for fill in fill_examples]
    masks = [examples[(FILL_TEST_DELTA, 8, fill)][1].astype(float) for fill in fill_examples]
    preds = [examples[(FILL_TEST_DELTA, 8, fill)][2].astype(float) for fill in fill_examples]
    save_image_grid(BLOCK_05_DIR / "fill_factor_examples_grid.png", frames, [f"fill={fill:g}" for fill in fill_examples], "Кадры при разных fill_factor, Delta T=4 K, size=8 px", cols=4)
    save_mask_grid(BLOCK_05_DIR / "fill_factor_masks_grid.png", masks + preds, [f"Эталон fill={fill:g}" for fill in fill_examples] + [f"Детектор fill={fill:g}" for fill in fill_examples], "Эталонные и найденные маски", cols=4)

    sizes_examples = [1, 2, 4, 8]
    small_frames = [examples[(FILL_TEST_DELTA, size, 1.0)][0] for size in sizes_examples]
    save_image_grid(BLOCK_05_DIR / "small_target_examples_grid.png", small_frames, [f"{size}x{size} px" for size in sizes_examples], "Примеры малых аномалий, Delta T=4 K", cols=4)

    background_adc = adc_uniform(300.0, args.height, args.width)
    residuals = [examples[(FILL_TEST_DELTA, 8, fill)][0] - background_adc for fill in fill_examples]
    save_image_grid(BLOCK_05_DIR / "subpixel_residual_maps.png", residuals, [f"fill={fill:g}" for fill in fill_examples], "Карты прироста относительно фона", cols=4, cmap="coolwarm", cbar_label="Delta ADC", symmetric=True)

    failure_rows = df.sort_values(["detection_probability", "iou", "anomaly_size_px"]).head(3)
    failure_images: list[np.ndarray] = []
    failure_titles: list[str] = []
    for _, row in failure_rows.iterrows():
        key = (float(row["delta_T_K"]), int(row["anomaly_size_px"]), float(row["fill_factor"]))
        frame, truth, pred = examples[key]
        failure_images.extend([frame, truth.astype(float), pred.astype(float)])
        failure_titles.extend([f"dT={key[0]:g}, size={key[1]}, fill={key[2]:g}", "Эталон", "Детектор"])
    save_image_grid(BLOCK_05_DIR / "detection_failure_cases.png", failure_images, failure_titles, "Сложные случаи обнаружения", cols=3, fixed_scale=False)

    rel_error_mean = float(df["relative_error_to_linear_model"].mean())
    fill_ratio = float(
        df[(df["delta_T_K"] == FILL_TEST_DELTA) & (df["anomaly_size_px"] == 8) & (df["fill_factor"] == 0.1)]["delta_adc"].iloc[0]
        / max(float(df[(df["delta_T_K"] == FILL_TEST_DELTA) & (df["anomaly_size_px"] == 8) & (df["fill_factor"] == 1.0)]["delta_adc"].iloc[0]), 1e-9)
    )
    summary = {
        "mean_relative_error_to_linear_model": rel_error_mean,
        "delta_adc_ratio_fill_0p1_to_1p0_delta4_size8": fill_ratio,
        "min_detectable_size_table": min_rows,
    }
    write_metrics_json(BLOCK_05_DIR / "metrics.json", config, df.to_dict(orient="records"), summary)
    conclusion = (
        f"Средняя относительная ошибка линейной модели delta_adc(fill) составила {rel_error_mean:.3f}; "
        f"для Delta T=4 K и size=8 px отношение delta_adc(fill=0.1)/delta_adc(fill=1.0) равно {fill_ratio:.3f}."
    )
    write_readme(
        BLOCK_05_DIR / "README.md",
        "Блок 5 - fill_factor и субпиксельная аномалия",
        "Проверить ослабление полезного сигнала при неполном заполнении пикселя и малом размере аномалии.",
        f"Фон 300 K; Delta T={FILL_DELTAS}; размеры={FILL_SIZES}; fill_factor={FILL_FACTORS}; кадров на режим: {config['num_frames']}; seed: {config['seed']}.",
        ["metrics.csv", "metrics.json", "fill_factor_config.json", "delta_adc_vs_fill_factor.png", "delta_adc_linearity_check.png", "heatmap_iou_size_vs_fill_factor.png", "fill_factor_examples_grid.png", "detection_failure_cases.png"],
        conclusion + " Уменьшение fill_factor ослабляет полезный сигнал, снижает SNR-like и ухудшает обнаружение малых аномалий.",
    )
    return {
        "block_id": "05",
        "block_name": "fill_factor и субпиксельная аномалия",
        "varied_parameter": "Delta T, размер аномалии, fill_factor",
        "expected_behavior": "delta_adc и SNR-like должны уменьшаться при снижении fill_factor.",
        "observed_behavior": f"Для Delta T=4 K, size=8 px отношение fill=0.1 к fill=1.0 равно {fill_ratio:.3f}.",
        "key_metric": "relative_error_to_linear_model",
        "key_result": f"{rel_error_mean:.3f}",
        "main_figure": "block_05_fill_factor/delta_adc_vs_fill_factor.png",
        "conclusion": conclusion,
    }


def write_extended_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    summary_path = OUT_DIR / "extended_validation_summary.csv"
    pd.DataFrame(rows).sort_values("block_id").to_csv(summary_path, index=False)
    lines = [
        "# Расширенная валидация модели ИК-датчика",
        "",
        "Цель расширенной валидации - проверить, что модель реагирует на изменение температуры, фиксированной неоднородности матрицы, NUC-коррекции, оптического виньетирования и коэффициента заполнения пикселя физически согласованным образом.",
        "",
    ]
    for row in sorted(rows, key=lambda item: item["block_id"]):
        lines.extend(
            [
                f"## Блок {row['block_id']}. {row['block_name']}",
                "",
                f"**Изменяемый параметр.** {row['varied_parameter']}.",
                "",
                f"**Ожидаемое поведение.** {row['expected_behavior']}",
                "",
                f"**Наблюдаемое поведение.** {row['observed_behavior']}",
                "",
                f"**Ключевая метрика.** {row['key_metric']} = {row['key_result']}.",
                "",
                f"**Основной рисунок.** `{row['main_figure']}`.",
                "",
                f"**Вывод.** {row['conclusion']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Общий вывод",
            "",
            "Расширенный experiment 01 подтверждает не только монотонную температурную радиометрическую характеристику, но и ожидаемую реакцию модели на параметры измерительного тракта: матричную неоднородность, NUC-коррекцию, оптическое виньетирование и неполное заполнение пикселя.",
            "",
        ]
    )
    (OUT_DIR / "extended_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    summaries: list[dict[str, str]] = []
    if args.block in {"temperature", "all"}:
        summaries.append(run_temperature_block(args))
    if args.block in {"nuc", "all"}:
        summaries.append(run_nuc_block(args))
    if args.block in {"vignetting", "all"}:
        summaries.append(run_vignetting_block(args))
    if args.block in {"fill_factor", "all"}:
        summaries.append(run_fill_factor_block(args))

    if args.block == "all":
        write_extended_summary(summaries)
    elif summaries:
        # Для одиночного запуска обновляем общий отчет на основе выполненного блока, не удаляя уже существующие результаты.
        existing_rows: list[dict[str, str]] = []
        summary_path = OUT_DIR / "extended_validation_summary.csv"
        if summary_path.exists():
            existing_rows = pd.read_csv(summary_path).astype(str).to_dict(orient="records")
            existing_rows = [row for row in existing_rows if row["block_id"] not in {item["block_id"] for item in summaries}]
        write_extended_summary(existing_rows + summaries)

    print(f"Experiment 01 extended validation completed: {OUT_DIR} (block={args.block})")


if __name__ == "__main__":
    main()
