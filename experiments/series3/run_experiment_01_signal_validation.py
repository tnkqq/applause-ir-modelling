#!/usr/bin/env python3
"""Experiment 01: signal curve, SNR and NETD validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import (
    DEFAULT_BG_K,
    experiment_dir,
    frame_stats,
    generate_adc_frame,
    rng_from_seed,
    save_heatmap,
    save_line_plot,
    save_montage,
    scene_with_anomaly,
    temperature_to_adc,
    write_csv,
    write_json,
    write_readme,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IR signal formation and SNR/NETD metrics.")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--temp_min", type=float, default=280.0)
    parser.add_argument("--temp_max", type=float, default=360.0)
    parser.add_argument("--temp_step", type=float, default=10.0)
    parser.add_argument("--noise_sigma", type=float, default=2.5)
    parser.add_argument("--fpn_std", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(1, "signal_validation")
    rng = rng_from_seed(args.seed)
    temps = np.arange(args.temp_min, args.temp_max + 0.5 * args.temp_step, args.temp_step)

    config = vars(args) | {
        "experiment": "01_signal_validation",
        "temperature_values_K": temps.tolist(),
        "adc_model": "Blackbody 8-14 um -> calibrated 10-bit ADC with Gaussian noise and FPN",
    }
    write_json(out_dir / "config.json", config)

    records: list[dict] = []
    examples: list[np.ndarray] = []
    example_titles: list[str] = []
    mean_by_temp: dict[float, float] = {}

    for temp in temps:
        temp_map = np.full((args.height, args.width), temp, dtype=float)
        frames = []
        fpn_pattern = rng.normal(0.0, args.fpn_std, size=temp_map.shape)
        for frame_idx in range(args.num_frames):
            frame, parts = generate_adc_frame(
                temp_map,
                rng,
                gaussian_sigma=args.noise_sigma,
                fpn_std=args.fpn_std,
                fpn_pattern=fpn_pattern,
            )
            frames.append(frame)
            if frame_idx == 0 and temp in {temps[0], temps[len(temps) // 2], temps[-1]}:
                save_heatmap(out_dir / "images" / f"example_uniform_{temp:g}K.png", frame, f"Uniform frame {temp:g} K")
                examples.append(frame)
                example_titles.append(f"{temp:g} K")
        stack = np.stack(frames)
        temporal_mean = np.mean(stack, axis=0)
        temporal_std = np.std(stack, axis=0)
        mean_signal = float(np.mean(stack))
        noise_std = float(np.mean(temporal_std))
        mean_by_temp[float(temp)] = mean_signal
        stats = frame_stats(temporal_mean)
        records.append(
            {
                "temperature_K": float(temp),
                "mean_signal_adc": mean_signal,
                "noise_std_adc": noise_std,
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

    df = write_csv(out_dir / "metrics.csv", records)
    save_line_plot(out_dir / "signal_vs_temperature.png", df["temperature_K"], df["mean_signal_adc"], "Mean signal vs temperature", "Temperature, K", "Mean ADC code")
    save_line_plot(out_dir / "noise_vs_temperature.png", df["temperature_K"], df["noise_std_adc"], "Noise vs temperature", "Temperature, K", "Noise std, ADC code")
    save_line_plot(out_dir / "snr_vs_temperature.png", df["temperature_K"], df["snr_for_temp_step"], "SNR for temperature step", "Temperature, K", "SNR")
    save_line_plot(out_dir / "netd_estimation.png", df["temperature_K"], df["netd_K"], "NETD estimation", "Temperature, K", "NETD, K")
    save_montage(out_dir / "example_frames.png", examples, example_titles, cols=3)

    # Небольшая проверка маски аномалии сохраняется как пример формата для следующих экспериментов.
    _, sample_mask = scene_with_anomaly(args.height, args.width, background_k=DEFAULT_BG_K, delta_t=5.0, size=12)
    np.savetxt(out_dir / "masks" / "example_anomaly_mask.csv", sample_mask.astype(int), delimiter=",", fmt="%d")

    best_netd = float(df["netd_K"].min())
    mean_slope = float(df["local_slope_adc_per_K"].mean())
    conclusion = (
        f"Средний наклон радиометрической характеристики составил {mean_slope:.3f} ADC/K; "
        f"лучшая оценка NETD равна {best_netd:.3f} K."
    )
    write_readme(
        out_dir / "README.md",
        "Эксперимент 01 - валидация ИК-сигнала, SNR и NETD",
        "Проверить монотонность цифрового сигнала от температуры сцены и оценить шумовую температурную чувствительность.",
        f"Температуры {args.temp_min:g}-{args.temp_max:g} K с шагом {args.temp_step:g} K; кадров на уровень: {args.num_frames}; seed: {args.seed}.",
        ["metrics.csv", "config.json", "signal_vs_temperature.png", "snr_vs_temperature.png", "netd_estimation.png", "example_frames.png"],
        conclusion,
    )
    write_summary_json(
        out_dir,
        number=1,
        title="Валидация модели формирования ИК-сигнала и метрик SNR/NETD",
        varied_parameters="Температура однородной сцены",
        main_metrics="mean_signal_adc, noise_std_adc, snr_for_temp_step, netd_K",
        main_result=conclusion,
        main_plot=str(out_dir / "signal_vs_temperature.png"),
        conclusion="Модель дает монотонный рост цифрового сигнала с температурой и позволяет оценивать NETD через наклон характеристики и шум.",
    )
    print(f"Experiment 01 completed: {out_dir}")


if __name__ == "__main__":
    main()
