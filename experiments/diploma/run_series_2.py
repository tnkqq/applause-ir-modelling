#!/usr/bin/env python3
"""Run diploma experiment series 2.0-2.5.

The script keeps the existing physical stages where they are already usable
and uses a vectorized equivalent of the readout equations for experiment-scale
runs. All generated outputs are written under results/2.x.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
MPL_CONFIG_DIR = RESULTS_DIR / ".mplconfig"
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import k


sys.path.append(str(ROOT / "backend"))
sys.path.append(str(ROOT / "models"))

from Blackbody import Blackbody  # noqa: E402
from Bolometers import Bolometers  # noqa: E402
import params  # noqa: E402


ADC_BITS = 10
ADC_MAX = 2**ADC_BITS - 1
ADC_VREF = 3.2
ADC_OFFSET_V = 1.65
PITCH_M = 17e-6
FOV_MEDIUM = math.pi / 6
AREA_M2 = PITCH_M * PITCH_M * 0.65
OMEGA_MEDIUM = float(math.pi * (math.sin(FOV_MEDIUM / 2)) ** 2)
TCAM_K = 300.0
T_AMBIENT_K = 300.0
READOUT_ITERATIONS = 32


@dataclass(frozen=True)
class SensorGeometry:
    resolution: tuple[int, int]  # width, height
    size_boundary: tuple[int, int, int, int] = (2, 2, 2, 2)
    size_blind: tuple[int, int, int, int] = (1, 1, 1, 1)

    @property
    def active_shape(self) -> tuple[int, int]:
        return (self.resolution[1], self.resolution[0])


BLACKBODY_CACHE: dict[tuple[float, float], float] = {}


def reset_run_dir(run_id: str) -> Path:
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "arrays",
        "images",
        "config.json",
        "blackbody_power.csv",
        "optics_power_stats.csv",
        "adc_metrics.csv",
        "optics_profiles.csv",
        "optics_radial_profile.csv",
        "optics_metrics.csv",
        "fpn_metrics.csv",
        "seed_manifest.json",
        "defect_metrics.csv",
        "nuc_coefficients_full.csv",
        "nuc_metrics.csv",
        "anomaly_metrics.csv",
        "summary.md",
    ]:
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (run_dir / "arrays").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    return run_dir


def json_dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def token(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace(".", "p").replace("-", "m")
    if isinstance(value, int):
        return str(value)
    return f"{value:.0e}".replace("-", "m").replace("+", "").replace(".", "p")


def save_array_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",")


def save_image(path: Path, array: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> None:
    data = np.asarray(array, dtype=float)
    finite = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if vmin is None:
        vmin = float(np.min(finite))
    if vmax is None:
        vmax = float(np.max(finite))
    if math.isclose(vmax, vmin):
        scaled = np.zeros_like(finite, dtype=np.uint8)
    else:
        scaled = np.clip((finite - vmin) / (vmax - vmin), 0.0, 1.0)
        scaled = np.round(scaled * 255).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def heatmap(path: Path, array: np.ndarray, title: str, cbar_label: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.imshow(array, cmap="inferno")
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def line_plot(path: Path, xs, ys, title: str, xlabel: str, ylabel: str, *, label: str | None = None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", label=label)
    if label:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def blackbody_power(temperature_k: float, omega: float = OMEGA_MEDIUM) -> float:
    key = (float(temperature_k), float(omega))
    if key not in BLACKBODY_CACHE:
        model = Blackbody(area=AREA_M2, omega=omega)
        BLACKBODY_CACHE[key] = float(model.process(args=float(temperature_k))["P"])
    return BLACKBODY_CACHE[key]


def focal_length(resolution: tuple[int, int], fov_rad: float) -> float:
    return (PITCH_M * resolution[0]) / (2 * math.atan(fov_rad / 2))


def optics_factor(resolution: tuple[int, int], fov_rad: float) -> np.ndarray:
    width, height = resolution
    fl = focal_length(resolution, fov_rad)
    x = (np.arange(width) - width / 2 + 0.5) * PITCH_M
    y = (np.arange(height) - height / 2 + 0.5) * PITCH_M
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2 + fl**2)
    return (fl / distance) ** 4


def uniform_power_distribution(temperature_k: float, geometry: SensorGeometry, fov_rad: float = FOV_MEDIUM) -> np.ndarray:
    return blackbody_power(temperature_k) * optics_factor(geometry.resolution, fov_rad)


def scene_power_distribution(scene_k: np.ndarray, geometry: SensorGeometry, fov_rad: float = FOV_MEDIUM) -> np.ndarray:
    factor = optics_factor(geometry.resolution, fov_rad)
    out = np.zeros(scene_k.shape, dtype=float)
    for temperature in np.unique(scene_k):
        out[scene_k == temperature] = blackbody_power(float(temperature))
    return out * factor


def flat_power_distribution(temperature_k: float, geometry: SensorGeometry) -> np.ndarray:
    return np.full(geometry.active_shape, blackbody_power(temperature_k), dtype=float)


def bolometer_stage(
    p_active: np.ndarray,
    geometry: SensorGeometry,
    *,
    tolerance: float,
    seed: int,
    tcam_k: float = TCAM_K,
) -> dict[str, np.ndarray]:
    model = Bolometers(
        Tcam=tcam_k,
        size_active=geometry.resolution,
        size_boundary=geometry.size_boundary,
        size_blind=geometry.size_blind,
        area=AREA_M2,
        omega=OMEGA_MEDIUM,
        R_ambient_tol=tolerance,
        G_thermal_tol=tolerance,
        C_thermal_tol=tolerance,
        seed=seed,
        visualize=False,
    )
    return model.process({"P_distribution": p_active}, args=tcam_k)


def active_slices(geometry: SensorGeometry) -> tuple[slice, slice]:
    bt, bb, bl, br = geometry.size_boundary
    st, sb, sl, sr = geometry.size_blind
    row_slice = slice(bt + st, bt + st + geometry.resolution[1])
    col_slice = slice(bl + sl, bl + sl + geometry.resolution[0])
    return row_slice, col_slice


def vectorized_readout(
    bolometer_data: dict[str, np.ndarray],
    geometry: SensorGeometry,
    *,
    t_ambient_k: float = T_AMBIENT_K,
    t_int_s: float = params.t_int,
    iterations: int = READOUT_ITERATIONS,
) -> dict[str, np.ndarray]:
    q = bolometer_data["P_total"]
    r0 = bolometer_data["R0"]
    g = bolometer_data["G_thermal"]
    tau = bolometer_data["tau"]
    i_bias = params.I_bias
    e_act = params.E_act

    with np.errstate(over="ignore", invalid="ignore"):
        v_int = i_bias * r0 * np.exp(np.clip(e_act / (k * t_ambient_k), -80, 80))
        thermal_factor = 1 + (tau / t_int_s) * (np.exp(-t_int_s / tau) - 1)
        for _ in range(iterations):
            denom = k * t_ambient_k + k * ((i_bias * v_int + q) / g) * thermal_factor
            exponent = np.clip(e_act / denom, -80, 80)
            v_int = i_bias * r0 * np.exp(exponent)

        v_skim = i_bias * r0 * np.exp(np.clip(e_act / (k * t_ambient_k), -80, 80))
        skim_factor = 1 - np.exp(-t_int_s / tau)
        for _ in range(iterations):
            denom = k * (t_ambient_k + ((i_bias * v_skim) / g) * skim_factor)
            exponent = np.clip(e_act / denom, -80, 80)
            v_skim = i_bias * r0 * np.exp(exponent)

    bt, bb, bl, br = geometry.size_boundary
    st, sb, sl, sr = geometry.size_blind
    rows, cols = active_slices(geometry)
    active = v_int[rows, cols]

    top_rows = slice(0, st)
    bottom_rows = slice(v_int.shape[0] - sb, v_int.shape[0])
    left_cols = slice(0, sl)
    right_cols = slice(v_int.shape[1] - sr, v_int.shape[1])

    top_int = v_int[top_rows, cols]
    bottom_int = v_int[bottom_rows, cols]
    left_int = v_int[rows, left_cols]
    right_int = v_int[rows, right_cols]
    top_skim = v_skim[top_rows, cols]
    bottom_skim = v_skim[bottom_rows, cols]
    left_skim = v_skim[rows, left_cols]
    right_skim = v_skim[rows, right_cols]

    ref_v = np.mean(np.vstack([top_int, bottom_int]), axis=0)
    ref_h = np.mean(np.hstack([left_int, right_int]), axis=1)
    skim_v = np.mean(np.vstack([top_skim, bottom_skim]), axis=0)
    skim_h = np.mean(np.hstack([left_skim, right_skim]), axis=1)

    gain = 1 / (params.R1 * params.C)
    skim_gain = params.R3 / (params.R2 + params.R3)

    v_bol_v = gain * (skim_gain * ref_v[None, :] - active) + skim_gain * skim_v[None, :]
    v_bol_h = gain * (skim_gain * ref_h[:, None] - active) + skim_gain * skim_h[:, None]
    v_bol = gain * (skim_gain * ref_v[None, :] - active)
    return {"V_bol": v_bol, "V_bol_h": v_bol_h, "V_bol_v": v_bol_v, "V_int": v_int}


def adc_window(frames: list[np.ndarray], margin_fraction: float = 0.03) -> dict[str, float]:
    minimum = min(float(np.min(frame)) for frame in frames)
    maximum = max(float(np.max(frame)) for frame in frames)
    span = maximum - minimum
    if span <= 0:
        span = 1.0
    return {
        "analog_min_V": minimum - span * margin_fraction,
        "analog_max_V": maximum + span * margin_fraction,
        "margin_fraction": margin_fraction,
    }


def adc_quantize_window(voltage: np.ndarray, window: dict[str, float]) -> np.ndarray:
    lower = window["analog_min_V"]
    upper = window["analog_max_V"]
    span = upper - lower
    if span <= 0:
        return np.zeros_like(voltage, dtype=np.float64)
    normalized = np.clip((voltage - lower) / span, 0.0, 1.0)
    return np.round(normalized * ADC_MAX).astype(np.float64)


def simulate_frame(
    p_active: np.ndarray,
    geometry: SensorGeometry,
    *,
    tolerance: float,
    seed: int,
    adc_channel: str = "V_bol",
) -> dict[str, np.ndarray]:
    bol_data = bolometer_stage(p_active, geometry, tolerance=tolerance, seed=seed)
    readout = vectorized_readout(bol_data, geometry)
    return {
        "P_distribution": p_active,
        "P_total": bol_data["P_total"],
        "readout": readout[adc_channel],
    }


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    frame = np.asarray(frame, dtype=float)
    min_v = float(np.min(frame))
    max_v = float(np.max(frame))
    mean_v = float(np.mean(frame))
    std_v = float(np.std(frame))
    return {
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "dynamic_range": max_v - min_v,
        "saturated_fraction": float(np.mean(frame >= ADC_MAX)),
        "zero_fraction": float(np.mean(frame <= 0)),
    }


def center_edge_ratio(frame: np.ndarray) -> tuple[float, float, float]:
    h, w = frame.shape
    center = float(frame[h // 2, w // 2])
    edge = float(np.mean([frame[0, w // 2], frame[-1, w // 2], frame[h // 2, 0], frame[h // 2, -1]]))
    corner = float(np.mean([frame[0, 0], frame[0, -1], frame[-1, 0], frame[-1, -1]]))
    return center, edge, corner


def radial_profile(array: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    h, w = array.shape
    y, x = np.indices(array.shape)
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    bin_edges = np.linspace(0, radius.max(), bins + 1)
    values = []
    centers = []
    for i in range(bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if np.any(mask):
            values.append(float(np.mean(array[mask])))
            centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
    return np.array(centers), np.array(values)


def make_rect_scene(geometry: SensorGeometry, background_k: float, anomaly_k: float, size: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = geometry.active_shape
    scene = np.full((h, w), background_k, dtype=float)
    sizes = {
        "small": (max(4, h // 10), max(4, w // 10)),
        "medium": (max(8, h // 5), max(8, w // 5)),
        "large": (max(12, h // 3), max(12, w // 3)),
    }
    ah, aw = sizes[size]
    r0 = h // 2 - ah // 2
    c0 = w // 2 - aw // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + ah, c0 : c0 + aw] = True
    scene[mask] = anomaly_k
    return scene, mask


def robust_nuc_coefficients(frame0: np.ndarray, frame1: np.ndarray, target0: float, target1: float, fractional_bits: int | None) -> tuple[np.ndarray, np.ndarray]:
    denom = frame0 - frame1
    safe = np.abs(denom) > 1e-9
    a = np.ones_like(frame0, dtype=float)
    b = np.zeros_like(frame0, dtype=float)
    a[safe] = (target0 - target1) / denom[safe]
    b[safe] = (target1 * frame0[safe] - target0 * frame1[safe]) / denom[safe]
    a[~safe] = 0.0
    b[~safe] = (target0 + target1) / 2
    if fractional_bits is not None:
        step = 2 ** (-fractional_bits)
        a = np.round(a / step) * step
        b = np.round(b / step) * step
    return a, b


def apply_nuc(frame: np.ndarray, coef_a: np.ndarray, coef_b: np.ndarray) -> np.ndarray:
    return np.clip(frame * coef_a + coef_b, 0, ADC_MAX)


def run_2_0() -> None:
    run_dir = reset_run_dir("2.0")
    geometry = SensorGeometry((160, 120))
    temperatures = [280, 300, 320, 340, 360, 380, 400]
    tolerance = 1e-6
    seed = 2001
    config = {
        "run_id": "2.0",
        "pipeline": ["Blackbody", "Optics", "Bolometers", "VectorizedReadout", "CalibratedADC"],
        "resolution_width_height": list(geometry.resolution),
        "temperatures_K": temperatures,
        "tolerance": tolerance,
        "seed": seed,
        "adc_bits": ADC_BITS,
        "adc_calibration": "series-wide analog_min..analog_max window with 3% margin",
        "readout_note": "Vectorized fixed-point equivalent of Readout equations; avoids per-pixel fsolve runtime.",
    }

    blackbody_rows, optics_rows, adc_rows = [], [], []
    results_by_temperature = {}
    for temperature in temperatures:
        p_active = uniform_power_distribution(temperature, geometry)
        results_by_temperature[temperature] = simulate_frame(p_active, geometry, tolerance=tolerance, seed=seed)
    window = adc_window([result["readout"] for result in results_by_temperature.values()])
    config["adc_window"] = window
    json_dump(run_dir / "config.json", config)

    for temperature in temperatures:
        p_center = blackbody_power(temperature)
        p_active = results_by_temperature[temperature]["P_distribution"]
        result = results_by_temperature[temperature]
        readout = result["readout"]
        adc = adc_quantize_window(readout, window)

        tname = f"{temperature}K"
        save_array_csv(run_dir / "arrays" / f"p_distribution_{tname}.csv", p_active)
        save_array_csv(run_dir / "arrays" / f"p_total_{tname}.csv", result["P_total"])
        save_array_csv(run_dir / "arrays" / f"readout_frame_{tname}.csv", readout)
        save_array_csv(run_dir / "arrays" / f"adc_frame_{tname}.csv", adc)
        heatmap(run_dir / "images" / f"p_distribution_{tname}.png", p_active, f"P distribution {tname}", "W")
        heatmap(run_dir / "images" / f"adc_frame_{tname}.png", adc, f"ADC frame {tname}", "ADC code")

        p_c, p_e, p_corner = center_edge_ratio(p_active)
        a_c, a_e, a_corner = center_edge_ratio(adc)
        blackbody_rows.append({"temperature_K": temperature, "central_pixel_power_W": p_center})
        optics_rows.append(
            {
                "temperature_K": temperature,
                "p_min_W": float(np.min(p_active)),
                "p_max_W": float(np.max(p_active)),
                "p_mean_W": float(np.mean(p_active)),
                "p_std_W": float(np.std(p_active)),
                "p_center_W": p_c,
                "p_edge_W": p_e,
                "p_corner_W": p_corner,
                "p_center_edge_ratio": p_c / p_e,
                "p_center_corner_ratio": p_c / p_corner,
            }
        )
        metrics = frame_metrics(adc)
        metrics.update(
            {
                "temperature_K": temperature,
                "center_adc": a_c,
                "edge_adc": a_e,
                "corner_adc": a_corner,
                "center_edge_ratio": a_c / a_e if a_e else None,
                "center_corner_ratio": a_c / a_corner if a_corner else None,
                "readout_min_V": float(np.min(readout)),
                "readout_max_V": float(np.max(readout)),
                "readout_mean_V": float(np.mean(readout)),
            }
        )
        adc_rows.append(metrics)

    write_rows(run_dir / "blackbody_power.csv", blackbody_rows)
    write_rows(run_dir / "optics_power_stats.csv", optics_rows)
    write_rows(run_dir / "adc_metrics.csv", adc_rows)
    line_plot(
        run_dir / "images" / "mean_adc_vs_temperature.png",
        temperatures,
        [r["mean"] for r in adc_rows],
        "Mean ADC code vs temperature",
        "Temperature, K",
        "Mean ADC code",
    )
    summary = f"""# Запуск 2.0 — базовый полноразмерный прогон

Выполнен базовый прогон однородной сцены через цепочку `Blackbody -> Optics -> Bolometers -> VectorizedReadout -> ADC`.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- температуры: {temperatures};
- tolerance матрицы: {tolerance};
- seed: {seed};
- ADC: {ADC_BITS} бит, калиброванное окно `{window['analog_min_V']:.6f}..{window['analog_max_V']:.6f}` V.

Ключевые результаты:
- мощность центрального пикселя растет от `{blackbody_rows[0]['central_pixel_power_W']:.6e}` W до `{blackbody_rows[-1]['central_pixel_power_W']:.6e}` W;
- средний ADC-код растет от `{adc_rows[0]['mean']:.3f}` до `{adc_rows[-1]['mean']:.3f}`;
- отрицательные ADC-коды исключены за счет калиброванного analog gain/offset перед квантованием;
- доля насыщенных пикселей при максимальной температуре: `{adc_rows[-1]['saturated_fraction']:.6f}`;
- доля нулевых пикселей при минимальной температуре: `{adc_rows[0]['zero_fraction']:.6f}`.

Вывод: базовая цепочка дает физически интерпретируемую зависимость `температура -> мощность -> цифровой код` без аварий и без отрицательного цифрового диапазона.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_2_1() -> None:
    run_dir = reset_run_dir("2.1")
    geometry = SensorGeometry((160, 120))
    temperatures = [300, 340, 380, 420]
    fovs = {"narrow": math.pi / 9, "medium": math.pi / 6, "wide": math.pi / 4}
    json_dump(
        run_dir / "config.json",
        {
            "run_id": "2.1",
            "resolution_width_height": list(geometry.resolution),
            "temperatures_K": temperatures,
            "fov_cases_rad": fovs,
            "stage": "Blackbody + vectorized Optics",
        },
    )
    profile_rows, radial_rows, metric_rows = [], [], []
    for fov_name, fov in fovs.items():
        factor = optics_factor(geometry.resolution, fov)
        for temperature in temperatures:
            p_active = blackbody_power(temperature) * factor
            name = f"{temperature}K_{fov_name}"
            save_array_csv(run_dir / "arrays" / f"p_distribution_{name}.csv", p_active)
            heatmap(run_dir / "images" / f"p_distribution_heatmap_{name}.png", p_active, f"P distribution {name}", "W")
            center_row = p_active[p_active.shape[0] // 2, :]
            center_col = p_active[:, p_active.shape[1] // 2]
            line_plot(run_dir / "images" / f"profile_row_{name}.png", range(center_row.size), center_row, f"Central row {name}", "Column", "Power, W")
            line_plot(run_dir / "images" / f"profile_col_{name}.png", range(center_col.size), center_col, f"Central column {name}", "Row", "Power, W")
            radii, values = radial_profile(p_active)
            line_plot(run_dir / "images" / f"radial_profile_{name}.png", radii, values, f"Radial profile {name}", "Radius, pixels", "Mean power, W")

            for idx, value in enumerate(center_row):
                profile_rows.append({"temperature_K": temperature, "fov_case": fov_name, "axis": "row", "index": idx, "power_W": float(value)})
            for idx, value in enumerate(center_col):
                profile_rows.append({"temperature_K": temperature, "fov_case": fov_name, "axis": "column", "index": idx, "power_W": float(value)})
            for radius, value in zip(radii, values):
                radial_rows.append({"temperature_K": temperature, "fov_case": fov_name, "radius_px": float(radius), "mean_power_W": float(value)})

            p_center, p_edge, p_corner = center_edge_ratio(p_active)
            metric_rows.append(
                {
                    "temperature_K": temperature,
                    "fov_case": fov_name,
                    "fov_rad": fov,
                    "P_center": p_center,
                    "P_edge": p_edge,
                    "P_corner": p_corner,
                    "edge_drop_percent": (p_center - p_edge) / p_center * 100,
                    "corner_drop_percent": (p_center - p_corner) / p_center * 100,
                    "radial_nonuniformity_index": float(np.std(values / values[0])),
                }
            )
    write_rows(run_dir / "optics_profiles.csv", profile_rows)
    write_rows(run_dir / "optics_radial_profile.csv", radial_rows)
    write_rows(run_dir / "optics_metrics.csv", metric_rows)
    for temperature in temperatures:
        subset = [r for r in metric_rows if r["temperature_K"] == temperature]
        plt.figure(figsize=(7, 4))
        plt.bar([r["fov_case"] for r in subset], [r["corner_drop_percent"] for r in subset])
        plt.title(f"Corner drop vs FOV, {temperature} K")
        plt.xlabel("FOV case")
        plt.ylabel("Corner drop, %")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "images" / f"corner_drop_vs_fov_{temperature}K.png", dpi=160)
        plt.close()
    worst = max(metric_rows, key=lambda row: row["corner_drop_percent"])
    summary = f"""# Запуск 2.1 — влияние оптики

Исследован этап `Blackbody -> Optics` для трех FOV: narrow, medium, wide.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- температуры: {temperatures};
- FOV: {fovs}.

Ключевой результат:
- максимальное падение мощности в углу: `{worst['corner_drop_percent']:.3f}%` для `{worst['fov_case']}` FOV при `{worst['temperature_K']} K`;
- профили строки, столбца и радиальные профили сохранены в `images/`;
- численные профили сохранены в `optics_profiles.csv` и `optics_radial_profile.csv`.

Вывод: даже идеально однородная температурная сцена формирует неоднородное поле мощности на матрице из-за геометрии оптического тракта.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_2_2() -> None:
    run_dir = reset_run_dir("2.2")
    geometry = SensorGeometry((64, 48))
    temperatures = [300, 340, 380]
    tolerances = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    seeds = [101, 202, 303]
    config = {
        "run_id": "2.2",
        "resolution_width_height": list(geometry.resolution),
        "temperatures_K": temperatures,
        "tolerances": tolerances,
        "seeds": seeds,
        "optics": "disabled for flat-field FPN isolation",
        "adc_calibration": "series-wide analog_min..analog_max window with 3% margin",
    }
    json_dump(run_dir / "seed_manifest.json", {"seeds": seeds})
    rows = []
    generated = []
    for temperature in temperatures:
        p_active = flat_power_distribution(temperature, geometry)
        for tolerance in tolerances:
            for seed in seeds:
                result = simulate_frame(p_active, geometry, tolerance=tolerance, seed=seed)
                generated.append((temperature, tolerance, seed, result))
    window = adc_window([item[3]["readout"] for item in generated])
    config["adc_window"] = window
    json_dump(run_dir / "config.json", config)
    for temperature, tolerance, seed, result in generated:
                readout = result["readout"]
                adc = adc_quantize_window(readout, window)
                name = f"T{temperature}_tol_{token(tolerance)}_seed_{seed}"
                save_array_csv(run_dir / "arrays" / f"readout_frame_{name}.csv", readout)
                save_array_csv(run_dir / "arrays" / f"adc_frame_{name}.csv", adc)
                heatmap(run_dir / "images" / f"readout_frame_{name}.png", readout, f"Readout {name}", "V")
                heatmap(run_dir / "images" / f"adc_frame_{name}.png", adc, f"ADC {name}", "ADC code")
                row_means = np.mean(adc, axis=1)
                col_means = np.mean(adc, axis=0)
                mean_v = float(np.mean(adc))
                std_v = float(np.std(adc))
                rows.append(
                    {
                        "temperature_K": temperature,
                        "tolerance": tolerance,
                        "seed": seed,
                        "mean": mean_v,
                        "std": std_v,
                        "coefficient_of_variation": std_v / abs(mean_v) if mean_v else None,
                        "peak_to_peak": float(np.max(adc) - np.min(adc)),
                        "row_mean_std": float(np.std(row_means)),
                        "col_mean_std": float(np.std(col_means)),
                    }
                )
    write_rows(run_dir / "fpn_metrics.csv", rows)
    aggregate = []
    for tolerance in tolerances:
        subset = [r for r in rows if r["tolerance"] == tolerance]
        aggregate.append(
            {
                "tolerance": tolerance,
                "std_mean": float(np.mean([r["std"] for r in subset])),
                "peak_to_peak_mean": float(np.mean([r["peak_to_peak"] for r in subset])),
            }
        )
    line_plot(run_dir / "images" / "fpn_std_vs_tolerance.png", tolerances, [r["std_mean"] for r in aggregate], "FPN std vs tolerance", "Tolerance", "ADC std")
    line_plot(run_dir / "images" / "fpn_range_vs_tolerance.png", tolerances, [r["peak_to_peak_mean"] for r in aggregate], "FPN range vs tolerance", "Tolerance", "ADC peak-to-peak")
    worst = max(aggregate, key=lambda item: item["std_mean"])
    summary = f"""# Запуск 2.2 — FPN от разброса болометров

Проведена серия flat-field прогонов с отключенным оптическим градиентом, чтобы выделить вклад разброса параметров микроболометров.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- температуры: {temperatures};
- tolerances: {tolerances};
- seeds: {seeds}.

Ключевой результат:
- максимальное среднее STD по сериям: `{worst['std_mean']:.3f}` ADC code при tolerance `{worst['tolerance']}`;
- зависимости `std(tolerance)` и `peak_to_peak(tolerance)` сохранены в `images/`;
- все карты readout/ADC сохранены в `arrays/` и `images/`.

Вывод: рост tolerance физически превращается в рост fixed pattern noise, что прямо обосновывает необходимость NUC.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def inject_defects(frame: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    total = frame.size
    count = int(round(total * rate))
    flat_indices = rng.choice(total, size=count, replace=False) if count else np.array([], dtype=int)
    mask = np.zeros(total, dtype=bool)
    mask[flat_indices] = True
    defect_type = np.zeros(total, dtype=int)
    if count:
        hot = rng.random(count) >= 0.5
        selected = flat_indices
        out_flat = out.ravel()
        out_flat[selected[hot]] = ADC_MAX
        out_flat[selected[~hot]] = 0
        defect_type[selected[hot]] = 1
        defect_type[selected[~hot]] = -1
    return out, mask.reshape(frame.shape), defect_type.reshape(frame.shape)


def run_2_3() -> None:
    run_dir = reset_run_dir("2.3")
    geometry = SensorGeometry((96, 72))
    defect_rates = [0, 0.001, 0.005, 0.01, 0.02]
    seeds = [11, 22, 33, 44, 55]
    scene, anomaly_mask = make_rect_scene(geometry, 300, 360, "medium")
    p_active = scene_power_distribution(scene, geometry)
    base_result = simulate_frame(p_active, geometry, tolerance=1e-5, seed=2300)
    window = adc_window([base_result["readout"]])
    base = adc_quantize_window(base_result["readout"], window)
    json_dump(
        run_dir / "config.json",
        {
            "run_id": "2.3",
            "resolution_width_height": list(geometry.resolution),
            "background_K": 300,
            "anomaly_K": 360,
            "defect_rates": defect_rates,
            "seeds": seeds,
            "defect_types": ["hot", "cold"],
            "adc_calibration": "base scene analog_min..analog_max window with 3% margin",
            "adc_window": window,
        },
    )
    rows = []
    for rate in defect_rates:
        for seed in seeds:
            frame, mask, defect_type = inject_defects(base, rate, seed)
            name = f"rate_{token(rate)}_seed_{seed}"
            save_array_csv(run_dir / f"defect_map_{name}.csv", defect_type)
            save_array_csv(run_dir / "arrays" / f"adc_frame_defect_{name}.csv", frame)
            save_image(run_dir / "images" / f"defect_mask_{name}.png", mask.astype(float), vmin=0, vmax=1)
            heatmap(run_dir / "images" / f"adc_frame_defect_{name}.png", frame, f"ADC with defects {name}", "ADC code")
            mean_v = float(np.mean(frame))
            std_v = float(np.std(frame))
            outliers = int(np.sum((frame < mean_v - 3 * std_v) | (frame > mean_v + 3 * std_v)))
            anomaly_values = frame[anomaly_mask]
            background_values = frame[~anomaly_mask]
            rows.append(
                {
                    "defect_rate_requested": rate,
                    "seed": seed,
                    "defect_pixel_count": int(np.sum(mask)),
                    "defect_fraction_actual": float(np.mean(mask)),
                    "outlier_count_mean_pm_3std": outliers,
                    "mean": mean_v,
                    "std": std_v,
                    "saturated_like_pixels": int(np.sum(frame >= ADC_MAX)),
                    "zero_like_pixels": int(np.sum(frame <= 0)),
                    "mean_anomaly": float(np.mean(anomaly_values)),
                    "mean_background": float(np.mean(background_values)),
                    "anomaly_contrast": float(np.mean(anomaly_values) - np.mean(background_values)),
                    "defects_inside_anomaly": int(np.sum(mask & anomaly_mask)),
                }
            )
    write_rows(run_dir / "defect_metrics.csv", rows)
    aggregate = []
    for rate in defect_rates:
        subset = [r for r in rows if r["defect_rate_requested"] == rate]
        aggregate.append(
            {
                "rate": rate,
                "std_mean": float(np.mean([r["std"] for r in subset])),
                "outliers_mean": float(np.mean([r["outlier_count_mean_pm_3std"] for r in subset])),
            }
        )
    line_plot(run_dir / "images" / "defect_rate_vs_std.png", defect_rates, [r["std_mean"] for r in aggregate], "Defect rate vs STD", "Defect rate", "ADC std")
    line_plot(run_dir / "images" / "defect_rate_vs_outlier_count.png", defect_rates, [r["outliers_mean"] for r in aggregate], "Defect rate vs outlier count", "Defect rate", "Outlier count")
    summary = f"""# Запуск 2.3 — дефектные пиксели

Смоделирована сцена с фоном 300 K и прямоугольной аномалией 360 K. В готовый ADC-кадр вводились hot/cold дефекты.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- defect rates: {defect_rates};
- seeds: {seeds}.

Ключевой результат:
- при 0% дефектов средний STD: `{aggregate[0]['std_mean']:.3f}`;
- при 2% дефектов средний STD: `{aggregate[-1]['std_mean']:.3f}`;
- среднее число выбросов при 2% дефектов: `{aggregate[-1]['outliers_mean']:.3f}`.

Вывод: дефектные пиксели создают статистически измеримую деградацию и могут попадать внутрь области аномалии, искажая оценку контраста.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_2_4() -> None:
    run_dir = reset_run_dir("2.4")
    geometry = SensorGeometry((96, 72))
    tolerance = 3e-5
    seed = 2400
    calibration_temperatures = [300, 360]
    test_temperatures = [320, 340, 380]
    fractional_bits_options: list[int | None] = [3, 4, 5, 6, None]
    config = {
        "run_id": "2.4",
        "resolution_width_height": list(geometry.resolution),
        "calibration_temperatures_K": calibration_temperatures,
        "test_temperatures_K": test_temperatures,
        "fractional_bits": [3, 4, 5, 6, "full"],
        "tolerance": tolerance,
        "seed": seed,
        "adc_calibration": "calibration+test frames analog_min..analog_max window with 3% margin",
    }
    cal_readouts = []
    for temp in calibration_temperatures:
        p_active = uniform_power_distribution(temp, geometry)
        cal_readouts.append(simulate_frame(p_active, geometry, tolerance=tolerance, seed=seed)["readout"])
    test_readouts = {}
    for temp in test_temperatures:
        p_active = uniform_power_distribution(temp, geometry)
        test_readouts[temp] = simulate_frame(p_active, geometry, tolerance=tolerance, seed=seed)["readout"]
    window = adc_window(cal_readouts + list(test_readouts.values()))
    config["adc_window"] = window
    json_dump(run_dir / "config.json", config)
    cal_frames = [adc_quantize_window(readout, window) for readout in cal_readouts]
    target0, target1 = float(np.mean(cal_frames[0])), float(np.mean(cal_frames[1]))
    full_a, full_b = robust_nuc_coefficients(cal_frames[0], cal_frames[1], target0, target1, None)
    coeff_rows = []
    for r in range(full_a.shape[0]):
        for c in range(full_a.shape[1]):
            coeff_rows.append({"row": r, "col": c, "coef_a": float(full_a[r, c]), "coef_b": float(full_b[r, c])})
    write_rows(run_dir / "nuc_coefficients_full.csv", coeff_rows)

    test_frames = {}
    for temp, readout in test_readouts.items():
        frame = adc_quantize_window(readout, window)
        test_frames[temp] = frame
        save_array_csv(run_dir / "arrays" / f"adc_uncorrected_{temp}K.csv", frame)
        heatmap(run_dir / "images" / f"adc_uncorrected_{temp}K.png", frame, f"Uncorrected {temp} K", "ADC code")

    rows = []
    for bits in fractional_bits_options:
        label = "full" if bits is None else f"{bits}bit"
        coef_a, coef_b = robust_nuc_coefficients(cal_frames[0], cal_frames[1], target0, target1, bits)
        save_array_csv(run_dir / "arrays" / f"coef_a_{label}.csv", coef_a)
        save_array_csv(run_dir / "arrays" / f"coef_b_{label}.csv", coef_b)
        for temp, frame in test_frames.items():
            corrected = apply_nuc(frame, coef_a, coef_b)
            save_array_csv(run_dir / "arrays" / f"adc_corrected_{temp}K_{label}.csv", corrected)
            heatmap(run_dir / "images" / f"adc_corrected_{temp}K_{label}.png", corrected, f"Corrected {temp} K {label}", "ADC code")
            before_std = float(np.std(frame))
            after_std = float(np.std(corrected))
            center, edge, _ = center_edge_ratio(corrected)
            rows.append(
                {
                    "temperature_K": temp,
                    "fractional_bits": label,
                    "std_before": before_std,
                    "std_after": after_std,
                    "residual_nonuniformity": after_std / before_std if before_std else None,
                    "mean_absolute_correction_error": float(np.mean(np.abs(corrected - np.mean(corrected)))),
                    "center_edge_residual": center - edge,
                    "ring_artifact_score": float(np.std(radial_profile(corrected)[1])),
                }
            )
            radii, values = radial_profile(corrected)
            line_plot(run_dir / "images" / f"nuc_residual_profile_{temp}K_{label}.png", radii, values - np.mean(corrected), f"NUC residual profile {temp}K {label}", "Radius, pixels", "Residual ADC")
    write_rows(run_dir / "nuc_metrics.csv", rows)
    for temp in test_temperatures:
        subset = [r for r in rows if r["temperature_K"] == temp]
        plt.figure(figsize=(7, 4))
        plt.bar([r["fractional_bits"] for r in subset], [r["std_after"] for r in subset])
        plt.title(f"NUC std vs fractional bits, {temp} K")
        plt.xlabel("Coefficient precision")
        plt.ylabel("STD after correction")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "images" / f"nuc_std_vs_fractional_bits_{temp}K.png", dpi=160)
        plt.close()
    best = min(rows, key=lambda row: row["std_after"])
    summary = f"""# Запуск 2.4 — двухточечная NUC

Выполнена калибровка по двум однородным кадрам {calibration_temperatures} K и коррекция тестовых температур {test_temperatures}.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- tolerance: {tolerance};
- seed: {seed};
- точности коэффициентов: 3, 4, 5, 6 бит и full precision.

Ключевой результат:
- лучший `std_after`: `{best['std_after']:.3f}` ADC code для `{best['temperature_K']} K`, точность `{best['fractional_bits']}`;
- коэффициенты и corrected/uncorrected кадры сохранены в `arrays/`;
- графики зависимости STD от точности сохранены в `images/`.

Вывод: двухточечная NUC существенно снижает фиксированную неоднородность; влияние разрядности коэффициентов теперь измерено численно.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_2_5() -> None:
    run_dir = reset_run_dir("2.5")
    geometry = SensorGeometry((96, 72))
    background_k = 300
    anomaly_temperatures = [320, 340, 360, 400]
    sizes = ["small", "medium", "large"]
    tolerance_cases = {"low": 1e-6, "medium": 3e-5}
    seed = 2500
    snr_noise_floor_adc = 1.0
    config = {
        "run_id": "2.5",
        "resolution_width_height": list(geometry.resolution),
        "background_K": background_k,
        "anomaly_temperatures_K": anomaly_temperatures,
        "sizes": sizes,
        "tolerance_cases": tolerance_cases,
        "seed": seed,
        "nuc_calibration_K": [300, 360],
        "snr_noise_floor_adc": snr_noise_floor_adc,
        "adc_calibration": "per-tolerance calibration+all anomaly frames analog_min..analog_max window with 3% margin",
        "adc_windows": {},
    }
    nuc_coeffs = {}
    readout_sets = {}
    for tol_name, tolerance in tolerance_cases.items():
        cal0_readout = simulate_frame(uniform_power_distribution(300, geometry), geometry, tolerance=tolerance, seed=seed)["readout"]
        cal1_readout = simulate_frame(uniform_power_distribution(360, geometry), geometry, tolerance=tolerance, seed=seed)["readout"]
        scenario_readouts = {}
        for size in sizes:
            for anomaly_k in anomaly_temperatures:
                scene, mask = make_rect_scene(geometry, background_k, anomaly_k, size)
                p_active = scene_power_distribution(scene, geometry)
                scenario_readouts[(size, anomaly_k)] = (scene, mask, simulate_frame(p_active, geometry, tolerance=tolerance, seed=seed)["readout"])
        window = adc_window([cal0_readout, cal1_readout] + [item[2] for item in scenario_readouts.values()])
        config["adc_windows"][tol_name] = window
        cal0 = adc_quantize_window(cal0_readout, window)
        cal1 = adc_quantize_window(cal1_readout, window)
        nuc_coeffs[tol_name] = robust_nuc_coefficients(cal0, cal1, float(np.mean(cal0)), float(np.mean(cal1)), None)
        readout_sets[tol_name] = scenario_readouts
    json_dump(run_dir / "config.json", config)

    rows = []
    for tol_name, tolerance in tolerance_cases.items():
        coef_a, coef_b = nuc_coeffs[tol_name]
        window = config["adc_windows"][tol_name]
        for size in sizes:
            for anomaly_k in anomaly_temperatures:
                scene, mask, readout = readout_sets[tol_name][(size, anomaly_k)]
                raw = adc_quantize_window(readout, window)
                corrected = apply_nuc(raw, coef_a, coef_b)
                name = f"tol_{tol_name}_size_{size}_Ta_{anomaly_k}K"
                save_array_csv(run_dir / "arrays" / f"scene_map_{name}.csv", scene)
                save_array_csv(run_dir / "arrays" / f"adc_uncorrected_{name}.csv", raw)
                save_array_csv(run_dir / "arrays" / f"adc_corrected_{name}.csv", corrected)
                heatmap(run_dir / "images" / f"scene_map_{name}.png", scene, f"Scene {name}", "K")
                heatmap(run_dir / "images" / f"adc_uncorrected_{name}.png", raw, f"Uncorrected {name}", "ADC code")
                heatmap(run_dir / "images" / f"adc_corrected_{name}.png", corrected, f"Corrected {name}", "ADC code")
                for state, frame in [("uncorrected", raw), ("corrected", corrected)]:
                    anomaly_values = frame[mask]
                    background_values = frame[~mask]
                    mean_anomaly = float(np.mean(anomaly_values))
                    mean_background = float(np.mean(background_values))
                    contrast = mean_anomaly - mean_background
                    std_background = float(np.std(background_values))
                    effective_std_background = max(std_background, snr_noise_floor_adc)
                    threshold = mean_background + 3 * effective_std_background
                    detected_fraction = float(np.mean(anomaly_values > threshold))
                    rows.append(
                        {
                            "tolerance_case": tol_name,
                            "tolerance": tolerance,
                            "anomaly_size": size,
                            "anomaly_temperature_K": anomaly_k,
                            "state": state,
                            "mean_anomaly": mean_anomaly,
                            "mean_background": mean_background,
                            "contrast": contrast,
                            "contrast_ratio": mean_anomaly / mean_background if mean_background else None,
                            "std_background": std_background,
                            "effective_std_background": effective_std_background,
                            "snr_like": contrast / effective_std_background,
                            "detectable_area_fraction": detected_fraction,
                            "contrast_preserved_fraction": float(np.mean(anomaly_values > mean_background)),
                            "anomaly_pixel_count": int(np.sum(mask)),
                        }
                    )
    write_rows(run_dir / "anomaly_metrics.csv", rows)
    for tol_name in tolerance_cases:
        for state in ["uncorrected", "corrected"]:
            subset = [r for r in rows if r["tolerance_case"] == tol_name and r["state"] == state and r["anomaly_size"] == "medium"]
            plt.figure(figsize=(7, 4))
            for size in sizes:
                part = [r for r in rows if r["tolerance_case"] == tol_name and r["state"] == state and r["anomaly_size"] == size]
                plt.plot([r["anomaly_temperature_K"] for r in part], [r["contrast"] for r in part], marker="o", label=size)
            plt.title(f"Anomaly contrast vs temperature ({tol_name}, {state})")
            plt.xlabel("Anomaly temperature, K")
            plt.ylabel("Contrast, ADC code")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / "images" / f"anomaly_contrast_vs_temperature_{tol_name}_{state}.png", dpi=160)
            plt.close()
            line_plot(
                run_dir / "images" / f"anomaly_visibility_vs_size_{tol_name}_{state}.png",
                [r["anomaly_size"] for r in subset],
                [r["snr_like"] for r in subset],
                f"Visibility vs size ({tol_name}, {state}, medium temp sweep sample)",
                "Anomaly size",
                "SNR-like",
            )
    best = max(rows, key=lambda row: row["snr_like"] if row["snr_like"] is not None else -1e9)
    summary = f"""# Запуск 2.5 — температурная аномалия

Проведен интеграционный эксперимент с фоном {background_k} K и прямоугольной локальной аномалией.

Параметры:
- разрешение: {geometry.resolution[0]}x{geometry.resolution[1]};
- температуры аномалии: {anomaly_temperatures};
- размеры: {sizes};
- tolerance cases: {tolerance_cases};
- режимы: без коррекции и после NUC.
- для SNR-like используется шумовой пол {snr_noise_floor_adc} ADC code, чтобы идеально выровненный фон после NUC не давал бесконечные значения.

Ключевой результат:
- максимальная SNR-like оценка: `{best['snr_like']:.3f}`;
- достигнута для `{best['anomaly_temperature_K']} K`, размер `{best['anomaly_size']}`, tolerance `{best['tolerance_case']}`, состояние `{best['state']}`;
- все карты сцены, uncorrected и corrected кадры сохранены в `arrays/` и `images/`.

Вывод: эксперимент связывает физическую модель формирования ИК-кадра с прикладной задачей обнаружения температурной аномалии и дает численные зависимости контраста от температуры, размера и NUC.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    print("Running experiment 2.0")
    run_2_0()
    print("Running experiment 2.1")
    run_2_1()
    print("Running experiment 2.2")
    run_2_2()
    print("Running experiment 2.3")
    run_2_3()
    print("Running experiment 2.4")
    run_2_4()
    print("Running experiment 2.5")
    run_2_5()
    print("Done")


if __name__ == "__main__":
    main()
