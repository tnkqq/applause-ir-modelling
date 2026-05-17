"""Common utilities for diploma experiment series 3.

The helpers in this module keep the experiments lightweight and reproducible.
They reuse the repository blackbody model for the temperature-to-IR-signal
step, then add controllable sensor effects that are sufficient for the
engineering experiments in results/3.1-3.8.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
MPL_CONFIG_DIR = RESULTS_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


sys.path.append(str(ROOT / "backend"))
sys.path.append(str(ROOT / "models"))

from Blackbody import Blackbody  # noqa: E402
import params  # noqa: E402


ADC_BITS = 10
ADC_MAX = float(2**ADC_BITS - 1)
PITCH_M = 17e-6
DEFAULT_FOV_RAD = math.pi / 6
DEFAULT_BG_K = 300.0
DEFAULT_CALIBRATION_K = (270.0, 380.0)
SIGNAL_LOW = 50.0
SIGNAL_HIGH = 950.0
NOISE_FLOOR = 1e-9

BLACKBODY_CACHE: dict[float, float] = {}


@dataclass(frozen=True)
class DetectionResult:
    mask: np.ndarray
    score: np.ndarray
    threshold: float


def experiment_dir(number: int, name: str, *, reset: bool = True) -> Path:
    out_dir = RESULTS_DIR / f"experiment_{number:02d}_{name}"
    preserve_existing = os.environ.get("SERIES3_PRESERVE_DIR") == "1"
    if reset and out_dir.exists() and not preserve_existing:
        shutil.rmtree(out_dir)
    for subdir in ["arrays", "images", "masks", "examples"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    df.to_csv(path, index=False)
    return df


def write_readme(path: Path, title: str, goal: str, params_text: str, outputs: list[str], conclusion: str) -> None:
    content = [
        f"# {title}",
        "",
        "## Цель",
        "",
        goal,
        "",
        "## Параметры",
        "",
        params_text,
        "",
        "## Выходные файлы",
        "",
    ]
    content.extend(f"- `{item}`" for item in outputs)
    content.extend(["", "## Краткий вывод", "", conclusion, ""])
    path.write_text("\n".join(content), encoding="utf-8")


def write_summary_json(
    out_dir: Path,
    *,
    number: int,
    title: str,
    varied_parameters: str,
    main_metrics: str,
    main_result: str,
    main_plot: str,
    conclusion: str,
) -> None:
    write_json(
        out_dir / "summary.json",
        {
            "experiment_number": number,
            "title": title,
            "varied_parameters": varied_parameters,
            "main_metrics": main_metrics,
            "main_result": main_result,
            "main_plot": main_plot,
            "conclusion": conclusion,
        },
    )


def token(value: Any) -> str:
    if isinstance(value, str):
        return value.replace(".", "p").replace("-", "m").replace(" ", "_")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def blackbody_power(temperature_k: float) -> float:
    key = float(temperature_k)
    if key not in BLACKBODY_CACHE:
        model = Blackbody(area=params.area, omega=params.omega, lambd=params.lambd)
        BLACKBODY_CACHE[key] = float(model.process(args=key)["P"])
    return BLACKBODY_CACHE[key]


def optics_factor(height: int, width: int, *, fov_rad: float = DEFAULT_FOV_RAD, pitch_m: float = PITCH_M) -> np.ndarray:
    focal = (pitch_m * width) / (2.0 * math.tan(fov_rad / 2.0))
    x = (np.arange(width) - (width - 1) / 2.0) * pitch_m
    y = (np.arange(height) - (height - 1) / 2.0) * pitch_m
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(focal**2 + xx**2 + yy**2)
    return (focal / distance) ** 4


def temperature_to_power_map(temp_map: np.ndarray) -> np.ndarray:
    temp_map = np.asarray(temp_map, dtype=float)
    power = np.zeros_like(temp_map, dtype=float)
    for temperature in np.unique(temp_map):
        power[temp_map == temperature] = blackbody_power(float(temperature))
    return power


def temperature_to_adc(
    temp_map: np.ndarray,
    *,
    apply_optics: bool = False,
    calibration_k: tuple[float, float] = DEFAULT_CALIBRATION_K,
    signal_low: float = SIGNAL_LOW,
    signal_high: float = SIGNAL_HIGH,
) -> np.ndarray:
    power = temperature_to_power_map(temp_map)
    if apply_optics:
        power = power * optics_factor(*temp_map.shape)

    p0 = blackbody_power(calibration_k[0])
    p1 = blackbody_power(calibration_k[1])
    gain = (signal_high - signal_low) / max(p1 - p0, NOISE_FLOOR)
    adc = signal_low + (power - p0) * gain
    return np.clip(adc, 0.0, ADC_MAX)


def weak_background(height: int, width: int, base_k: float = DEFAULT_BG_K, amplitude_k: float = 0.25) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height)[:, None]
    x = np.linspace(-1.0, 1.0, width)[None, :]
    return base_k + amplitude_k * (0.5 * x + 0.35 * y + 0.15 * np.sin(2 * np.pi * x))


def rectangle_mask(height: int, width: int, box_h: int, box_w: int, center: tuple[int, int] | None = None) -> np.ndarray:
    if center is None:
        center = (height // 2, width // 2)
    cy, cx = center
    r0 = max(0, int(round(cy - box_h / 2)))
    c0 = max(0, int(round(cx - box_w / 2)))
    r1 = min(height, r0 + int(box_h))
    c1 = min(width, c0 + int(box_w))
    mask = np.zeros((height, width), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def circle_mask(height: int, width: int, radius: float, center: tuple[int, int] | None = None) -> np.ndarray:
    if center is None:
        center = (height // 2, width // 2)
    yy, xx = np.indices((height, width))
    cy, cx = center
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def gaussian_hotspot(height: int, width: int, sigma: float, center: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if center is None:
        center = (height // 2, width // 2)
    yy, xx = np.indices((height, width))
    cy, cx = center
    weights = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    mask = weights >= 0.5
    return weights, mask


def scene_with_anomaly(
    height: int,
    width: int,
    *,
    background_k: float = DEFAULT_BG_K,
    delta_t: float = 4.0,
    shape: str = "circle",
    size: int = 12,
    fill_factor: float = 1.0,
    center: tuple[int, int] | None = None,
    weak_bg: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    scene = weak_background(height, width, background_k, 0.25) if weak_bg else np.full((height, width), background_k, dtype=float)
    effective_delta = float(delta_t) * float(fill_factor)
    if shape == "circle":
        mask = circle_mask(height, width, radius=max(1.0, size / 2), center=center)
        scene[mask] += effective_delta
    elif shape == "rectangle":
        mask = rectangle_mask(height, width, size, size, center=center)
        scene[mask] += effective_delta
    elif shape == "gaussian":
        weights, mask = gaussian_hotspot(height, width, sigma=max(1.0, size / 3), center=center)
        scene += effective_delta * weights
    else:
        raise ValueError(f"Unknown anomaly shape: {shape}")
    return scene, mask


def add_gaussian_noise(frame: np.ndarray, sigma_adc: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_adc <= 0:
        return frame.copy()
    return frame + rng.normal(0.0, sigma_adc, size=frame.shape)


def add_fpn(frame: np.ndarray, std_adc: float, rng: np.random.Generator, pattern: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if pattern is None:
        pattern = rng.normal(0.0, std_adc, size=frame.shape) if std_adc > 0 else np.zeros_like(frame)
    return frame + pattern, pattern


def apply_quantization(frame: np.ndarray, bits: int = ADC_BITS) -> np.ndarray:
    levels = float(2**int(bits) - 1)
    if levels <= 1:
        return np.zeros_like(frame)
    normalized = np.clip(frame / ADC_MAX, 0.0, 1.0)
    return np.round(normalized * levels) / levels * ADC_MAX


def inject_defects(frame: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = frame.copy()
    total = out.size
    count = int(round(total * max(rate, 0.0)))
    mask = np.zeros(total, dtype=bool)
    defect_type = np.zeros(total, dtype=int)
    if count > 0:
        selected = rng.choice(total, size=count, replace=False)
        hot = rng.random(count) >= 0.5
        flat = out.ravel()
        flat[selected[hot]] = ADC_MAX
        flat[selected[~hot]] = 0.0
        mask[selected] = True
        defect_type[selected[hot]] = 1
        defect_type[selected[~hot]] = -1
    return out, mask.reshape(out.shape), defect_type.reshape(out.shape)


def generate_adc_frame(
    temp_map: np.ndarray,
    rng: np.random.Generator,
    *,
    gaussian_sigma: float = 2.0,
    fpn_std: float = 0.0,
    quant_bits: int = ADC_BITS,
    defect_rate: float = 0.0,
    apply_optics: bool = False,
    fpn_pattern: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ideal = temperature_to_adc(temp_map, apply_optics=apply_optics)
    frame, pattern = add_fpn(ideal, fpn_std, rng, fpn_pattern)
    frame = add_gaussian_noise(frame, gaussian_sigma, rng)
    frame = apply_quantization(frame, quant_bits)
    frame, defect_mask, defect_type = inject_defects(frame, defect_rate, rng)
    frame = np.clip(frame, 0.0, ADC_MAX)
    return frame, {"ideal": ideal, "fpn_pattern": pattern, "defect_mask": defect_mask, "defect_type": defect_type}


def robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-9:
        sigma = float(np.std(arr))
    return max(sigma, 1e-9)


def remove_small_components(mask: np.ndarray, min_area: int = 4) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 1
    return cleaned.astype(bool)


def detect_global_threshold(frame: np.ndarray, *, k: float = 3.0, min_area: int = 4) -> DetectionResult:
    bg = float(np.median(frame))
    sigma = robust_sigma(frame)
    threshold = bg + k * sigma
    mask = remove_small_components(frame > threshold, min_area=min_area)
    return DetectionResult(mask=mask, score=frame.astype(float), threshold=threshold)


def detect_local_threshold(frame: np.ndarray, *, k: float = 2.5, sigma: float = 5.0, min_area: int = 4) -> DetectionResult:
    local = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = frame.astype(float) - local.astype(float)
    threshold = k * robust_sigma(residual)
    mask = remove_small_components(residual > threshold, min_area=min_area)
    return DetectionResult(mask=mask, score=residual, threshold=threshold)


def detect_otsu(frame: np.ndarray, *, offset: float = 0.0, min_area: int = 4) -> DetectionResult:
    img = normalize_uint8(frame)
    threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted = float(np.clip(threshold + offset, 0, 255))
    mask = remove_small_components(img > adjusted, min_area=min_area)
    return DetectionResult(mask=mask, score=img.astype(float), threshold=adjusted)


def detect_dog(frame: np.ndarray, *, k: float = 2.0, sigma_small: float = 1.0, sigma_large: float = 5.0, min_area: int = 4) -> DetectionResult:
    small = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigmaX=sigma_small, sigmaY=sigma_small)
    large = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigmaX=sigma_large, sigmaY=sigma_large)
    score = small.astype(float) - large.astype(float)
    threshold = k * robust_sigma(score)
    mask = remove_small_components(score > threshold, min_area=min_area)
    return DetectionResult(mask=mask, score=score, threshold=threshold)


def morphology_postprocess(mask: np.ndarray, kernel_size: int = 3, min_area: int = 4) -> np.ndarray:
    if kernel_size <= 1:
        return remove_small_components(mask, min_area=min_area)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return remove_small_components(out.astype(bool), min_area=min_area)


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not np.any(truth) else 0.0)
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": float(tpr),
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "iou": float(iou),
    }


def region_snr(frame: np.ndarray, mask: np.ndarray, *, noise_floor: float = 1.0) -> float:
    mask = mask.astype(bool)
    if not np.any(mask) or not np.any(~mask):
        return 0.0
    anomaly = frame[mask]
    background = frame[~mask]
    return float((np.mean(anomaly) - np.mean(background)) / max(float(np.std(background)), noise_floor))


def local_contrast(frame: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask) or not np.any(~mask):
        return 0.0
    return float(np.mean(frame[mask]) - np.mean(frame[~mask]))


def frame_stats(frame: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(frame)),
        "std": float(np.std(frame)),
        "min": float(np.min(frame)),
        "max": float(np.max(frame)),
        "dynamic_range": float(np.max(frame) - np.min(frame)),
    }


def normalize_uint8(frame: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    if math.isclose(vmax, vmin):
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def save_array_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, np.asarray(array), delimiter=",")


def save_gray_png(path: Path, array: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> None:
    cv2.imwrite(str(path), normalize_uint8(array, vmin=vmin, vmax=vmax))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_heatmap(path: Path, array: np.ndarray, title: str, cbar_label: str = "Код ADC") -> None:
    plt.figure(figsize=(6.2, 4.4))
    plt.imshow(array, cmap="inferno")
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_line_plot(
    path: Path,
    x: Iterable[Any],
    y: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    label: str | None = None,
) -> None:
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(list(x), list(y), marker="o", label=label)
    if label:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_multi_line_plot(
    path: Path,
    series: dict[str, tuple[Iterable[Any], Iterable[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for label, (x, y) in series.items():
        plt.plot(list(x), list(y), marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(7.0, 4.4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_heatmap_matrix(
    path: Path,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cbar_label: str,
) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.imshow(matrix, cmap="viridis", vmin=0, vmax=max(1.0, float(np.nanmax(matrix))))
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white" if matrix[y, x] > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_montage(path: Path, images: list[np.ndarray], titles: list[str], *, cmap: str = "inferno", cols: int = 3) -> None:
    rows = int(math.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 2.8))
    for idx, (image, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def apply_filter(frame: np.ndarray, filter_name: str, param: dict[str, Any]) -> np.ndarray:
    img = frame.astype(np.float32)
    if filter_name == "none":
        return frame.copy()
    if filter_name == "gaussian":
        return cv2.GaussianBlur(img, (int(param["ksize"]), int(param["ksize"])), float(param["sigma"]))
    if filter_name == "median":
        return cv2.medianBlur(img, int(param["ksize"]))
    if filter_name == "bilateral":
        return cv2.bilateralFilter(img, int(param["diameter"]), float(param["sigma_color"]), float(param["sigma_space"]))
    if filter_name == "nlm":
        u8 = normalize_uint8(img, vmin=0, vmax=ADC_MAX)
        denoised = cv2.fastNlMeansDenoising(u8, None, float(param["h"]), 7, 21)
        return denoised.astype(float) / 255.0 * ADC_MAX
    raise ValueError(f"Unknown filter: {filter_name}")


def moving_average_sequence(sequence: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return sequence.copy()
    out = np.zeros_like(sequence, dtype=float)
    for idx in range(sequence.shape[0]):
        start = max(0, idx - window + 1)
        out[idx] = np.mean(sequence[start : idx + 1], axis=0)
    return out


def inertia_sequence(sequence: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(sequence, dtype=float)
    out[0] = sequence[0]
    for idx in range(1, sequence.shape[0]):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * sequence[idx]
    return out


def auc_from_points(fpr: np.ndarray, tpr: np.ndarray) -> float:
    if len(fpr) < 2:
        return 0.0
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
