#!/usr/bin/env python3
"""Experiment 08: aggregate reports for experiment_* folders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.series3.common import RESULTS_DIR, experiment_dir, write_json, write_readme


GOALS = {
    1: "Проверить базовую радиометрическую зависимость цифрового сигнала от температуры сцены и оценить SNR/NETD.",
    2: "Определить минимальный температурный контраст, при котором аномалия начинает стабильно выделяться простым пороговым алгоритмом.",
    3: "Показать, как разные шумовые составляющие ИК-датчика ухудшают качество обнаружения температурной аномалии.",
    4: "Оценить, улучшают ли простые методы предварительной фильтрации качество сегментации аномалии.",
    5: "Проверить влияние размера аномалии и коэффициента заполнения пикселя на обнаружимость.",
    6: "Показать влияние временной инерционности датчика и кадрового усреднения на SNR, задержку и качество обнаружения.",
    7: "Сравнить несколько интерпретируемых алгоритмов обнаружения и выбрать базовый вариант для дипломной работы.",
}

MODELED = {
    1: "Серия однородных ИК-кадров без аномалий при разных температурах сцены и разных реализациях шума.",
    2: "Кадры с однородным фоном и одной локальной аномалией разной формы при изменяемом температурном контрасте.",
    3: "Одна фиксированная температурная аномалия на фоне при добавлении Gaussian noise, FPN, квантования, дефектных пикселей и комбинированного шума.",
    4: "Одинаковый набор зашумленных ИК-кадров, обработанный различными фильтрами перед применением одного детектора.",
    5: "Кадры с прямоугольной аномалией разного пиксельного размера и разного fill_factor при фиксированном температурном контрасте.",
    6: "Последовательности кадров со статической, постепенно появляющейся и движущейся аномалией с моделью инерционности первого порядка.",
    7: "Общий тестовый набор кадров с разными dT, размерами, положениями, шумами и кадрами без аномалии.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary report for series 3 experiments.")
    parser.add_argument("--include_pattern", default="experiment_0[1-7]_*")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = experiment_dir(8, "summary_report")
    figures_dir = RESULTS_DIR / "figures_for_diploma"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = sorted(RESULTS_DIR.glob(args.include_pattern))
    rows = []
    report_lines = [
        "# Итоговый отчет по экспериментам серии 3",
        "",
        "Отчет сформирован автоматически на основе сохраненных `summary.json`, `config.json` и CSV-файлов из папок `results/experiment_*/`. Результаты не дописывались вручную.",
        "",
    ]

    for exp_dir in experiment_dirs:
        summary_path = exp_dir / "summary.json"
        config_path = exp_dir / "config.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        config = load_json(config_path) if config_path.exists() else {}
        number = summary["experiment_number"]
        title = summary["title"]
        main_plot = Path(summary["main_plot"])
        copied_plot = ""
        if main_plot.exists():
            copied = figures_dir / f"experiment_{number:02d}_{main_plot.name}"
            shutil.copy2(main_plot, copied)
            copied_plot = str(copied)

        rows.append(
            {
                "experiment_number": number,
                "title": title,
                "varied_parameters": summary["varied_parameters"],
                "main_metrics": summary["main_metrics"],
                "main_result": summary["main_result"],
                "main_plot": copied_plot or str(main_plot),
                "conclusion": summary["conclusion"],
            }
        )

        csv_files = sorted(p.name for p in exp_dir.glob("*.csv"))
        png_files = sorted(p.name for p in exp_dir.glob("*.png"))
        report_lines.extend(
            [
                f"## Эксперимент {number:02d}. {title}",
                "",
                f"**Цель.** {GOALS.get(number, summary['conclusion'])}",
                "",
                f"**Что моделировалось.** {MODELED.get(number, title)} "
                f"Экспериментальная папка `{exp_dir.name}` содержит воспроизводимый запуск с параметрами из `config.json`.",
                "",
                f"**Изменяемые параметры.** {summary['varied_parameters']}.",
                "",
                f"**Метрики.** {summary['main_metrics']}.",
                "",
                f"**Главный численный результат.** {summary['main_result']}",
                "",
                f"**Графики.** Основной график: `{copied_plot or main_plot}`. Дополнительные PNG в папке: {', '.join(png_files) if png_files else 'нет'}.",
                "",
                f"**Данные.** CSV-файлы: {', '.join(csv_files) if csv_files else 'нет'}.",
                "",
                f"**Вывод для диплома.** {summary['conclusion']}",
                "",
            ]
        )

    df = pd.DataFrame(rows).sort_values("experiment_number")
    summary_csv = RESULTS_DIR / "summary_all_experiments.csv"
    df.to_csv(summary_csv, index=False)
    summary_md = RESULTS_DIR / "summary_report.md"
    summary_md.write_text("\n".join(report_lines), encoding="utf-8")
    df.to_csv(out_dir / "summary_all_experiments.csv", index=False)
    shutil.copy2(summary_md, out_dir / "summary_report.md")
    write_json(
        out_dir / "config.json",
        {
            "experiment": "08_summary_report",
            "include_pattern": args.include_pattern,
            "experiments_found": [p.name for p in experiment_dirs],
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "figures_dir": str(figures_dir),
        },
    )
    write_readme(
        out_dir / "README.md",
        "Эксперимент 08 - итоговый отчет по серии 3",
        "Собрать общую таблицу, Markdown-отчет и папку ключевых графиков для диплома.",
        f"Обработаны папки: {', '.join(p.name for p in experiment_dirs)}.",
        ["../summary_all_experiments.csv", "../summary_report.md", "../figures_for_diploma/"],
        f"Собрано {len(df)} экспериментов в единую таблицу и Markdown-отчет.",
    )
    print(f"Experiment 08 summary completed: {summary_md}")


if __name__ == "__main__":
    main()
