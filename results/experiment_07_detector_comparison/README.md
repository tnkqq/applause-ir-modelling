# Эксперимент 07 - сравнение алгоритмов обнаружения

## Цель

Сравнить простые интерпретируемые алгоритмы обнаружения температурных аномалий на общем синтетическом наборе.

## Параметры

Кадров: 82; алгоритмы: global threshold, adaptive local, Otsu, DoG, morphology; seed: 3707.

## Выходные файлы

- `metrics.csv`
- `algorithm_summary.csv`
- `roc_curves.png`
- `f1_by_algorithm.png`
- `iou_by_algorithm.png`
- `runtime_by_algorithm.png`
- `success_failure_examples.png`

## Краткий вывод

Рекомендуемый базовый алгоритм: `global_threshold` с параметром `k=2.5` (F1=0.872, IoU=0.964).
