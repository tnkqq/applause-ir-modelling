# Эксперимент 03 - влияние типа и уровня шума

## Цель

Оценить, как разные шумовые составляющие ИК-датчика ухудшают обнаружение температурной аномалии.

## Параметры

Типы шума: gaussian, fpn, quantization, defects, combined; уровни: [0, 1, 2, 4, 6, 8]; seed: 3303. Gaussian - временный шум электроники; FPN - фиксированная неоднородность матрицы; quantization - дискретизация АЦП; defects - hot/cold пиксели; combined - совместное действие факторов.

## Выходные файлы

- `metrics.csv`
- `noise_type_comparison.csv`
- `snr_vs_noise_level.png`
- `tpr_fpr_vs_noise_level.png`
- `iou_vs_noise_level.png`
- `example_noise_types.png`

## Краткий вывод

Наиболее сильное среднее ухудшение IoU дал шум `combined`: средний IoU=0.497.
