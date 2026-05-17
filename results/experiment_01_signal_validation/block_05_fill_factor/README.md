# Блок 5 - fill_factor и субпиксельная аномалия

## Цель

Проверить ослабление полезного сигнала при неполном заполнении пикселя и малом размере аномалии.

## Параметры

Фон 300 K; Delta T=[2.0, 4.0, 8.0]; размеры=[1, 2, 4, 8, 16]; fill_factor=[1.0, 0.75, 0.5, 0.25, 0.1]; кадров на режим: 12; seed: 3602.

## Выходные файлы

- `metrics.csv`
- `metrics.json`
- `fill_factor_config.json`
- `delta_adc_vs_fill_factor.png`
- `delta_adc_linearity_check.png`
- `heatmap_iou_size_vs_fill_factor.png`
- `fill_factor_examples_grid.png`
- `detection_failure_cases.png`

## Краткий вывод

Средняя относительная ошибка линейной модели delta_adc(fill) составила 0.041; для Delta T=4 K и size=8 px отношение delta_adc(fill=0.1)/delta_adc(fill=1.0) равно 0.102. Уменьшение fill_factor ослабляет полезный сигнал, снижает SNR-like и ухудшает обнаружение малых аномалий.
