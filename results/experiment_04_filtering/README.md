# Эксперимент 04 - эффективность фильтрации ИК-изображений

## Цель

Проверить, повышает ли предварительная фильтрация качество обнаружения аномалий на зашумленных кадрах.

## Параметры

Единый набор из 24 кадров; фильтры: none, Gaussian, median, bilateral, non-local means; seed: 3404.

## Выходные файлы

- `metrics.csv`
- `filter_comparison.csv`
- `snr_improvement_by_filter.png`
- `iou_by_filter.png`
- `tpr_fpr_by_filter.png`
- `filter_examples.png`

## Краткий вывод

Наиболее устойчивый вариант по среднему IoU: `median_k3` с IoU=0.965 и SNR improvement=10.171.
