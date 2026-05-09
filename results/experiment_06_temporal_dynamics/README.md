# Эксперимент 06 - временная инерционность и усреднение кадров

## Цель

Показать компромисс между ростом SNR при временной фильтрации и задержкой обнаружения динамической аномалии.

## Параметры

Сценарии: ['static', 'appearing', 'moving']; alpha=[0.0, 0.3, 0.6, 0.9]; windows=[1, 3, 5, 10]; длина=80; seed: 3606.

## Выходные файлы

- `metrics.csv`
- `amplitude_over_time.csv`
- `snr_vs_window_size.png`
- `detection_delay_vs_alpha.png`
- `tpr_fpr_vs_alpha.png`
- `anomaly_amplitude_over_time.png`
- `example_sequence_frames.png`

## Краткий вывод

Кадровое усреднение повышало средний SNR до 17.607, но высокая инерционность увеличивала задержку до 2.56 кадров.
