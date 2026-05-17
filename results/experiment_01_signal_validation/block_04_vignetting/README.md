# Блок 4 - оптическое виньетирование и распределение мощности

## Цель

Проверить радиальный спад сигнала при равномерной температуре сцены из-за cos^4-виньетирования.

## Параметры

Температура 320 K; режимы none/weak/medium/strong; дополнительно сохранены пример с шумом и flat-field компенсацией.

## Выходные файлы

- `metrics.csv`
- `metrics.json`
- `vignetting_config.json`
- `vignetting_radial_profile.png`
- `center_to_corner_drop_vs_strength.png`
- `vignetting_comparison_grid_320K.png`
- `vignetting_residual_maps.png`

## Краткий вывод

В режиме strong падение центр-угол составило 133.285 ADC (56.48%), STD кадра 39.371 ADC. Модель воспроизводит систематический радиальный градиент, который необходимо учитывать или компенсировать.
