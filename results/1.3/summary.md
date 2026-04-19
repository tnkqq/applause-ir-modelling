# Run 1.3: Small full chain to ADC

Goal: run the existing physical pipeline through readout and ADC on a tiny matrix, avoiding the expensive full 640x480 demonstration run.

Pipeline: `Blackbody -> Optics -> Bolometers -> Readout -> ADC`.

Parameters:
- temperatures: [300, 320] K
- active resolution: 8x6
- boundary pixels: (2, 2, 2, 2)
- blind pixels: (1, 1, 1, 1)
- bolometer seed: 123
- ADC channel: raw `V_bol`, no horizontal/vertical skimming
- NumPy: 1.26.4
- SciPy: 1.17.1

Results:
- ADC frame shape at 300 K: (6, 8)
- mean ADC at 300 K: -184.937500
- std ADC at 300 K: 108.455906
- mean ADC at 320 K: -149.229167
- std ADC at 320 K: 108.820617
- mean delta 320 K - 300 K: 35.708333
- captured stdout lines: 584

Files:
- arrays/adc_*K.npy
- arrays/adc_*K.csv
- figures/adc_*K.png
- figures/mean_adc_by_temperature.png
- adc_metrics.csv
- parameters.json
- metrics.json
- stdout.txt

Interpretation: the full chain executed successfully at small resolution after using NumPy 1.26.4. This verifies that the active model can produce digital ADC frames when the raw readout channel is used. The captured stdout is large relative to the tiny experiment because `Readout.process()` prints debug information inside pixel loops.
