# Run 1.0: Blackbody temperature sweep

Goal: check the first physical stage of the model and estimate how the incident IR power changes with blackbody temperature.

Parameters:
- temperature range: 280..420 K, step 20 K
- wavelength range: 8.00e-06..1.40e-05 m
- receiver area: 2.890000e-10 m^2
- projected solid angle: 0.75

Results:
- samples: 8
- min power: 8.602584900394e-09 W
- max power: 4.555746762243e-08 W
- max/min ratio: 5.295788

Files:
- blackbody_power.csv
- parameters.json
- metrics.json
- power_vs_temperature.png
- stdout.txt

Interpretation: the power grows monotonically with temperature, as expected from thermal radiation in the 8..14 um band.
