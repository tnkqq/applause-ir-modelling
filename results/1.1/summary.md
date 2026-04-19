# Run 1.1: Blackbody + Optics

Goal: run the first two active pipeline stages and inspect the optical distribution over a small sensor matrix.

Pipeline: `Blackbody -> Optics`.

Parameters:
- temperatures: [300, 350, 400] K
- resolution: 64x48
- array shape: 48 rows x 64 cols
- FOV: 0.523599 rad
- pitch: 1.70e-05 m
- focal length: 2.124562e-03 m

Results:
- 300 K center power: 2.171514293252e-09 W
- 300 K corner power: 1.798386521788e-09 W
- 300 K corner/center ratio: 0.828172
- 400 K center power: 6.994223899892e-09 W
- 400 K corner power: 5.792417775475e-09 W
- 400 K corner/center ratio: 0.828172

Files:
- arrays/p_distribution_*K.npy
- arrays/p_distribution_*K.csv
- figures/p_distribution_*K.png
- figures/power_distribution_summary.png
- optics_metrics.csv
- parameters.json
- metrics.json
- stdout.txt

Interpretation: the optical stage preserves the temperature dependence from Blackbody and adds spatial non-uniformity across the frame. The corner power is lower than the center power because of the field-position factor.
