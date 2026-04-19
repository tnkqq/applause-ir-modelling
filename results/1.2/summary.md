# Run 1.2: Small full chain to ADC failed

Goal: run `Blackbody -> Optics -> Bolometers -> Readout -> ADC` on a tiny 8x6 active matrix.

Status: failed during `Readout.process()`.

Error:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
ValueError: setting an array element with a sequence.
```

Cause observed: `scipy.optimize.fsolve()` returns a one-element array, while `Readout.py` assigns that array directly into a scalar NumPy cell, for example `V_int[r,c] = fsolve(...)`. With the installed modern NumPy this is an error.

Engineering note: this is valuable as a compatibility finding. The readout model likely expects older NumPy behavior or needs a small code fix such as extracting the scalar result from `fsolve(...)[0]`.
