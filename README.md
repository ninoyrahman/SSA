# Space Situational Awareness (SSA)

Python utilities for orbital mechanics calculations and regression checks based on examples from Vallado's *Fundamentals of Astrodynamics and Applications*.

## Modules

- `util.py` - canonical/SI unit conversions, elliptic/parabolic/hyperbolic Kepler solvers, anomaly conversions, state-vector/orbital-element conversions, two-body propagation, and time-of-flight calculations.
- `util_maneuvers.py` - transfer orbits, plane changes, phasing, low-thrust transfers, and relative motion in Hill coordinates.
- `util_transformation.py` - vector transformations between orbital reference frames.
- `test_util.py` and `test_maneuvers.py` - regression checks for orbital utilities and maneuver calculations.
- `ssa_gui.py` - a Tkinter interface for choosing a test suite (Util or Maneuvers) and running one test or the whole suite.

## Units and conventions

Angles are in radians unless stated otherwise. Keep distance and time units consistent with the gravitational parameter `mu`; for example, use kilometers, seconds, and km^3/s^2 together. Canonical unit conversions use Earth scales by default and can be configured with a radius and gravitational parameter.

## Run the checks

Install `pytest` if it is not already available, then run both suites from the project directory:

```bash
python -m pip install pytest
python -m pytest test_util.py test_maneuvers.py
```

To open the interactive test runner:

```bash
python ssa_gui.py
```

Select **Util** or **Maneuvers** in the first dropdown, then choose an individual test or **All tests** in the second.
