# Space Situational Awareness (SSA)

Python utilities for orbital mechanics calculations and regression checks based on examples from Vallado's *Fundamentals of Astrodynamics and Applications*.

## Modules

- `util.py` - canonical/SI unit conversions, elliptic/parabolic/hyperbolic Kepler solvers, anomaly conversions, state-vector/orbital-element conversions, two-body propagation, and time-of-flight calculations.
- `util_maneuvers.py` - transfer orbits, plane changes, phasing, low-thrust transfers, and relative motion in Hill coordinates.
- `util_propagate.py` - sampled two-body orbital propagation.
- `plotter.py` - orbit plotting and 3D locus animation helpers.
- `util_transformation.py` - vector transformations between orbital reference frames.
- `test_util.py`, `test_maneuvers.py`, and `test_propagate.py` - regression checks for orbital utilities, maneuver calculations, and propagation/animation examples.
- `ssa_gui.py` - a Tkinter interface for choosing the Util, Maneuvers, or Propagate test suite and running one test or the whole suite.

## Units and conventions

Angles are in radians unless stated otherwise. Keep distance and time units consistent with the gravitational parameter `mu`; for example, use kilometers, seconds, and km^3/s^2 together. Canonical unit conversions use Earth scales by default and can be configured with a radius and gravitational parameter.

## Run the checks

Install `pytest` if it is not already available, then run both suites from the project directory:

```bash
python -m pip install pytest
python -m pytest test_util.py test_maneuvers.py test_propagate.py
```

To open the interactive test runner:

```bash
python ssa_gui.py
```

Select **Util**, **Maneuvers**, or **Propagate** in the first dropdown, then choose an individual test or **All tests** in the second. The Propagate suite includes orbital propagation and 3D animation examples.
