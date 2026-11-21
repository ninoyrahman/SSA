# Space Situational Awareness (SSA)

Python utilities for orbital mechanics, spacecraft maneuvers, two-body propagation, and visualizing example trajectories. Numerical examples and regression checks are based on Vallado's *Fundamentals of Astrodynamics and Applications*.

## Features

- Convert between Earth canonical units and SI units.
- Solve elliptic, parabolic, and hyperbolic Kepler equations; convert anomalies; convert between Cartesian state vectors and classical orbital elements; propagate two-body states; and calculate time of flight.
- Calculate Hohmann, bi-elliptic, and one-tangent transfers; plane changes including minimum-cost combined plane changes; circular-orbit phasing; low-thrust transfers; and relative motion in Hill coordinates.
- Sample two-body trajectories with optional orbital orientation (inclination, RAAN, and argument of periapsis).
- Plot and animate 3D trajectories, including an equatorial reference curve, spacecraft speed, and optional GIF export.
- Explore interactive propagation examples for inclined orbits, combined inclination and RAAN changes, Hohmann and bi-elliptic transfers, and minimum-cost plane changes.
- Run the Util, Maneuvers, and Propagate example suites from a Tkinter test runner, or preview an existing GIF in the runner.

## Project modules

- `util.py` — unit conversions, Kepler solvers, anomaly conversions, state-vector/orbital-element conversions, propagation, and time-of-flight calculations.
- `util_maneuvers.py` — orbital transfer, plane change, phasing, low-thrust, arc-length, and Hill relative-motion calculations.
- `util_transformation.py` — transformations among orbital reference frames.
- `util_propagate.py` — sampled two-body orbital propagation with optional orbital orientation.
- `plotter.py` — 3D trajectory animation and GIF saving.
- `test_util.py`, `test_maneuvers.py`, `test_propagate.py` — numerical examples and regression checks for the utilities, maneuvers, propagation, and animation.
- `ssa_gui.py` — Tkinter test runner with suite/test selection, menu actions, output display, and GIF preview.

## Requirements

Python 3, NumPy, SciPy, Matplotlib, Pillow, and Tkinter are used by the project. Install the Python packages with:

```bash
python -m pip install numpy scipy matplotlib pillow pytest
```

Tkinter is included with many Python distributions; some Linux distributions package it separately.

## Units and conventions

Angles are in radians unless stated otherwise. Keep distance and time units consistent with the gravitational parameter `mu`; for example, use kilometers, seconds, and km³/s² together. Canonical unit conversions use Earth scales by default and can be configured with a radius and gravitational parameter.

## Run the checks

From the project directory, run:

```bash
python -m pytest test_util.py test_maneuvers.py test_propagate.py
```

To open the interactive test runner:

```bash
python ssa_gui.py
```

Select **Util**, **Maneuvers**, or **Propagate** and then choose an individual test or **All tests**. The **Action** menu provides shortcuts for each suite and test. The Propagate suite includes transfer and plane-change trajectory animations; when running examples that save an animation, a file dialog lets you choose a GIF destination. Use **File → Open GIF…** or **Ctrl+O** to load a GIF into the **Preview** tab.
