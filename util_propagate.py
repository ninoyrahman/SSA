"""Convenience routines for sampling two-body orbital propagation."""

import math

import numpy as np

from util import coe_to_rv, kepler_coe


def propagate(N, t0, tf, nu0, mu, e, p, *, inclination=0.0, raan=0.0,
              argp=0.0):
    """Propagate an orbit at N+1 equally spaced times from t0 through tf.

    Args:
        N: Number of time intervals (must be a positive integer).
        t0: Initial time.
        tf: Final time.
        nu0: Initial true anomaly, in radians.
        mu: Gravitational parameter.
        e: Eccentricity.
        p: Semi-latus rectum.
        inclination: Orbital inclination in radians (default: 0).
        raan: Right ascension of the ascending node in radians (default: 0).
        argp: Argument of periapsis in radians (default: 0).

    Returns:
        NumPy arrays ``(t, r, v)`` with shapes ``(N+1,)`` and ``(N+1, 3)``.
    """
    if isinstance(N, bool) or not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not all(math.isfinite(value) for value in
               (t0, tf, nu0, mu, e, p, inclination, raan, argp)):
        raise ValueError(
            "t0, tf, nu0, mu, e, p, inclination, raan, and argp must be finite"
        )
    if mu <= 0 or e < 0 or p <= 0:
        raise ValueError("mu and p must be positive and e must be nonnegative")

    dt = (tf - t0) / N
    t = t0 + dt * np.arange(N + 1)
    r0, v0 = coe_to_rv(p, e, inclination, raan, argp, nu0, mu)
    states = [kepler_coe(r0, v0, ti - t0, mu) for ti in t]
    r = np.asarray([state[0] for state in states])
    v = np.asarray([state[1] for state in states])
    return t, r, v
