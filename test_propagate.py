"""Tests for sampled orbital propagation."""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from util_propagate import propagate
from plotter import animate_locus_3d
from util_maneuvers import bi_elliptic_transfer, hohmann_transfer


def test_propagate_and_animate_locus_3d():
    t, r, v = propagate(100, 0, 19.743, 0, 1, 0, 1.03136)
    _, _, animation = animate_locus_3d(t, r, v, show=False)
    assert animation is not None


def test_leo_to_geo_hohmann_transfer_animation():
    earth_radius_km = 6378.137
    mu = 1.0  # Earth canonical units
    r_leo = (earth_radius_km + 200.0) / earth_radius_km
    r_geo = (earth_radius_km + 35_786.0) / earth_radius_km

    # Hohmann transfer ellipse: LEO is periapsis and GEO is apoapsis.
    a_transfer, transfer_time, _, _ = hohmann_transfer(r_leo, r_geo, mu)
    e_transfer = (r_geo - r_leo) / (r_geo + r_leo)
    p_transfer = a_transfer * (1 - e_transfer**2)
    leo_period = 2 * math.pi * math.sqrt(r_leo**3 / mu)
    geo_period = 2 * math.pi * math.sqrt(r_geo**3 / mu)

    factor = 300/(leo_period+transfer_time+geo_period)
    N = int(leo_period*factor)
    t_leo, r_leo_path, v_leo_path = propagate(
        N, 0.0, leo_period, 0.0, mu, 0.0, r_leo
    )
    N = int(transfer_time*factor)
    t_transfer, r_transfer, v_transfer = propagate(
        N, leo_period, leo_period + transfer_time, 0.0, mu,
        e_transfer, p_transfer,
    )
    N = int(geo_period*factor)
    t_geo, r_geo_path, v_geo_path = propagate(
        N, leo_period + transfer_time,
        leo_period + transfer_time + geo_period,
        math.pi, mu, 0.0, r_geo,
    )

    # Drop duplicate endpoint samples where the orbit segments join.
    t = np.concatenate((t_leo, t_transfer[1:], t_geo[1:]))
    r = np.concatenate((r_leo_path, r_transfer[1:], r_geo_path[1:]))
    v = np.concatenate((v_leo_path, v_transfer[1:], v_geo_path[1:]))

    assert np.linalg.norm(r_transfer[0]) == pytest.approx(r_leo)
    assert np.linalg.norm(r_transfer[-1]) == pytest.approx(r_geo)
    assert np.linalg.norm(r_geo_path[-1]) == pytest.approx(r_geo, abs=1e-7)

    _, _, animation = animate_locus_3d(t, r, v, show=False)
    assert animation is not None


def test_leo_to_geo_bi_elliptic_transfer_animation():
    earth_radius_km = 6378.137
    mu = 1.0  # Earth canonical units
    r_leo = (earth_radius_km + 200.0) / earth_radius_km
    r_geo = (earth_radius_km + 35_786.0) / earth_radius_km
    r_b = 2.0 * r_geo

    # The first transfer ellipse runs from LEO to the intermediate apoapsis;
    # the second runs from that apoapsis down to GEO.
    (a_transfer1, a_transfer2, transfer_time1, transfer_time2,
     _, _, _, _) = bi_elliptic_transfer(r_leo, r_b, r_geo, mu)
    e_transfer1 = (r_b - r_leo) / (r_b + r_leo)
    p_transfer1 = a_transfer1 * (1 - e_transfer1**2)
    e_transfer2 = (r_b - r_geo) / (r_b + r_geo)
    p_transfer2 = a_transfer2 * (1 - e_transfer2**2)

    leo_period = 2 * math.pi * math.sqrt(r_leo**3 / mu)
    geo_period = 2 * math.pi * math.sqrt(r_geo**3 / mu)
    total_time = leo_period + transfer_time1 + transfer_time2 + geo_period
    factor = 400 / total_time

    N = int(leo_period * factor)
    t_leo, r_leo_path, v_leo_path = propagate(
        N, 0.0, leo_period, 0.0, mu, 0.0, r_leo
    )
    N = int(transfer_time1 * factor)
    t_transfer1, r_transfer1, v_transfer1 = propagate(
        N, leo_period, leo_period + transfer_time1, 0.0, mu,
        e_transfer1, p_transfer1,
    )
    N = int(transfer_time2 * factor)
    t_transfer2, r_transfer2, v_transfer2 = propagate(
        N, leo_period + transfer_time1,
        leo_period + transfer_time1 + transfer_time2, math.pi, mu,
        e_transfer2, p_transfer2,
    )
    N = int(geo_period * factor)
    t_geo, r_geo_path, v_geo_path = propagate(
        N, leo_period + transfer_time1 + transfer_time2,
        total_time, 0.0, mu, 0.0, r_geo,
    )

    # Drop duplicate endpoint samples where the orbit segments join.
    t = np.concatenate((t_leo, t_transfer1[1:], t_transfer2[1:], t_geo[1:]))
    r = np.concatenate((r_leo_path, r_transfer1[1:], r_transfer2[1:], r_geo_path[1:]))
    v = np.concatenate((v_leo_path, v_transfer1[1:], v_transfer2[1:], v_geo_path[1:]))

    assert np.linalg.norm(r_transfer1[0]) == pytest.approx(r_leo)
    assert np.linalg.norm(r_transfer1[-1]) == pytest.approx(r_b)
    assert np.linalg.norm(r_transfer2[0]) == pytest.approx(r_b)
    assert np.linalg.norm(r_transfer2[-1]) == pytest.approx(r_geo)
    assert np.linalg.norm(r_geo_path[-1]) == pytest.approx(r_geo, abs=1e-7)

    _, _, animation = animate_locus_3d(t, r, v, show=False)
    assert animation is not None


def run_all_tests():
    """Run every test function in this module and raise if any fail."""
    test_functions = [
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    ]
    failures = []

    for name, function in test_functions:
        try:
            function()
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")

    passed = len(test_functions) - len(failures)
    print(f"{passed}/{len(test_functions)} tests passed")
    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        raise AssertionError(f"Test failures: {failed_names}") from failures[0][1]

if __name__ == "__main__":
    run_all_tests()
