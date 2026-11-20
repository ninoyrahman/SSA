"""Tests for sampled orbital propagation."""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from util_propagate import propagate
from plotter import animate_locus_3d
from util_maneuvers import (
    bi_elliptic_transfer,
    combined_inclination_raan_change,
    hohmann_transfer,
    inclination_only,
)


def test_propagate_and_animate_locus_3d():
    t, r, v = propagate(100, 0, 19.743, 0, 1, 0, 1.03136)
    _, _, animation = animate_locus_3d(t, r, v, show=False)
    assert animation is not None


def test_leo_inclination_change_node_to_node():
    """Propagate one LEO revolution, change inclination at the ascending node,
    then propagate another revolution to the ascending node.
    """
    earth_radius_km = 6378.137
    mu = 1.0  # Earth canonical units
    radius = (earth_radius_km + 200.0) / earth_radius_km
    inclination_initial = math.radians(20.0)
    delta_i = math.radians(5.0)
    inclination_final = inclination_initial + delta_i
    speed = math.sqrt(mu / radius)
    period = 2 * math.pi * math.sqrt(radius**3 / mu)

    N = 100
    t_before, r_before, v_before = propagate(
        N, 0.0, period, 0.0, mu, 0.0, radius,
        inclination=inclination_initial,
    )

    # At the ascending node, rotate the velocity within the local plane by
    # 5 degrees; the burn leaves the circular speed unchanged.
    delta_v = inclination_only(delta_i, 0.0, speed)
    t_after_relative, r_after, v_after = propagate(
        N, 0.0, period, 0.0, mu, 0.0, radius,
        inclination=inclination_final,
    )
    assert np.linalg.norm(v_after[0] - v_before[-1]) == pytest.approx(delta_v)
    t_after = t_after_relative + t_before[-1]

    # Omit the repeated ascending-node sample where the two orbit segments join.
    t = np.concatenate((t_before, t_after[1:]))
    r = np.concatenate((r_before, r_after[1:]))
    v = np.concatenate((v_before, v_after[1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)
    assert animation is not None


def test_leo_combined_inclination_raan_change_node_to_node():
    """Animate one revolution on each side of a combined plane change."""
    earth_radius_km = 6378.137
    mu = 1.0  # Earth canonical units
    radius = (earth_radius_km + 200.0) / earth_radius_km
    inclination_initial = math.radians(55.0)
    raan_initial = 0.0
    inclination_final = math.radians(40.0)
    raan_final = math.radians(45.0)
    delta_raan = raan_final - raan_initial
    speed = math.sqrt(mu / radius)

    delta_v, theta = combined_inclination_raan_change(
        inclination_initial, inclination_final, delta_raan, speed
    )

    # Eqs. 6.25 and 6.26 give cos(u_i) and cos(u_f); use their positive
    # arccos branches, which describe the same common position in both planes.
    cos_u_initial = (
        math.sin(inclination_final) * math.cos(delta_raan)
        - math.cos(theta) * math.sin(inclination_initial)
    ) / (math.sin(theta) * math.cos(inclination_initial))
    cos_u_final = (
        math.cos(inclination_initial) * math.sin(inclination_final)
        - math.sin(inclination_initial) * math.cos(inclination_final)
        * math.cos(delta_raan)
    ) / math.sin(theta)
    u_initial = math.acos(max(-1.0, min(1.0, cos_u_initial)))
    u_final = math.acos(max(-1.0, min(1.0, cos_u_final)))

    period = 2 * math.pi * math.sqrt(radius**3 / mu)
    N = 100

    t_initial, r_initial, v_initial = propagate(
        N, 0.0, period, u_initial, mu, 0.0, radius,
        inclination=inclination_initial,
        raan=raan_initial,
    )

    t_final_relative, r_final, v_final = propagate(
        N, 0.0, period, u_final, mu, 0.0, radius,
        inclination=inclination_final,
        raan=raan_final,
    )
    t_final = t_final_relative + t_initial[-1]

    # Both propagated arcs meet at the common point. The velocity-vector jump
    # there must match the combined plane-change maneuver calculation.
    assert r_initial[-1] == pytest.approx(r_final[0], abs=1e-10)
    assert np.linalg.norm(v_final[0] - v_initial[-1]) == pytest.approx(delta_v)

    # Omit the duplicated maneuver point in the rendered locus.
    t = np.concatenate((t_initial, t_final[1:]))
    r = np.concatenate((r_initial, r_final[1:]))
    v = np.concatenate((v_initial, v_final[1:]))
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
    # test_leo_combined_inclination_raan_change_node_to_node()
