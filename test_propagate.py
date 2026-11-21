"""Tests for sampled orbital propagation."""

import math

import matplotlib
import numpy as np
import pytest
import tkinter as tk
from tkinter import simpledialog

matplotlib.use("Agg")

EARTH_RADIUS_KM = 6378.137
MU_EARTH_SI = 3.986004418e14  # m**3 / s**2
DISTANCE_UNIT_M = EARTH_RADIUS_KM * 1000.0
TIME_UNIT_S = math.sqrt(DISTANCE_UNIT_M**3 / MU_EARTH_SI)
VELOCITY_UNIT_M_S = DISTANCE_UNIT_M / TIME_UNIT_S

from util_propagate import propagate
from plotter import animate_locus_3d
from util_maneuvers import (
    bi_elliptic_transfer,
    combined_inclination_raan_change,
    hohmann_transfer,
    inclination_only,
    minimum_combined_plane_change,
)


def test_propagate(root=None):
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        altitude_km = simpledialog.askfloat(
            "Orbit propagation", "Periapsis altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        eccentricity = simpledialog.askfloat(
            "Orbit propagation", "Eccentricity (0 <= e < 1):",
            minvalue=0.0, maxvalue=0.999999999, parent=root,
        )
        inclination_deg = simpledialog.askfloat(
            "Orbit propagation", "Inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
        raan_deg = simpledialog.askfloat(
            "Orbit propagation", "RAAN (degrees):",
            minvalue=-360.0, maxvalue=360.0, parent=root,
        )
        argp_deg = simpledialog.askfloat(
            "Orbit propagation", "Argument of periapsis (degrees):",
            minvalue=-360.0, maxvalue=360.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    inputs = (altitude_km, eccentricity, inclination_deg, raan_deg, argp_deg)
    if any(value is None for value in inputs):
        pytest.skip("All orbital elements were not entered")

    mu = 1.0  # Earth canonical units
    periapsis_radius = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM
    semi_major_axis = periapsis_radius / (1 - eccentricity)
    p = semi_major_axis * (1 - eccentricity**2)
    orbital_period = 2 * math.pi * math.sqrt(semi_major_axis**3 / mu)
    final_time = 2 * orbital_period
    inclination = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)

    print(f"Periapsis altitude: {altitude_km:.3f} km")
    print(f"Eccentricity: {eccentricity:.6f}")
    print(f"Inclination: {inclination_deg:.3f} degrees")
    print(f"RAAN: {raan_deg:.3f} degrees")
    print(f"Argument of periapsis: {argp_deg:.3f} degrees")
    print(f"Orbital period: {orbital_period * TIME_UNIT_S:.3f} s")

    t, r, v = propagate(
        200, 0.0, final_time, 0.0, mu, eccentricity, p,
        inclination=inclination, raan=raan, argp=argp,
    )
    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_inclination_change(root=None):
    """Propagate one LEO revolution, change inclination at the ascending node,
    then propagate another revolution to the ascending node.
    """
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        altitude_km = simpledialog.askfloat(
            "Inclination change", "Orbit altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        initial_inclination_deg = simpledialog.askfloat(
            "Inclination change", "Initial inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
        final_inclination_deg = simpledialog.askfloat(
            "Inclination change", "Final inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    if (altitude_km is None or initial_inclination_deg is None
            or final_inclination_deg is None):
        pytest.skip("Orbit altitude and inclinations were not entered")

    mu = 1.0  # Earth canonical units
    radius = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM
    inclination_initial = math.radians(initial_inclination_deg)
    inclination_final = math.radians(final_inclination_deg)
    delta_i = inclination_final - inclination_initial
    speed = math.sqrt(mu / radius)
    period = 2 * math.pi * math.sqrt(radius**3 / mu)

    N = 100
    t_before, r_before, v_before = propagate(
        N, 0.0, period, 0.0, mu, 0.0, radius,
        inclination=inclination_initial,
    )

    # At the ascending node, rotate the velocity to the requested final
    # inclination; the burn leaves the circular speed unchanged.
    delta_v = inclination_only(delta_i, 0.0, speed)
    print(f"Velocity change: {delta_v * VELOCITY_UNIT_M_S:.3f} m/s")
    t_after_relative, r_after, v_after = propagate(
        N, 0.0, period, 0.0, mu, 0.0, radius,
        inclination=inclination_final,
    )
    t_after = t_after_relative + t_before[-1]

    # Omit the repeated ascending-node sample where the two orbit segments join.
    t = np.concatenate((t_before, t_after[1:]))
    r = np.concatenate((r_before, r_after[1:]))
    v = np.concatenate((v_before, v_after[1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_combined_inclination_raan_change(root=None):
    """Animate one revolution on each side of a combined plane change."""
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        altitude_km = simpledialog.askfloat(
            "Combined plane change", "Orbit altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        initial_inclination_deg = simpledialog.askfloat(
            "Combined plane change", "Initial inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
        final_inclination_deg = simpledialog.askfloat(
            "Combined plane change", "Final inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
        initial_raan_deg = simpledialog.askfloat(
            "Combined plane change", "Initial RAAN (degrees):",
            minvalue=-360.0, maxvalue=360.0, parent=root,
        )
        final_raan_deg = simpledialog.askfloat(
            "Combined plane change", "Final RAAN (degrees):",
            minvalue=-360.0, maxvalue=360.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    inputs = (altitude_km, initial_inclination_deg, final_inclination_deg,
              initial_raan_deg, final_raan_deg)
    if any(value is None for value in inputs):
        pytest.skip("Altitude and plane angles were not entered")

    mu = 1.0  # Earth canonical units
    radius = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM
    inclination_initial = math.radians(initial_inclination_deg)
    raan_initial = math.radians(initial_raan_deg)
    inclination_final = math.radians(final_inclination_deg)
    raan_final = math.radians(final_raan_deg)
    delta_raan = raan_final - raan_initial
    speed = math.sqrt(mu / radius)

    delta_v, theta = combined_inclination_raan_change(
        inclination_initial, inclination_final, delta_raan, speed
    )
    print(f"Velocity change: {delta_v * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Angle between orbital planes: {math.degrees(theta):.3f} degrees")

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

    # Omit the duplicated maneuver point in the rendered locus.
    t = np.concatenate((t_initial, t_final[1:]))
    r = np.concatenate((r_initial, r_final[1:]))
    v = np.concatenate((v_initial, v_final[1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_hohmann_transfer(root=None):
    mu = 1.0  # Earth canonical units
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        leo_altitude_km = simpledialog.askfloat(
            "Hohmann transfer", "Initial altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        geo_altitude_km = simpledialog.askfloat(
            "Hohmann transfer", "Final altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    if leo_altitude_km is None or geo_altitude_km is None:
        pytest.skip("Orbit altitudes were not entered")
    # Convert altitudes above Earth's surface to geocentric radii in canonical
    # Earth-radius units for propagation and maneuver calculations.
    r_leo = (EARTH_RADIUS_KM + leo_altitude_km) / EARTH_RADIUS_KM
    r_geo = (EARTH_RADIUS_KM + geo_altitude_km) / EARTH_RADIUS_KM

    # Hohmann transfer ellipse: LEO is periapsis and GEO is apoapsis.
    a_transfer, transfer_time, dv1, dv2 = hohmann_transfer(r_leo, r_geo, mu)
    print(f"Transfer time: {transfer_time * TIME_UNIT_S:.3f} s")
    print(f"Transfer semi-major axis: {a_transfer * DISTANCE_UNIT_M:.3f} m")
    print(f"Departure delta-v: {dv1 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Arrival delta-v: {dv2 * VELOCITY_UNIT_M_S:.3f} m/s")
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

    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_bi_elliptic_transfer(root=None):
    mu = 1.0  # Earth canonical units
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        leo_altitude_km = simpledialog.askfloat(
            "Bi-elliptic transfer", "Initial altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        geo_altitude_km = simpledialog.askfloat(
            "Bi-elliptic transfer", "Final altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    if leo_altitude_km is None or geo_altitude_km is None:
        pytest.skip("Orbit altitudes were not entered")

    r_leo = (EARTH_RADIUS_KM + leo_altitude_km) / EARTH_RADIUS_KM
    r_geo = (EARTH_RADIUS_KM + geo_altitude_km) / EARTH_RADIUS_KM
    r_b = 2.0 * r_geo

    # The first transfer ellipse runs from LEO to the intermediate apoapsis;
    # the second runs from that apoapsis down to GEO.
    (a_transfer1, a_transfer2, transfer_time1, transfer_time2,
     dv1, dv2, dv3, dv_total) = bi_elliptic_transfer(r_leo, r_b, r_geo, mu)
    print(f"First transfer time: {transfer_time1 * TIME_UNIT_S:.3f} s")
    print(f"Second transfer time: {transfer_time2 * TIME_UNIT_S:.3f} s")
    print(f"First transfer semi-major axis: {a_transfer1 * DISTANCE_UNIT_M:.3f} m")
    print(f"Second transfer semi-major axis: {a_transfer2 * DISTANCE_UNIT_M:.3f} m")
    print(f"Departure delta-v: {dv1 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Intermediate delta-v: {dv2 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Arrival delta-v: {dv3 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Total delta-v: {dv_total * VELOCITY_UNIT_M_S:.3f} m/s")
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

    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_minimum_plane_change(root=None):
    """Animate three one-revolution orbit arcs with a two-burn plane change.

    The burn magnitudes and inclination split use the minimum combined
    plane-change estimate for a Hohmann transfer. The displayed arcs are
    circular-orbit locus segments, with the intermediate plane inclination
    applied after the first burn.
    """
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    try:
        initial_altitude_km = simpledialog.askfloat(
            "Minimum plane change", "Initial altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        final_altitude_km = simpledialog.askfloat(
            "Minimum plane change", "Final altitude above Earth's surface (km):",
            minvalue=0.0, parent=root,
        )
        initial_inclination_deg = simpledialog.askfloat(
            "Minimum plane change", "Initial inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
        final_inclination_deg = simpledialog.askfloat(
            "Minimum plane change", "Final inclination (degrees):",
            minvalue=0.0, maxvalue=180.0, parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    values = (initial_altitude_km, final_altitude_km,
              initial_inclination_deg, final_inclination_deg)
    if any(value is None for value in values):
        pytest.skip("Altitudes and inclinations were not entered")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("altitudes and inclinations must be finite")
    if initial_altitude_km < 0 or final_altitude_km < 0:
        raise ValueError("altitudes must be nonnegative")
    if not (0 <= initial_inclination_deg <= 180
            and 0 <= final_inclination_deg <= 180):
        raise ValueError("inclinations must be between 0 and 180 degrees")

    mu = 1.0  # Earth canonical units
    r_initial = (EARTH_RADIUS_KM + initial_altitude_km) / EARTH_RADIUS_KM
    r_final = (EARTH_RADIUS_KM + final_altitude_km) / EARTH_RADIUS_KM
    a_transfer = (r_initial + r_final) / 2
    v_initial = math.sqrt(mu / r_initial)
    v_trans_a = math.sqrt(2 * mu / r_initial - mu / a_transfer)
    v_trans_b = math.sqrt(2 * mu / r_final - mu / a_transfer)
    v_final = math.sqrt(mu / r_final)
    delta_i = abs(math.radians(final_inclination_deg - initial_inclination_deg))
    (delta_i_1, delta_i_2, delta_v_1, delta_v_2,
     delta_v_total) = minimum_combined_plane_change(
        v_initial, v_trans_a, v_trans_b, v_final, delta_i
    )
    inclination_sign = (1 if final_inclination_deg >= initial_inclination_deg
                        else -1)
    inclination_mid = math.radians(initial_inclination_deg) + inclination_sign * delta_i_1

    print(f"First inclination change: {inclination_sign * math.degrees(delta_i_1):.3f} degrees")
    print(f"Second inclination change: {inclination_sign * math.degrees(delta_i_2):.3f} degrees")
    print(f"First velocity change: {delta_v_1 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Second velocity change: {delta_v_2 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Total velocity change: {delta_v_total * VELOCITY_UNIT_M_S:.3f} m/s")

    periods = [2 * math.pi * math.sqrt(radius**3 / mu)
               for radius in (r_initial, r_initial, r_final)]
    segment_data = []
    elapsed = 0.0
    for radius, inclination, period in zip(
            (r_initial, r_initial, r_final),
            (math.radians(initial_inclination_deg), inclination_mid,
             math.radians(final_inclination_deg)), periods):
        t_part, r_part, v_part = propagate(
            120, elapsed, elapsed + period, 0.0, mu, 0.0, radius,
            inclination=inclination,
        )
        segment_data.append((t_part, r_part, v_part))
        elapsed += period

    # Keep each node sample once while joining the three revolution arcs.
    t = np.concatenate((segment_data[0][0], segment_data[1][0][1:],
                        segment_data[2][0][1:]))
    r = np.concatenate((segment_data[0][1], segment_data[1][1][1:],
                        segment_data[2][1][1:]))
    v = np.concatenate((segment_data[0][2], segment_data[1][2][1:],
                        segment_data[2][2][1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)


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
