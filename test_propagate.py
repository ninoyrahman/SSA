"""Tests for sampled orbital propagation."""

import math

import matplotlib
import numpy as np
import pytest
import tkinter as tk
from tkinter import messagebox, ttk

matplotlib.use("Agg")

EARTH_RADIUS_KM = 6378.137
MU_EARTH_SI = 3.986004418e14  # m**3 / s**2
DISTANCE_UNIT_M = EARTH_RADIUS_KM * 1000.0
TIME_UNIT_S = math.sqrt(DISTANCE_UNIT_M**3 / MU_EARTH_SI)
VELOCITY_UNIT_M_S = DISTANCE_UNIT_M / TIME_UNIT_S


def _ask_numeric_fields(root, title, fields):
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.transient(root)
    dialog.resizable(False, False)

    entries = []
    for row, (label, default, _minimum, _maximum) in enumerate(fields):
        ttk.Label(dialog, text=label).grid(
            row=row, column=0, padx=(12, 8), pady=5, sticky="w"
        )
        entry = ttk.Entry(dialog, width=18)
        entry.insert(0, default)
        entry.grid(row=row, column=1, padx=(0, 12), pady=5, sticky="ew")
        entries.append(entry)

    values = {"result": None}

    def submit():
        parsed = []
        try:
            for entry, (label, _default, minimum, maximum) in zip(entries, fields):
                value = float(entry.get())
                if not math.isfinite(value):
                    raise ValueError(f"{label} must be a finite number")
                if value < minimum or (maximum is not None and value > maximum):
                    raise ValueError(f"{label} is outside the allowed range")
                parsed.append(value)
        except ValueError as exc:
            messagebox.showerror("Invalid orbital input", str(exc), parent=dialog)
            return
        values["result"] = parsed
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=len(fields), column=0, columnspan=2, pady=(8, 12))
    ttk.Button(buttons, text="OK", command=submit).pack(side=tk.LEFT, padx=5)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
        side=tk.LEFT, padx=5
    )
    dialog.bind("<Return>", lambda _event: submit())
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    entries[0].focus_set()
    dialog.grab_set()
    root.wait_window(dialog)
    return values["result"]

from util_propagate import propagate
from plotter import animate_locus_3d
from util_maneuvers import (
    bi_elliptic_transfer,
    combined_inclination_raan_change,
    hohmann_transfer,
    inclination_only,
    minimum_combined_plane_change,
    payload_angle,
    circular_coplanar_phasing_same_orbit,
    circular_coplanar_phasing_different_orbits,
)


def test_propagate(root=None):
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    fields = (
        ("Periapsis altitude above Earth's surface (km):", "200", 0.0, None),
        ("Eccentricity (0 <= e < 1):", "0", 0.0, 0.999999999),
        ("Inclination (degrees):", "0", 0.0, 180.0),
        ("RAAN (degrees):", "0", -360.0, 360.0),
        ("Argument of periapsis (degrees):", "0", -360.0, 360.0),
    )
    try:
        inputs = _ask_numeric_fields(root, "Orbit propagation", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("All orbital elements were not entered")
    altitude_km, eccentricity, inclination_deg, raan_deg, argp_deg = inputs

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


def test_rendezvous_same_orbit(root=None):
    """Prompt for inputs and animate a same-orbit rendezvous."""
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    fields = (
        ("Target/interceptor altitude above Earth's surface (km):", "200", 0.0, None),
        ("Angle from target to interceptor (degrees):", "-20", -360.0, 360.0),
        ("Interceptor phasing revolutions (k_int):", "1", 1.0, None),
        ("Target revolutions (k_tgt):", "1", 0.0, None),
    )
    try:
        inputs = _ask_numeric_fields(root, "Same-orbit rendezvous", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Rendezvous inputs were not entered")
    altitude_km, theta_deg, k_int_input, k_tgt_input = inputs
    if not k_int_input.is_integer() or not k_tgt_input.is_integer():
        raise ValueError("k_int and k_tgt must be whole numbers")
    k_int, k_tgt = int(k_int_input), int(k_tgt_input)
    a_tgt = (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM

    mu = 1.0
    theta = math.radians(theta_deg)
    phase_time, delta_v, a_phase = circular_coplanar_phasing_same_orbit(
        a_tgt, theta, k_tgt, k_int, mu
    )
    phase_period = 2 * math.pi * math.sqrt(a_phase**3 / mu)
    interceptor_phase_time = k_int * phase_period
    target_period = 2 * math.pi * math.sqrt(a_tgt**3 / mu)

    print(f"Phasing time: {phase_time * TIME_UNIT_S:.3f} s")
    print(f"Phasing orbit semi-major axis: {a_phase:.6f} Earth radii")
    print(f"Interceptor impulse velocity change: {delta_v * VELOCITY_UNIT_M_S:.3f} m/s")

    # The target starts at the negative entered angle relative to the
    # interceptor. Sampling its
    # one-revolution path also establishes the rendezvous point at phase_time.
    n_samples = max(120, int(240 * phase_time / target_period))
    t_target, r_target, v_target = propagate(
        n_samples, 0.0, phase_time, -theta, mu, 0.0, a_tgt
    )
    # The first burn is tangential at the original circular radius. Depending
    # on the phasing semi-major axis, that point is periapsis or apoapsis.
    eccentricity = abs(a_tgt / a_phase - 1.0)
    nu0 = 0.0 if a_phase >= a_tgt else math.pi
    argp = 0.0 if nu0 == 0.0 else math.pi
    p_phase = a_phase * (1.0 - eccentricity**2)
    _, r_int, v_int = propagate(
        n_samples, 0.0, interceptor_phase_time, nu0, mu, eccentricity, p_phase,
        argp=argp,
    )
    # Both paths share the same rendezvous point. Animate the target's full
    # revolution followed by the interceptor's phasing trajectory.
    t = np.concatenate((t_target, t_target[-1] + t_target[1:]))
    r = np.concatenate((r_target, r_int[1:]))
    v = np.concatenate((v_target, v_int[1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_rendezvous_different_orbits(root=None):
    """Animate a rendezvous from a lower circular orbit to a higher one."""
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    fields = (
        ("Target altitude above Earth's surface (km):", "35786", 0.0, None),
        ("Interceptor altitude above Earth's surface (km):", "6400", 0.0, None),
        ("Angle from target to interceptor (degrees):", "20", -360.0, 360.0),
        ("Interceptor initial-orbit revolutions (k):", "1", 0.0, None),
    )
    try:
        inputs = _ask_numeric_fields(root, "Different-orbit rendezvous", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Rendezvous inputs were not entered")
    target_altitude_km, interceptor_altitude_km, theta_deg, k_input = inputs
    if not k_input.is_integer():
        raise ValueError("k must be a whole number")
    if interceptor_altitude_km >= target_altitude_km:
        raise ValueError("Interceptor altitude must be below target altitude")

    mu = 1.0  # Earth canonical units
    a_tgt = (EARTH_RADIUS_KM + target_altitude_km) / EARTH_RADIUS_KM
    a_int = (EARTH_RADIUS_KM + interceptor_altitude_km) / EARTH_RADIUS_KM
    theta_i = math.radians(theta_deg)
    tau_trans, delta_v, a_trans, tau_wait = (
        circular_coplanar_phasing_different_orbits(
            theta_i, a_int, a_tgt, int(k_input), mu
        )
    )
    if tau_wait < 0:
        raise ValueError("Selected k produces a negative wait time; increase k")

    e_trans = (a_tgt - a_int) / (a_tgt + a_int)
    p_trans = a_trans * (1.0 - e_trans**2)
    period_int = 2 * math.pi * math.sqrt(a_int**3 / mu)
    total_time = tau_wait + tau_trans
    samples = 100 * (1 + tau_wait / tau_trans)
    wait_samples = int(samples * tau_wait / total_time)
    transfer_samples = int(samples * tau_trans / total_time)

    # The target starts at angle zero; the interceptor begins theta_i radians
    # ahead of it and coasts for tau_wait before the pericenter burn.
    t_wait, r_wait, v_wait = propagate(
        wait_samples, 0.0, tau_wait, theta_i, mu, 0.0, a_int
    )
    interceptor_angle_at_burn = theta_i + math.sqrt(mu / a_int**3) * tau_wait
    t_transfer, r_transfer, v_transfer = propagate(
        transfer_samples, tau_wait, total_time, 0.0, mu, e_trans, p_trans,
        argp=interceptor_angle_at_burn,
    )

    print(f"Wait time: {tau_wait * TIME_UNIT_S:.3f} s")
    print(f"Transfer time: {tau_trans * TIME_UNIT_S:.3f} s")
    print(f"Transfer semi-major axis: {a_trans:.6f} Earth radii")
    print(f"Total delta-v: {delta_v * VELOCITY_UNIT_M_S:.3f} m/s")

    t = np.concatenate((t_wait, t_transfer[1:]))
    r = np.concatenate((r_wait, r_transfer[1:]))
    v = np.concatenate((v_wait, v_transfer[1:]))
    _, _, animation = animate_locus_3d(t, r, v, show=False)


def test_inclination_change(root=None):
    """Propagate one LEO revolution, change inclination at the ascending node,
    then propagate another revolution to the ascending node.
    """
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    fields = (
        ("Orbit altitude above Earth's surface (km):", "200", 0.0, None),
        ("Initial inclination (degrees):", "20", 0.0, 180.0),
        ("Final inclination (degrees):", "25", 0.0, 180.0),
    )
    try:
        inputs = _ask_numeric_fields(root, "Inclination change", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Orbit altitude and inclinations were not entered")
    altitude_km, initial_inclination_deg, final_inclination_deg = inputs

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
    fields = (
        ("Orbit altitude above Earth's surface (km):", "200", 0.0, None),
        ("Initial inclination (degrees):", "20", 0.0, 180.0),
        ("Final inclination (degrees):", "25", 0.0, 180.0),
        ("Initial RAAN (degrees):", "0", -360.0, 360.0),
        ("Final RAAN (degrees):", "30", -360.0, 360.0),
    )
    try:
        inputs = _ask_numeric_fields(root, "Combined plane change", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Altitude and plane angles were not entered")
    (altitude_km, initial_inclination_deg, final_inclination_deg,
     initial_raan_deg, final_raan_deg) = inputs

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
    fields = (
        ("Initial altitude above Earth's surface (km):", "200", 0.0, None),
        ("Final altitude above Earth's surface (km):", "35786", 0.0, None),
    )
    try:
        inputs = _ask_numeric_fields(root, "Hohmann transfer", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Orbit altitudes were not entered")
    leo_altitude_km, geo_altitude_km = inputs
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
    fields = (
        ("Initial altitude above Earth's surface (km):", "200", 0.0, None),
        ("Final altitude above Earth's surface (km):", "35786", 0.0, None),
    )
    try:
        inputs = _ask_numeric_fields(root, "Bi-elliptic transfer", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Orbit altitudes were not entered")
    leo_altitude_km, geo_altitude_km = inputs

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
    """Animate a two-burn minimum-cost plane change over a Hohmann transfer."""
    owns_root = root is None
    if owns_root:
        root = tk.Tk()
        root.withdraw()
    fields = (
        ("Initial altitude above Earth's surface (km):", "200", 0.0, None),
        ("Final altitude above Earth's surface (km):", "35786", 0.0, None),
        ("Initial inclination (degrees):", "20", 0.0, 180.0),
        ("Final inclination (degrees):", "25", 0.0, 180.0),
    )
    try:
        inputs = _ask_numeric_fields(root, "Minimum plane change", fields)
    finally:
        if owns_root:
            root.destroy()
    if inputs is None:
        pytest.skip("Altitudes and inclinations were not entered")
    (initial_altitude_km, final_altitude_km,
     initial_inclination_deg, final_inclination_deg) = inputs
    values = (initial_altitude_km, final_altitude_km,
              initial_inclination_deg, final_inclination_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("altitudes and inclinations must be finite")
    if initial_altitude_km < 0 or final_altitude_km < 0:
        raise ValueError("altitudes must be nonnegative")
    if final_altitude_km < initial_altitude_km:
        raise ValueError("final altitude must be at least the initial altitude")
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

    transfer_time = math.pi * math.sqrt(a_transfer**3 / mu)
    e_transfer = (r_final - r_initial) / (r_final + r_initial)
    p_transfer = a_transfer * (1 - e_transfer**2)

    gamma_peri = payload_angle(v_initial, v_trans_a, delta_v_1)
    gamma_apo = payload_angle(v_trans_b, v_final, delta_v_2)

    print(f"Transfer time: {transfer_time * TIME_UNIT_S:.3f} s")
    peri_angle = "undefined (zero impulse)" if gamma_peri is None else f"{gamma_peri:.3f} degrees"
    apo_angle = "undefined (zero impulse)" if gamma_apo is None else f"{gamma_apo:.3f} degrees"
    print(f"Payload angle at transfer pericenter: {peri_angle}")
    print(f"Payload angle at transfer apocenter: {apo_angle}")
    print(f"First inclination change: {inclination_sign * math.degrees(delta_i_1):.3f} degrees")
    print(f"Second inclination change: {inclination_sign * math.degrees(delta_i_2):.3f} degrees")
    print(f"First velocity change: {delta_v_1 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Second velocity change: {delta_v_2 * VELOCITY_UNIT_M_S:.3f} m/s")
    print(f"Total velocity change: {delta_v_total * VELOCITY_UNIT_M_S:.3f} m/s")

    initial_period = 2 * math.pi * math.sqrt(r_initial**3 / mu)
    final_period = 2 * math.pi * math.sqrt(r_final**3 / mu)
    transfer_start = initial_period
    transfer_end = transfer_start + transfer_time
    total_time = transfer_end + final_period
    factor = 240 / total_time
    t_initial, r_initial_path, v_initial_path = propagate(
        max(2, int(initial_period * factor)), 0.0, initial_period, 0.0,
        mu, 0.0, r_initial,
        inclination=math.radians(initial_inclination_deg),
    )
    t_transfer, r_transfer, v_transfer = propagate(
        max(2, int(transfer_time * factor)), transfer_start, transfer_end, 0.0,
        mu, e_transfer, p_transfer, inclination=inclination_mid,
    )
    t_final, r_final_path, v_final_path = propagate(
        max(2, int(final_period * factor)), transfer_end, total_time, math.pi,
        mu, 0.0, r_final,
        inclination=math.radians(final_inclination_deg),
    )

    # Each arc joins at an ascending node or transfer apocenter; retain one
    # sample at each maneuver point.
    t = np.concatenate((t_initial, t_transfer[1:], t_final[1:]))
    r = np.concatenate((r_initial_path, r_transfer[1:], r_final_path[1:]))
    v = np.concatenate((v_initial_path, v_transfer[1:], v_final_path[1:]))
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
