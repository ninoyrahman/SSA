"""Checks for Vallado's Chapter 2 Kepler equation examples."""

import math

from util import (
    anomaly_to_nu,
    coe_to_rv,
    find_tof,
    kep_eqn_e,
    kep_eqn_h,
    kep_eqn_p,
    kepler_universal,
    nu_to_anomaly,
)


def test_problem_1_anomalies_and_universal_variable():
    """Chapter 2, Problem 1: anomalies for M=235.4 deg, e=0.4, a=8000 km."""
    mean_anomaly = math.radians(235.4)
    eccentricity = 0.4
    semimajor_axis_km = 8000.0
    eccentric_anomaly = kep_eqn_e(mean_anomaly, eccentricity)
    true_anomaly = anomaly_to_nu(eccentric_anomaly, eccentricity)
    universal_variable_km_sqrt = math.sqrt(semimajor_axis_km) * eccentric_anomaly

    assert math.isclose(
        nu_to_anomaly(true_anomaly, eccentricity), eccentric_anomaly,
        abs_tol=1e-8,
    )
    assert math.isclose(
        eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly),
        mean_anomaly,
        abs_tol=1e-8,
    )
    return math.degrees(eccentric_anomaly), math.degrees(true_anomaly), universal_variable_km_sqrt


def test_example_2_parabolic_kepler_equation():
    """Vallado Example 2-2 for a parabolic Earth orbit.

    Uses delta_t=53.7874 minutes, p=25,512 km, and mu=398600 km^3/s^2.
    """
    delta_t_seconds = 53.7874 * 60
    semi_latus_rectum_km = 25_512
    earth_mu_km3_s2 = 398600
    anomaly = kep_eqn_p(delta_t_seconds, semi_latus_rectum_km, earth_mu_km3_s2)

    # Barker's equation is B + B^3/3 = 2*sqrt(mu/p^3)*delta_t.
    n_p = 2 * math.sqrt(earth_mu_km3_s2 / semi_latus_rectum_km**3)
    assert math.isclose(anomaly + anomaly**3 / 3, n_p * delta_t_seconds, abs_tol=1e-12)
    return anomaly


def test_problem_3_rogue_asteroid_time_from_periapsis():
    """Chapter 2, Problem 3: time to the given hyperbolic true anomaly."""
    earth_mu_km3_s2 = 398600.0
    semimajor_axis_km = -2797.425069
    eccentricity = 2.8
    true_anomaly = math.radians(249.27)
    stated_radius_km = 2_105_124.388
    hyperbolic_anomaly = nu_to_anomaly(true_anomaly, eccentricity)
    mean_anomaly = eccentricity * math.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    mean_motion = math.sqrt(earth_mu_km3_s2 / (-semimajor_axis_km) ** 3)
    time_to_periapsis_seconds = -mean_anomaly / mean_motion
    semilatus_rectum_km = semimajor_axis_km * (1 - eccentricity**2)
    calculated_radius_km = semilatus_rectum_km / (1 + eccentricity * math.cos(true_anomaly))

    # The book rounds the anomaly, which creates a visible radius difference
    # this close to the hyperbolic asymptote.
    assert math.isclose(calculated_radius_km, stated_radius_km, rel_tol=0.03)
    assert math.isclose(
        kep_eqn_h(mean_anomaly, eccentricity), hyperbolic_anomaly,
        abs_tol=1e-7,
    )
    assert math.isclose(
        anomaly_to_nu(hyperbolic_anomaly, eccentricity) % (2 * math.pi),
        true_anomaly % (2 * math.pi),
        abs_tol=1e-7,
    )
    assert time_to_periapsis_seconds > 0
    return time_to_periapsis_seconds


def test_example_4_kepler_problem_propagation():
    """Vallado Example 2-4: propagate a state for 40 minutes."""
    r0_km = (1131.340, -2282.343, 6672.423)
    v0_km_s = (-5.64305, 4.30333, 2.42879)
    delta_t_seconds = 40 * 60
    earth_mu_km3_s2 = 398600.4418
    expected_r_km = (-4219.7527, 4363.0292, -3958.7666)
    expected_v_km_s = (3.689866, -1.916735, -6.112511)

    r, v = kepler_universal(r0_km, v0_km_s, delta_t_seconds, earth_mu_km3_s2)

    # The book rounds its final state, so allow a small rounding tolerance.
    assert all(math.isclose(actual, expected, abs_tol=1e-3)
               for actual, expected in zip(r, expected_r_km))
    assert all(math.isclose(actual, expected, abs_tol=1e-5)
               for actual, expected in zip(v, expected_v_km_s))
    return r, v


def test_problem_5_shuttle_prediction_is_inconsistent():
    """Chapter 2, Problem 5: check the stated circular-orbit prediction."""
    r0_km = (6275.396, 2007.268, 1089.857)
    semimajor_axis_km = 6678.137
    inclination = math.radians(28.5)
    predicted_mean_anomaly = math.radians(278.94688)
    delta_t_seconds = 55 * 60
    earth_mu_km3_s2 = 398600.0

    radius = math.sqrt(sum(component**2 for component in r0_km))
    # For a circular orbit, the argument of latitude u is determined (up to
    # its ascending/descending branch) by z/r = sin(i) sin(u).
    sin_u = r0_km[2] / (radius * math.sin(inclination))
    assert abs(sin_u) <= 1
    u = math.asin(sin_u)
    u_candidates = (u, math.pi - u)
    mean_motion = math.sqrt(earth_mu_km3_s2 / semimajor_axis_km**3)
    predicted_candidates = tuple(
        (candidate + mean_motion * delta_t_seconds) % (2 * math.pi)
        for candidate in u_candidates
    )

    # The vector's radius agrees with the claimed circular orbit, but neither
    # possible orbital branch advances to the stated mean anomaly in 55 min.
    assert math.isclose(radius, semimajor_axis_km, abs_tol=0.001)
    assert all(
        abs(math.remainder(predicted_mean_anomaly - candidate, 2 * math.pi)) > math.radians(1)
        for candidate in predicted_candidates
    )
    return tuple(math.degrees(value) for value in predicted_candidates)


def test_example_6_coe_to_rv():
    """Vallado Example 2-6: convert classical elements to state vectors."""
    p_km = 11067.790
    eccentricity = 0.83285
    inclination = math.radians(87.87)
    raan = math.radians(227.89)
    argument_of_periapsis = math.radians(53.38)
    true_anomaly = math.radians(92.335)
    earth_mu_km3_s2 = 398600.4418
    expected_r_km = (6525.344, 6861.535, 6449.125)
    expected_v_km_s = (4.902276, 5.533124, -1.975709)

    r, v = coe_to_rv(
        p_km,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly,
        earth_mu_km3_s2,
    )

    # Vallado rounds the elements and resulting state vectors in the example.
    assert all(math.isclose(actual, expected, abs_tol=0.1)
               for actual, expected in zip(r, expected_r_km))
    assert all(math.isclose(actual, expected, abs_tol=0.001)
               for actual, expected in zip(v, expected_v_km_s))
    return r, v


def test_problem_7_time_of_flight_between_positions():
    """Chapter 2, Problem 7: time between two positions on p=6681.571 km."""
    r0_km = (-2574.9533, 4267.0671, 4431.5026)
    r_km = (2700.6738, -4303.5378, -4358.2499)
    semi_latus_rectum_km = 6681.571
    earth_mu_km3_s2 = 398600.0
    time_seconds = find_tof(r0_km, r_km, semi_latus_rectum_km, earth_mu_km3_s2)

    assert math.isfinite(time_seconds) and time_seconds > 0
    return time_seconds


def run_all_tests():
    """Run every Vallado check in this module and return their results."""
    tests = (
        test_problem_1_anomalies_and_universal_variable,
        test_example_2_parabolic_kepler_equation,
        test_problem_3_rogue_asteroid_time_from_periapsis,
        test_example_4_kepler_problem_propagation,
        test_problem_5_shuttle_prediction_is_inconsistent,
        test_example_6_coe_to_rv,
        test_problem_7_time_of_flight_between_positions,
    )
    return {test.__name__: test() for test in tests}
