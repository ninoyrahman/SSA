"""Regression checks against Vallado, Chapter 6 worked examples.

The PDF provided is a chapter excerpt (printed pages 316–420). Its end-of-
chapter problems are prompts without answer keys in this excerpt, so these
checks use the numerical results printed in Examples 6-1 through 6-10 and
6-14 instead.
"""

import math

import pytest

from util import canonical_to_si, si_to_canonical
from util_maneuvers import (
    bi_elliptic_transfer,
    change_ascending_node_circular,
    circular_coplanar_phasing_different_orbits,
    circular_coplanar_phasing_same_orbit,
    combined_inclination_raan_change,
    hill_equations,
    hohmann_transfer,
    inclination_only,
    minimum_combined_plane_change,
    one_tangent_burn,
)


def test_example_6_1_hohmann_transfer():
    # Vallado uses canonical Earth units: mu=1 ER^3/TU^2.
    a, tof, dv_a, dv_b = hohmann_transfer(1.03, 6.61, 1.0)
    assert a == pytest.approx(3.82, abs=1e-3)
    assert tof == pytest.approx(23.455510, abs=1e-3)
    assert dv_a == pytest.approx(1.296136 - 0.985329, abs=1e-3)
    assert dv_b == pytest.approx(0.388955 - 0.201970, abs=1e-3)


def test_example_6_2_bi_elliptic_transfer():
    # Vallado's Example 6-2 rounds the orbit radii to 1.03, 80, and 60 ER.
    a1, a2, tof1, tof2, dv_a, dv_b, dv_c, dv_total = bi_elliptic_transfer(
        1.03, 80.0, 60.0, 1.0
    )
    assert a1 == pytest.approx(40.515, abs=1e-3)
    assert a2 == pytest.approx(70.0, abs=1e-3)
    assert tof1 == pytest.approx(math.pi * math.sqrt(40.515**3), abs=1e-3)
    assert tof2 == pytest.approx(math.pi * math.sqrt(70.0**3), abs=1e-3)
    assert dv_a == pytest.approx(1.384581 - 1 / math.sqrt(1.03), abs=1e-3)
    assert dv_c == pytest.approx(1 / math.sqrt(60) - 0.138013, abs=1e-3)
    assert dv_total == pytest.approx(abs(dv_a) + abs(dv_b) + abs(dv_c), abs=1e-3)


def test_example_6_3_one_tangent_burn():
    e, a, tof, dv_a, dv_b, dv_total = one_tangent_burn(
        1.03, 6.61, math.radians(160), 1.0
    )
    assert e == pytest.approx(0.770578, abs=1e-3)
    assert a == pytest.approx(4.489438, abs=1e-3)
    eccentric_anomaly = 2 * math.atan2(
        math.sqrt(1 - e) * math.sin(math.radians(160) / 2),
        math.sqrt(1 + e) * math.cos(math.radians(160) / 2),
    )
    expected_tof = a**1.5 * (eccentric_anomaly - e * math.sin(eccentric_anomaly))
    assert tof == pytest.approx(expected_tof, abs=1e-3)
    assert dv_a == pytest.approx(1.311108 - 1 / math.sqrt(1.03), abs=1e-3)
    assert dv_b >= 0
    assert dv_total == pytest.approx(abs(dv_a) + abs(dv_b), abs=1e-3)


def test_example_6_4_inclination_only():
    dv = inclination_only(math.radians(15), 0.0, 0.745356)
    assert dv == pytest.approx(0.1945770, abs=1e-3)


def test_example_6_5_circular_ascending_node_change():
    dv = change_ascending_node_circular(
        math.radians(55), math.radians(45), 0.745356
    )
    assert dv == pytest.approx(0.4673023, abs=1e-3)


def test_example_6_6_combined_inclination_and_raan_change():
    dv, plane_angle = combined_inclination_raan_change(
        math.radians(55), math.radians(40), math.radians(45), 0.745356
    )
    assert math.degrees(plane_angle) == pytest.approx(35.73709, abs=1e-3)
    assert dv == pytest.approx(0.4574014, abs=1e-3)


def test_example_6_7_combined_maneuver():
    """Reproduce Example 6-7 using the Hohmann and Algorithm 42 helpers."""
    mu = 1.0  # ER^3/TU^2
    r_initial = 1.0 + si_to_canonical(191_000.0, "length")
    r_final = 1.0 + si_to_canonical(35_780_000.0, "length")
    (a_transfer, tof, delta_v_hohmann_a,
     delta_v_hohmann_b) = hohmann_transfer(r_initial, r_final, mu)

    # Algorithm 42 needs the circular speeds; Hohmann's signed burns recover
    # the two transfer-orbit speeds without duplicating vis-viva calculations.
    v_initial = math.sqrt(mu / r_initial)
    v_final = math.sqrt(mu / r_final)
    v_transfer_a = v_initial + delta_v_hohmann_a
    v_transfer_b = v_final - delta_v_hohmann_b

    (delta_i_initial, delta_i_final, delta_v_a, delta_v_b,
     delta_v_total) = minimum_combined_plane_change(
        v_initial, v_transfer_a, v_transfer_b, v_final, math.radians(28.5)
    )

    assert r_initial == pytest.approx(1.029946, abs=1e-3)
    assert r_final == pytest.approx(6.609789, abs=1e-3)
    assert a_transfer == pytest.approx(3.819867, abs=1e-3)
    tof_hours = canonical_to_si(tof, "time") / 3600.0
    assert tof_hours == pytest.approx(5.256, abs=1e-3)

    # Vallado's Algorithm 42 analytical estimate (about 0.5-degree accuracy).
    assert math.degrees(delta_i_initial) == pytest.approx(1.59492, abs=1e-3)
    assert math.degrees(delta_i_final) == pytest.approx(26.90508, abs=1e-3)
    assert delta_v_a == pytest.approx(0.312404, abs=1e-3)
    assert delta_v_b == pytest.approx(0.227974, abs=1e-3)

    # Vallado reports 4271.8902 m/s.
    total_delta_v_km_s = canonical_to_si(delta_v_total, "velocity") / 1000.0
    assert total_delta_v_km_s == pytest.approx(4.2718902, abs=3e-3)


def test_example_6_8_same_orbit_phasing():
    # Vallado's Example 6-8: a=2 ER, target 20 degrees behind, one extra turn.
    tof, dv, a_phase = circular_coplanar_phasing_same_orbit(
        2.0, math.radians(-20), 1, 1, 1.0
    )
    assert tof == pytest.approx(16.7842, abs=1e-3)
    assert a_phase == pytest.approx(1.9252, abs=1e-3)
    assert dv == pytest.approx(2 * 0.013868, abs=1e-3)


def test_example_6_9_different_orbit_phasing():
    # Vallado's Example 6-9: a_int=2 ER, a_tgt=6.61 ER, initial phase=-20 deg.
    tof, dv, a_transfer, wait = circular_coplanar_phasing_different_orbits(
        math.radians(-20), 2.0, 6.61, 1, 1.0
    )
    assert a_transfer == pytest.approx(4.305, abs=1e-3)
    assert tof == pytest.approx(28.0614, abs=1e-3)
    assert dv == pytest.approx(0.2929289, abs=1e-3)
    assert wait == pytest.approx(28.1072, abs=1e-3)


def test_example_6_14_hill_relative_motion():
    # Initial relative position is zero. Vallado's shuttle-bay ejection rates
    # use the negative signs shown in the book's Hill frame convention.
    state_5_min = hill_equations(
        0.0, 0.0, 0.0,
        -0.1, -0.04, -0.02,
        0.0010854, 300.0,
    )
    assert state_5_min == pytest.approx(
        (-33.345, -1.473, -5.894, -0.12, 0.03, -0.02), abs=1e-2
    )
    state_20_min = hill_equations(
        0.0, 0.0, 0.0,
        -0.1, -0.04, -0.02,
        0.0010854, 1200.0,
    )
    assert state_20_min == pytest.approx(
        (-143.000, 137.279, -17.766, -0.10, 0.27, -0.01), abs=1e-2
    )


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

