import math
from util import kepler_universal, rv_to_coe
from util_transformation import (
    eci_to_rsw,
    ntw_to_eci,
    pqw_to_rsw,
    rsw_to_eci,
    rsw_to_sez,
    sez_to_rsw,
)


def arc_integral(p, eccentricity, nu_start, nu_end, *, epsabs=1.49e-8,
                 epsrel=1.49e-8, limit=100):
    """Compute signed orbital arc length between two true anomalies.

    Uses ``scipy.integrate.quad`` on ``ds/dnu`` for an elliptic orbit, where
    ``p`` is the semi-latus rectum and angles are in radians. The signed result
    has distance units consistent with ``p``. ``epsabs`` and ``epsrel`` set
    the absolute and relative error tolerances.
    """
    if (not all(math.isfinite(value) for value in
                (p, eccentricity, nu_start, nu_end, epsabs, epsrel))
            or p <= 0 or not 0 <= eccentricity < 1):
        raise ValueError("arguments must be finite; p must be positive and eccentricity in [0, 1)")
    if epsabs < 0 or epsrel <= 0 or limit < 1:
        raise ValueError("epsabs must be nonnegative, epsrel positive, and limit at least 1")
    try:
        from scipy.integrate import quad
    except ImportError as exc:
        raise ImportError("arc_integral requires SciPy; install scipy to use orbital arc calculations") from exc

    def ds_dnu(anomaly):
        denominator = 1 + eccentricity * math.cos(anomaly)
        return (p * math.sqrt(1 + 2 * eccentricity * math.cos(anomaly)
                              + eccentricity**2) / denominator**2)

    result, _ = quad(ds_dnu, nu_start, nu_end, epsabs=epsabs,
                     epsrel=epsrel, limit=limit)
    return result


def arc_residual(anomaly_change, p, eccentricity, nu_start,
                 direction, target_arc):
    """Return computed orbital arc length minus ``target_arc``.

    ``direction`` must be +1 or -1 and indicates increasing or decreasing
    true anomaly. This residual is suitable for a bracketed scalar root solver.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if target_arc < 0 or not math.isfinite(target_arc):
        raise ValueError("target_arc must be finite and nonnegative")
    arc = arc_integral(p, eccentricity, nu_start,
                       nu_start + direction * anomaly_change,
                       epsabs=1e-10, epsrel=1e-10)
    return abs(arc) - target_arc


def hohmann_transfer(r_initial, r_final, mu):
    """Algorithm 36: calculate a coplanar Hohmann transfer.

    Args:
        r_initial: Initial circular-orbit radius.
        r_final: Final circular-orbit radius.
        mu: Gravitational parameter (distance**3 / time**2).

    Returns:
        (a_trans, tau_trans, delta_v_a, delta_v_b), where ``a_trans`` is the
        transfer ellipse's semi-major axis, ``tau_trans`` is its half-period,
        and the delta-v values are the signed velocity changes at departure
        and arrival. Units must be consistent with ``mu``.
    """
    if (not math.isfinite(r_initial) or not math.isfinite(r_final)
            or not math.isfinite(mu) or r_initial <= 0 or r_final <= 0 or mu <= 0):
        raise ValueError("r_initial, r_final, and mu must be finite and positive")

    a_trans = (r_initial + r_final) / 2

    v_initial = math.sqrt(mu / r_initial)
    v_trans_a = math.sqrt(2 * mu / r_initial - mu / a_trans)
    v_final = math.sqrt(mu / r_final)
    v_trans_b = math.sqrt(2 * mu / r_final - mu / a_trans)

    delta_v_a = v_trans_a - v_initial
    delta_v_b = v_final - v_trans_b
    tau_trans = math.pi * math.sqrt(a_trans**3 / mu)

    return a_trans, tau_trans, delta_v_a, delta_v_b


def bi_elliptic_transfer(r_initial, r_b, r_final, mu):
    """Algorithm 37: calculate a bi-elliptic transfer via ``r_b``.

    Returns ``(a_trans1, a_trans2, tau_trans1, tau_trans2,
    delta_v_a, delta_v_b, delta_v_c, delta_v_total)``. Delta-v values at
    each burn are signed; ``delta_v_total`` is the sum of their magnitudes.
    """
    if (any(not math.isfinite(r) or r <= 0 for r in (r_initial, r_b, r_final))
            or not math.isfinite(mu) or mu <= 0):
        raise ValueError("r_initial, r_b, r_final, and mu must be finite and positive")

    a_trans1 = (r_initial + r_b) / 2
    a_trans2 = (r_b + r_final) / 2
    v_initial = math.sqrt(mu / r_initial)
    v_trans1_a = math.sqrt(2 * mu / r_initial - mu / a_trans1)
    v_trans1_b = math.sqrt(2 * mu / r_b - mu / a_trans1)
    v_trans2_b = math.sqrt(2 * mu / r_b - mu / a_trans2)
    v_trans2_c = math.sqrt(2 * mu / r_final - mu / a_trans2)
    v_final = math.sqrt(mu / r_final)

    delta_v_a = v_trans1_a - v_initial
    delta_v_b = v_trans2_b - v_trans1_b
    delta_v_c = v_final - v_trans2_c
    tau_trans1 = math.pi * math.sqrt(a_trans1**3 / mu)
    tau_trans2 = math.pi * math.sqrt(a_trans2**3 / mu)
    delta_v_total = abs(delta_v_a) + abs(delta_v_b) + abs(delta_v_c)
    return (a_trans1, a_trans2, tau_trans1, tau_trans2,
            delta_v_a, delta_v_b, delta_v_c, delta_v_total)


def one_tangent_burn(r_initial, r_final, nu_trans_b, mu, *, apsis="periapsis",
                     revolutions=0):
    """Algorithm 38: calculate a one-tangent transfer ending at ``nu_trans_b``.

    ``apsis`` selects whether the initial point is transfer periapsis or
    apoapsis. ``revolutions`` adds complete transfer-orbit periods to the
    elapsed time. Returns ``(e_trans, a_trans, tau_trans, delta_v_a,
    delta_v_b, delta_v_total)``. Angles are in radians.
    """
    if (any(not math.isfinite(x) for x in (r_initial, r_final, nu_trans_b, mu))
            or r_initial <= 0 or r_final <= 0 or mu <= 0):
        raise ValueError("radii and mu must be finite and positive; anomaly must be finite")
    if apsis not in ("periapsis", "apoapsis"):
        raise ValueError("apsis must be 'periapsis' or 'apoapsis'")
    if not isinstance(revolutions, int) or revolutions < 0:
        raise ValueError("revolutions must be a nonnegative integer")

    r_inverse = r_initial / r_final
    cos_nu = math.cos(nu_trans_b)
    if apsis == "periapsis":
        e_trans = (r_inverse - 1) / (cos_nu - r_inverse)
        a_trans = r_initial / (1 - e_trans)
        eccentric_anomaly_initial = 0.0
    else:
        e_trans = (r_inverse - 1) / (cos_nu + r_inverse)
        a_trans = r_initial / (1 + e_trans)
        eccentric_anomaly_initial = math.pi
    if not 0 <= e_trans < 1 or a_trans <= 0:
        raise ValueError("these radii, anomaly, and apsis do not define an elliptic transfer")

    v_initial = math.sqrt(mu / r_initial)
    v_trans_a = math.sqrt(2 * mu / r_initial - mu / a_trans)
    v_trans_b = math.sqrt(2 * mu / r_final - mu / a_trans)
    v_final = math.sqrt(mu / r_final)
    delta_v_a = v_trans_a - v_initial

    flight_path_angle = math.atan2(e_trans * math.sin(nu_trans_b),
                                   1 + e_trans * cos_nu)
    delta_v_b = math.sqrt(v_trans_b**2 + v_final**2
                          - 2 * v_trans_b * v_final * math.cos(flight_path_angle))

    eccentric_anomaly = 2 * math.atan2(
        math.sqrt(1 - e_trans) * math.sin(nu_trans_b / 2),
        math.sqrt(1 + e_trans) * math.cos(nu_trans_b / 2))
    mean_anomaly_initial = eccentric_anomaly_initial - e_trans * math.sin(eccentric_anomaly_initial)
    mean_anomaly_final = eccentric_anomaly - e_trans * math.sin(eccentric_anomaly)
    tau_trans = math.sqrt(a_trans**3 / mu) * (
        2 * math.pi * revolutions + mean_anomaly_final - mean_anomaly_initial)
    delta_v_total = abs(delta_v_a) + abs(delta_v_b)
    return e_trans, a_trans, tau_trans, delta_v_a, delta_v_b, delta_v_total


def inclination_only(delta_i, phi_pa, v_initial):
    """Algorithm 39: return delta-v for a pure inclination change (radians)."""
    if not all(math.isfinite(x) for x in (delta_i, phi_pa, v_initial)) or v_initial < 0:
        raise ValueError("angles and velocity must be finite; velocity must be nonnegative")
    return 2 * v_initial * math.cos(phi_pa) * math.sin(delta_i / 2)


def change_ascending_node_circular(inclination_initial, delta_raan, v_initial):
    """Algorithm 40: return delta-v for changing RAAN in a circular orbit.

    Inclination and RAAN change are in radians; the input velocity and result
    use consistent distance/time units.
    """
    if (not all(math.isfinite(x) for x in
                (inclination_initial, delta_raan, v_initial)) or v_initial < 0):
        raise ValueError("angles and velocity must be finite; velocity must be nonnegative")
    cos_theta = (math.cos(inclination_initial)**2
                 + math.sin(inclination_initial)**2 * math.cos(delta_raan))
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    return 2 * v_initial * math.sin(theta / 2)


def combined_inclination_raan_change(inclination_initial, inclination_final,
                                     delta_raan, v_initial):
    """Algorithm 41: delta-v for combined inclination and RAAN changes.

    Angles are in radians. Returns ``(delta_v, theta)``, where ``theta`` is
    the angle between the initial and final orbital planes.
    """
    if (not all(math.isfinite(x) for x in
                (inclination_initial, inclination_final, delta_raan, v_initial))
            or v_initial < 0):
        raise ValueError("angles and velocity must be finite; velocity must be nonnegative")
    cos_theta = (math.cos(inclination_initial) * math.cos(inclination_final)
                 + math.sin(inclination_initial) * math.sin(inclination_final)
                 * math.cos(delta_raan))
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    delta_v = 2 * v_initial * math.sin(theta / 2)
    return delta_v, theta


def minimum_combined_plane_change(v_initial, v_trans_a, v_trans_b,
                                  v_final, delta_i):
    """Algorithm 42: minimize delta-v by splitting a plane change.

    ``delta_i`` is the total plane change in radians. Returns
    ``(delta_i_initial, delta_i_final, delta_v_initial, delta_v_final,
    delta_v_total)``. Uses a bounded golden-section minimization over the
    fraction of plane change applied at the initial burn.
    """
    speeds = (v_initial, v_trans_a, v_trans_b, v_final)
    if (not all(math.isfinite(x) and x >= 0 for x in speeds)
            or not math.isfinite(delta_i) or not 0 <= delta_i <= math.pi):
        raise ValueError("speeds must be finite and nonnegative; delta_i must be in [0, pi]")

    def costs(fraction):
        initial_angle = fraction * delta_i
        final_angle = (1 - fraction) * delta_i
        dv_initial = math.sqrt(max(0.0, v_initial**2 + v_trans_a**2
                                  - 2 * v_initial * v_trans_a * math.cos(initial_angle)))
        dv_final = math.sqrt(max(0.0, v_final**2 + v_trans_b**2
                                - 2 * v_final * v_trans_b * math.cos(final_angle)))
        return dv_initial + dv_final, dv_initial, dv_final

    # Golden-section search finds the minimum over the full allocation interval.
    left, right = 0.0, 1.0
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = costs(x1)[0], costs(x2)[0]
    for _ in range(100):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = costs(x1)[0]
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = costs(x2)[0]
    fraction = (left + right) / 2
    total, dv_initial, dv_final = costs(fraction)
    return fraction * delta_i, (1 - fraction) * delta_i, dv_initial, dv_final, total


def fixed_delta_v_maneuver(v_initial, v_final, v_trans_a, delta_v,
                            inclination_direction="increasing"):
    """Algorithm 43: derive flight-path and inclination angles for fixed Δv.

    ``inclination_direction`` is ``"increasing"`` or ``"decreasing"``.
    Returns ``(gamma, delta_i_1)`` in radians.
    """
    if (not all(math.isfinite(x) and x >= 0 for x in
                (v_initial, v_final, v_trans_a, delta_v)) or delta_v == 0
            or v_initial == 0 or v_trans_a == 0):
        raise ValueError("velocities must be finite and nonnegative; v_initial, v_trans_a, and delta_v must be positive")
    if inclination_direction not in ("increasing", "decreasing"):
        raise ValueError("inclination_direction must be 'increasing' or 'decreasing'")
    cos_gamma = (v_initial**2 + delta_v**2 - v_final**2) / (2 * v_initial * delta_v)
    cos_delta_i = (v_initial**2 + v_trans_a**2 - delta_v**2) / (2 * v_initial * v_trans_a)
    if not -1 <= cos_gamma <= 1 or not -1 <= cos_delta_i <= 1:
        raise ValueError("the supplied speeds and delta-v do not define a physical maneuver")
    gamma = math.acos(cos_gamma)
    if inclination_direction == "decreasing":
        gamma = -gamma
    return gamma, math.acos(cos_delta_i)


def circular_coplanar_phasing_same_orbit(a_tgt, theta, k_tgt, k_int, mu):
    """Algorithm 44: calculate phasing time and orbit for the same circular orbit.

    ``theta`` is the desired angular separation in radians. Returns
    ``(tau_phase, delta_v, a_phase)``.
    """
    if (not all(math.isfinite(x) for x in (a_tgt, theta, mu))
            or a_tgt <= 0 or mu <= 0 or k_tgt < 0 or k_int <= 0):
        raise ValueError("a_tgt and mu must be positive; revolution counts must be valid")
    omega_tgt = math.sqrt(mu / a_tgt**3)
    tau_phase = (2 * math.pi * k_tgt + theta) / omega_tgt
    if tau_phase <= 0:
        raise ValueError("the requested phasing time must be positive")
    a_phase = (mu * (tau_phase / (2 * math.pi * k_int))**2)**(1 / 3)
    delta_v = 2 * abs(math.sqrt(2 * mu / a_tgt - mu / a_phase)
                      - math.sqrt(mu / a_tgt))
    return tau_phase, delta_v, a_phase


def circular_coplanar_phasing_different_orbits(theta_i, a_int, a_tgt, k, mu):
    """Algorithm 45: calculate phasing between different circular orbits.

    ``theta_i`` is the initial phase angle in radians. Returns
    ``(tau_trans, delta_v, a_trans, tau_wait)``.
    """
    if (not all(math.isfinite(x) for x in (theta_i, a_int, a_tgt, mu))
            or a_int <= 0 or a_tgt <= 0 or mu <= 0 or k < 0):
        raise ValueError("orbit radii and mu must be positive and k nonnegative")
    omega_tgt = math.sqrt(mu / a_tgt**3)
    omega_int = math.sqrt(mu / a_int**3)
    a_trans = (a_int + a_tgt) / 2
    tau_trans = math.pi * math.sqrt(a_trans**3 / mu)
    theta = omega_tgt * tau_trans - math.pi
    relative_rate = omega_int - omega_tgt
    if relative_rate == 0:
        raise ValueError("the two orbits have no relative angular rate")
    tau_wait = (theta - theta_i + 2 * math.pi * k) / relative_rate
    delta_v = (abs(math.sqrt(2 * mu / a_int - mu / a_trans) - math.sqrt(mu / a_int))
               + abs(math.sqrt(2 * mu / a_tgt - mu / a_trans) - math.sqrt(mu / a_tgt)))
    return tau_trans, delta_v, a_trans, tau_wait


def noncoplanar_phasing(theta_i, a_int, a_tgt, k_tgt, u_int, raan_int,
                        lambda_true0, delta_i, mu, *, k_int=1):
    """Algorithm 46: calculate a noncoplanar phasing maneuver.

    All angles are in radians. ``u_int`` is the interceptor argument of
    latitude and ``lambda_true0`` is the target true longitude at the initial
    epoch. Returns ``(tau_trans, tau_phase, delta_v_phase, delta_v_trans1,
    delta_v_trans2, a_phase, tau_total)``. ``k_int`` is the phasing-orbit
    revolution count (the figure's equation uses it although its input line
    omits it).
    """
    values = (theta_i, a_int, a_tgt, u_int, raan_int, lambda_true0, delta_i, mu)
    if (not all(math.isfinite(x) for x in values) or a_int <= 0 or a_tgt <= 0
            or mu <= 0 or k_tgt < 0 or k_int <= 0):
        raise ValueError("angles must be finite; orbit radii and mu positive; revolution counts valid")

    omega_tgt = math.sqrt(mu / a_tgt**3)
    omega_int = math.sqrt(mu / a_int**3)
    a_trans = (a_int + a_tgt) / 2
    tau_trans = math.pi * math.sqrt(a_trans**3 / mu)
    alpha_l = omega_tgt * tau_trans

    # The next ascending or descending node is the first of u = 0 or pi ahead.
    node_offsets = ((-u_int) % (2 * math.pi), (math.pi - u_int) % (2 * math.pi))
    delta_u_int = min(node_offsets)
    delta_t_node = delta_u_int / omega_int
    lambda_true_tgt1 = lambda_true0 + omega_tgt * delta_t_node
    lambda_true_int1 = raan_int + math.pi
    theta_new = lambda_true_int1 - lambda_true_tgt1
    alpha_new = math.pi + theta_new

    tau_phase = (alpha_new - alpha_l + 2 * math.pi * k_tgt) / omega_tgt
    if tau_phase <= 0:
        raise ValueError("selected k_tgt must make the phasing time positive")
    a_phase = (mu * (tau_phase / (2 * math.pi * k_int))**2)**(1 / 3)

    v_int = math.sqrt(mu / a_int)
    v_phase_at_int = math.sqrt(2 * mu / a_int - mu / a_phase)
    delta_v_phase = abs(v_phase_at_int - v_int)
    v_trans_at_int = math.sqrt(2 * mu / a_int - mu / a_trans)
    delta_v_trans1 = abs(v_trans_at_int - v_phase_at_int)

    v_trans_at_tgt = math.sqrt(2 * mu / a_tgt - mu / a_trans)
    v_tgt = math.sqrt(mu / a_tgt)
    delta_v_trans2 = math.sqrt(max(0.0, v_trans_at_tgt**2 + v_tgt**2
                                    - 2 * v_trans_at_tgt * v_tgt * math.cos(delta_i)))
    tau_total = 2 * math.pi * math.sqrt(a_phase**3 / mu) + tau_trans + delta_t_node
    return (tau_trans, tau_phase, delta_v_phase, delta_v_trans1,
            delta_v_trans2, a_phase, tau_total)

def low_thrust_transfer(r_target0, v_target0, r_interceptor0, v_interceptor0,
                        a_final, i_final, isp, thrust_force, mass_initial, mu,
                        lambda_value, n_common, *, steps=10, g0=9.80665,
                        scale="short", max_steps=100000,
                        semi_major_axis_tolerance=1e-3,
                        inclination_tolerance=1e-5):
    """Algorithm 47: numerically propagate a low-thrust orbit transfer.

    ``lambda_value`` and the unit ``n_common`` common-node vector are supplied
    by the user because the source algorithm says to find them iteratively but
    does not give their iteration procedure. ``scale`` selects its short- or
    long-scale ``c_v`` approximation. Inputs use consistent orbital units;
    ``g0`` must use the acceleration units consistent with ``isp`` and thrust.
    Returns ``(transfer_time, accumulated_delta_v)``.
    """
    low_thrust_alpha = (
        0.0, 2.467410607, -1.907470562, 35.892442177, -214.672979624,
        947.773273608, -2114.861134906, 2271.240058672, -1127.457440108,
        192.953875268, 8.577733773,
    )
    low_thrust_beta = (
        1.0, 0.4609698838, 13.7756315324, -69.1245316678, 279.0671832500,
        -397.6628952136, -70.0139935047, 528.0334266841, -324.9303836520,
        20.5838245170, 18.8165370778,
    )

    if (steps <= 0 or max_steps <= 0 or isp <= 0 or thrust_force <= 0
            or mass_initial <= 0 or mu <= 0 or g0 <= 0 or lambda_value <= 0):
        raise ValueError("steps, max_steps, isp, thrust, mass, mu, g0, and lambda_value must be positive")
    if scale not in ("short", "long"):
        raise ValueError("scale must be 'short' or 'long'")
    if len(n_common) != 3:
        raise ValueError("n_common must be a 3-vector")
    norm_common = math.sqrt(sum(x*x for x in n_common))
    if norm_common == 0:
        raise ValueError("n_common must be nonzero")
    n_common = tuple(x / norm_common for x in n_common)

    r_target, v_target = tuple(r_target0), tuple(v_target0)
    r_interceptor, v_interceptor = tuple(r_interceptor0), tuple(v_interceptor0)
    if any(len(state) != 3 for state in
           (r_target, v_target, r_interceptor, v_interceptor)):
        raise ValueError("target and interceptor position/velocity inputs must be 3-vectors")
    distance_unit = math.sqrt(sum(x*x for x in r_target))
    if distance_unit == 0:
        raise ValueError("initial target position must be nonzero")
    mass = mass_initial
    elapsed = 0.0
    delta_v_total = 0.0

    def polynomial(coefficients, z):
        result = 0.0
        for coefficient in reversed(coefficients):
            result = result * z + coefficient
        return result

    for _ in range(max_steps):
        a_current, _, _, inclination_current, _, _, _ = rv_to_coe(
            r_interceptor, v_interceptor, mu)
        if (abs(a_current - a_final) <= semi_major_axis_tolerance
                and abs(inclination_current - i_final) <= inclination_tolerance):
            return elapsed, delta_v_total
        if not math.isfinite(a_current) or a_current <= 0:
            raise ValueError("low-thrust transfer entered a non-elliptic orbit")

        period = 2 * math.pi * math.sqrt(a_current**3 / mu)
        delta_t = period / steps
        x = (math.pi / (2 * lambda_value)
             * math.sqrt(distance_unit / a_current))
        a_current_du = a_current / distance_unit
        if scale == "short":
            c_v = 1 / (4 * lambda_value**2 * a_current_du**2 + 1)
        else:
            z = x**2
            denominator = polynomial(low_thrust_beta, z)
            if denominator == 0:
                raise ValueError("long-scale polynomial denominator is zero")
            c_v = polynomial(low_thrust_alpha, z) / denominator
        if not 0 < c_v <= 1:
            raise ValueError("the c_v approximation must evaluate to a value in (0, 1]")

        radius = math.sqrt(sum(x*x for x in r_interceptor))
        radial_cosine = sum(r_interceptor[i] * n_common[i] for i in range(3)) / radius
        radial_cosine = max(-1.0, min(1.0, radial_cosine))
        phi_c = math.atan2(radial_cosine, math.sqrt(max(0.0, 1 / c_v - 1)))
        mass_flow = thrust_force / (isp * g0)
        mass_next = mass - mass_flow * delta_t
        if mass_next <= 0:
            raise ValueError("spacecraft mass became nonpositive before reaching the target orbit")
        acceleration = thrust_force / mass
        _, _, _, target_inclination, target_raan, target_argp, target_nu = rv_to_coe(
            r_target, v_target, mu)
        thrust_eci = ntw_to_eci((acceleration, 0.0, 0.0), target_raan,
                                target_inclination,
                                (target_argp + target_nu) % (2 * math.pi), phi_c)

        # Follow the book's split step: two-body propagation, then thrust dv.
        r_interceptor, v_interceptor = kepler_universal(
            r_interceptor, v_interceptor, delta_t, mu)
        v_interceptor = tuple(v_interceptor[i] + thrust_eci[i] * delta_t
                              for i in range(3))
        r_target, v_target = kepler_universal(r_target, v_target, delta_t, mu)
        mass = mass_next
        elapsed += delta_t
        delta_v_total += acceleration * delta_t

    raise RuntimeError("low-thrust transfer did not reach the target orbit within max_steps")

def hill_equations(x0, y0, z0, xdot0, ydot0, zdot0, omega, delta_t):
    """Algorithm 48: propagate relative position/velocity with Hill's equations.

    Inputs use consistent distance and time units; ``omega`` is the target
    orbit's angular rate. Returns ``(x, y, z, xdot, ydot, zdot)``.
    """
    if not all(math.isfinite(v) for v in
               (x0, y0, z0, xdot0, ydot0, zdot0, omega, delta_t)) or omega < 0:
        raise ValueError("state, omega, and delta_t must be finite; omega must be nonnegative")
    c = math.cos(omega * delta_t)
    s = math.sin(omega * delta_t)
    x = (xdot0 / omega * s - (3 * x0 + 2 * ydot0 / omega) * c
         + 4 * x0 + 2 * ydot0 / omega) if omega else x0 + xdot0 * delta_t
    y = ((6 * x0 + 4 * ydot0 / omega) * s + 2 * xdot0 / omega * c
         - (6 * omega * x0 + 3 * ydot0) * delta_t
         + y0 - 2 * xdot0 / omega) if omega else y0 + ydot0 * delta_t
    z = (z0 * c + zdot0 / omega * s) if omega else z0 + zdot0 * delta_t
    xdot = (xdot0 * c + (3 * omega * x0 + 2 * ydot0) * s
            if omega else xdot0)
    ydot = ((6 * omega * x0 + 4 * ydot0) * c - 2 * xdot0 * s
            - (6 * omega * x0 + 3 * ydot0) if omega else ydot0)
    zdot = (-z0 * omega * s + zdot0 * c if omega else zdot0)
    return x, y, z, xdot, ydot, zdot


def hill_eqcm_to_eci(r_target_eci, v_target_eci, x, y, z,
                     xdot, ydot, zdot, mu):
    """Algorithm 49: convert Hill EQCM state to interceptor ECI state.

    ``r_target_eci`` and ``v_target_eci`` are the target's ECI state;
    ``x, y, z, xdot, ydot, zdot`` are its relative Hill EQCM state. This
    implementation interprets ``y`` as signed orbital arc length from the
    target's initial true anomaly and numerically solves for the ending
    anomaly. Returns ``(r_interceptor_eci, v_interceptor_eci)``.
    """
    if len(r_target_eci) != 3 or len(v_target_eci) != 3:
        raise ValueError("target position and velocity must be 3-vectors")
    values = (*r_target_eci, *v_target_eci, x, y, z, xdot, ydot, zdot, mu)
    if not all(math.isfinite(value) for value in values) or mu <= 0:
        raise ValueError("state components must be finite and mu must be positive")

    a, p, eccentricity, inclination, raan, argp, nu_current = rv_to_coe(
        r_target_eci, v_target_eci, mu)
    if not math.isfinite(a) or a <= 0 or p <= 0 or eccentricity >= 1:
        raise ValueError("Algorithm 49 requires an elliptic target orbit")
    argument_latitude = (argp + nu_current) % (2 * math.pi)
    r_target_rsw1 = eci_to_rsw(r_target_eci, raan, inclination, argument_latitude)
    v_target_rsw1 = eci_to_rsw(v_target_eci, raan, inclination, argument_latitude)

    # Eccentricity-vector components in RSW are (e cos(nu), -e sin(nu), 0).
    if eccentricity <= 1e-12:
        nu1 = nu_current
    else:
        e_r = ((sum(v*v for v in v_target_rsw1) - mu / r_target_rsw1[0])
               * r_target_rsw1[0] / mu
               - (r_target_rsw1[0] * v_target_rsw1[0]
                  + r_target_rsw1[1] * v_target_rsw1[1]
                  + r_target_rsw1[2] * v_target_rsw1[2])
               * v_target_rsw1[0] / mu)
        e_s = (-sum(r_target_rsw1[i] * v_target_rsw1[i] for i in range(3))
               * v_target_rsw1[1] / mu)
        nu1 = (-math.atan2(e_s, e_r)) % (2 * math.pi)

    circumference = arc_integral(p, eccentricity, nu1, nu1 + 2 * math.pi)
    complete_revolutions = int(abs(y) // circumference)
    remaining_arc = abs(y) - complete_revolutions * circumference
    direction = 1 if y >= 0 else -1
    if remaining_arc == 0:
        residual_anomaly = 0.0
    else:
        try:
            from scipy.optimize import brentq
        except ImportError as exc:
            raise ImportError(
                "hill_eqcm_to_eci requires SciPy to solve for the ending anomaly"
            ) from exc

        residual_anomaly = brentq(
            arc_residual, 0.0, 2 * math.pi,
            args=(p, eccentricity, nu1, direction, remaining_arc),
            xtol=1e-12, rtol=1e-12)
    anomaly_change = direction * (complete_revolutions * 2 * math.pi
                                  + residual_anomaly)
    nu2 = nu1 + anomaly_change

    denominator = 1 + eccentricity * math.cos(nu2)
    rmag_target2 = p / denominator
    root_mu_p = math.sqrt(mu / p)
    r_target_pqw2 = (rmag_target2 * math.cos(nu2),
                     rmag_target2 * math.sin(nu2), 0.0)
    v_target_pqw2 = (-root_mu_p * math.sin(nu2),
                     root_mu_p * (eccentricity + math.cos(nu2)), 0.0)
    r_target_rsw2 = pqw_to_rsw(r_target_pqw2, nu2)
    v_target_rsw2 = pqw_to_rsw(v_target_pqw2, nu2)

    if abs(z) > rmag_target2:
        raise ValueError("cross-track offset z exceeds the target radius")
    delta_phi = math.asin(z / rmag_target2)
    delta_lambda = nu2 - nu1

    # Resolve the interceptor direction in RSW from latitude/longitude offsets.
    rhat_interceptor_rsw1 = (
        math.cos(delta_phi) * math.cos(delta_lambda),
        math.cos(delta_phi) * math.sin(delta_lambda),
        math.sin(delta_phi),
    )
    rhat_interceptor_sez = rsw_to_sez(
        rhat_interceptor_rsw1, delta_phi, delta_lambda)
    if abs(rhat_interceptor_sez[2]) <= 1e-12:
        raise ValueError("Algorithm 49 geometry gives a zero SEZ-Z direction cosine")
    r_scale = (x + r_target_rsw2[0]) / rhat_interceptor_sez[2]
    r_interceptor_rsw1 = tuple(r_scale * component
                               for component in rhat_interceptor_rsw1)

    radius_target1 = math.sqrt(sum(component**2 for component in r_target_rsw1))
    omega1 = v_target_rsw1[1] / radius_target1
    lambda_dot = (ydot + omega1 * radius_target1) / p
    phi_dot = zdot / p
    v_interceptor_sez = (
        -phi_dot * r_scale,
        lambda_dot * r_scale * math.cos(delta_phi),
        zdot + v_target_rsw2[2],
    )
    v_interceptor_rsw1 = sez_to_rsw(
        v_interceptor_sez, delta_phi, delta_lambda)

    r_interceptor_eci = rsw_to_eci(
        r_interceptor_rsw1, raan, inclination, argument_latitude)
    v_interceptor_eci = rsw_to_eci(
        v_interceptor_rsw1, raan, inclination, argument_latitude)
    return r_interceptor_eci, v_interceptor_eci


def eci_to_hill_eqcm(r_target_eci, v_target_eci,
                     r_interceptor_eci, v_interceptor_eci, mu):
    """Algorithm 50: convert target/interceptor ECI states to Hill EQCM.

    Returns ``(x, y, z, xdot, ydot, zdot)``. The along-track coordinate
    ``y`` is the signed arc length along the target's orbit between the
    target's current anomaly and the interceptor's longitude-like location.
    The arc is evaluated numerically because the target orbit may be elliptic.
    """
    if any(len(state) != 3 for state in
           (r_target_eci, v_target_eci, r_interceptor_eci, v_interceptor_eci)):
        raise ValueError("target and interceptor positions and velocities must be 3-vectors")
    values = (*r_target_eci, *v_target_eci, *r_interceptor_eci,
              *v_interceptor_eci, mu)
    if not all(math.isfinite(value) for value in values) or mu <= 0:
        raise ValueError("state components must be finite and mu must be positive")

    a, p, eccentricity, inclination, raan, argp, nu_current = rv_to_coe(
        r_target_eci, v_target_eci, mu)
    if not math.isfinite(a) or a <= 0 or p <= 0 or eccentricity >= 1:
        raise ValueError("Algorithm 50 requires an elliptic target orbit")
    argument_latitude = (argp + nu_current) % (2 * math.pi)

    target_r_rsw1 = eci_to_rsw(
        r_target_eci, raan, inclination, argument_latitude)
    target_v_rsw1 = eci_to_rsw(
        v_target_eci, raan, inclination, argument_latitude)
    interceptor_r_rsw1 = eci_to_rsw(
        r_interceptor_eci, raan, inclination, argument_latitude)
    interceptor_v_rsw1 = eci_to_rsw(
        v_interceptor_eci, raan, inclination, argument_latitude)

    # The target eccentricity vector in its instantaneous RSW frame is
    # (e cos(nu), -e sin(nu), 0), as used by the algorithm's perigee angle.
    if eccentricity <= 1e-12:
        lambda_perigee = -nu_current
    else:
        e_r = eccentricity * math.cos(nu_current)
        e_s = -eccentricity * math.sin(nu_current)
        lambda_perigee = math.atan2(e_s, e_r)
    nu1 = -lambda_perigee

    # Longitude offset of point 2 in target RSW1; atan2 retains its quadrant.
    delta_lambda = math.atan2(interceptor_r_rsw1[1], interceptor_r_rsw1[0])
    nu2 = delta_lambda - lambda_perigee

    radius_target2 = p / (1 + eccentricity * math.cos(nu2))
    root_mu_p = math.sqrt(mu / p)
    target_r_pqw2 = (radius_target2 * math.cos(nu2),
                     radius_target2 * math.sin(nu2), 0.0)
    target_v_pqw2 = (-root_mu_p * math.sin(nu2),
                     root_mu_p * (eccentricity + math.cos(nu2)), 0.0)
    target_r_rsw2 = pqw_to_rsw(target_r_pqw2, nu2)
    target_v_rsw2 = pqw_to_rsw(target_v_pqw2, nu2)

    # The interceptor's W component supplies the cross-track geometry.
    cross_track_ratio = interceptor_r_rsw1[2] / radius_target2
    if abs(cross_track_ratio) > 1:
        raise ValueError("interceptor cross-track position exceeds the target radius")
    delta_phi = math.asin(cross_track_ratio)

    interceptor_r_sez2 = rsw_to_sez(
        interceptor_r_rsw1, delta_phi, delta_lambda)
    interceptor_v_sez2 = rsw_to_sez(
        interceptor_v_rsw1, delta_phi, delta_lambda)

    # Arc length along the target conic from point 1 to point 2 (arc_1).
    arc1 = arc_integral(p, eccentricity, nu1, nu2)

    x = interceptor_r_sez2[2] - target_r_rsw2[0]
    y = arc1
    z = delta_phi * radius_target2

    target_radius1 = math.sqrt(sum(component**2 for component in target_r_rsw1))
    interceptor_radius2 = math.sqrt(sum(component**2 for component in interceptor_r_sez2))
    omega1 = target_v_rsw1[1] / target_radius1
    lambda_dot = (interceptor_v_sez2[1]
                  / (interceptor_radius2 * math.cos(delta_phi)))
    phi_dot = -interceptor_v_sez2[0] / interceptor_radius2

    xdot = interceptor_v_sez2[2] - target_v_rsw2[0]
    ydot = lambda_dot * radius_target2 - omega1 * target_radius1
    zdot = phi_dot * radius_target2
    return x, y, z, xdot, ydot, zdot