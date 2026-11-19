import math


CANONICAL_EARTH_RADIUS_KM = 6378.137
CANONICAL_EARTH_MU_KM3_S2 = 398600.4418


def _canonical_si_scale(quantity, radius_km, mu_km3_s2):
    """Return the SI scale for one ER/TU-derived quantity."""
    if not isinstance(quantity, str):
        raise ValueError("quantity must be a supported string")
    if (not math.isfinite(radius_km) or radius_km <= 0
            or not math.isfinite(mu_km3_s2) or mu_km3_s2 <= 0):
        raise ValueError("radius_km and mu_km3_s2 must be finite and positive")

    distance_m = radius_km * 1000.0
    mu_m3_s2 = mu_km3_s2 * 1e9
    time_s = math.sqrt(distance_m**3 / mu_m3_s2)
    scales = {
        "length": distance_m,
        "area": distance_m**2,
        "volume": distance_m**3,
        "time": time_s,
        "frequency": 1 / time_s,
        "velocity": distance_m / time_s,
        "acceleration": distance_m / time_s**2,
        "jerk": distance_m / time_s**3,
        "gravitational_parameter": distance_m**3 / time_s**2,
        "specific_energy": distance_m**2 / time_s**2,
    }
    try:
        return scales[quantity]
    except KeyError as exc:
        supported = ", ".join(scales)
        raise ValueError(f"unsupported quantity {quantity!r}; choose from {supported}") from exc


def canonical_to_si(value, quantity, *, radius_km=CANONICAL_EARTH_RADIUS_KM,
                    mu_km3_s2=CANONICAL_EARTH_MU_KM3_S2):
    """Convert an ER/TU canonical value to SI units.

    ``quantity`` is one of ``length``, ``area``, ``volume``, ``time``,
    ``frequency``, ``velocity``, ``acceleration``, ``jerk``,
    ``gravitational_parameter``, or ``specific_energy``. The default scales
    use Earth's equatorial radius and gravitational parameter. The returned
    SI units are respectively m, m², m³, s, s⁻¹, m/s, m/s², m/s³, m³/s²,
    and m²/s².
    """
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value * _canonical_si_scale(quantity, radius_km, mu_km3_s2)


def si_to_canonical(value, quantity, *, radius_km=CANONICAL_EARTH_RADIUS_KM,
                    mu_km3_s2=CANONICAL_EARTH_MU_KM3_S2):
    """Convert an SI value to ER/TU canonical units; see ``canonical_to_si``."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value / _canonical_si_scale(quantity, radius_km, mu_km3_s2)


def find_c2_c3(psi):
    """Algorithm 1: Return c2 and c3 using Vallado's."""
    if psi > 1e-6:
        root = math.sqrt(psi)
        c2 = (1 - math.cos(root)) / psi
        c3 = (root - math.sin(root)) / root**3
    elif psi < -1e-6:
        root = math.sqrt(-psi)
        c2 = (1 - math.cosh(root)) / psi
        c3 = (math.sinh(root) - root) / root**3
    else:
        c2, c3 = 0.5, 1 / 6

    return c2, c3


def kep_eqn_e(M, e, tolerance=1e-8, max_iterations=100):
    """Algorithm 2: solve Kepler's equation for eccentric anomaly E."""
    if not 0 <= e < 1:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    E = M - e if (-math.pi < M < 0 or M > math.pi) else M + e
    for _ in range(max_iterations):
        next_E = E + (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        if abs(next_E - E) < tolerance:
            return next_E
        E = next_E

    raise RuntimeError("Kepler equation did not converge")


def kep_eqn_p(delta_t, p, mu):
    """Algorithm 3: Solve the parabolic Kepler equation for B.

    Args:
        delta_t: Time since periapsis, in time units consistent with ``mu``.
        p: Semi-latus rectum, in distance units.
        mu: Gravitational parameter (distance**3 / time**2).
    """
    if p <= 0 or mu <= 0:
        raise ValueError("p and mu must be positive")

    n_p = 2 * math.sqrt(mu / p**3)
    q = n_p * delta_t
    return 2 * math.sinh(math.asinh(1.5 * q) / 3)


def kep_eqn_h(M, e, tolerance=1e-8, max_iterations=100):
    """Algorithm 4: solve the hyperbolic Kepler equation for H."""
    if e <= 1:
        raise ValueError("hyperbolic eccentricity must be greater than 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if e < 1.6:
        H = M - e if (-math.pi < M < 0 or M > math.pi) else M + e
    elif e < 3.6 and abs(M) > math.pi:
        H = M - math.copysign(e, M)
    else:
        H = M / (e - 1)

    for _ in range(max_iterations):
        next_H = H + (M - e * math.sinh(H) + H) / (e * math.cosh(H) - 1)
        if abs(next_H - H) < tolerance:
            return next_H
        H = next_H

    raise RuntimeError("Hyperbolic Kepler equation did not converge")


def nu_to_anomaly(nu, e):
    """Algorithm 5: convert true anomaly to E, B, or H."""
    if e < 1:
        E = math.atan2(math.sqrt(1 - e**2) * math.sin(nu), e + math.cos(nu))
        return E % (2 * math.pi)
    if math.isclose(e, 1.0, abs_tol=1e-12):
        return math.tan(nu / 2)
    sinh_H = math.sqrt(e**2 - 1) * math.sin(nu) / (1 + e * math.cos(nu))
    return math.asinh(sinh_H)


def anomaly_to_nu(anomaly, e, *, p=None, r=None):
    """Algorithm 6: convert E, B, or H to true anomaly; parabola needs p and r."""
    if e < 1:
        sin_nu = math.sqrt(1 - e**2) * math.sin(anomaly) / (1 - e * math.cos(anomaly))
        cos_nu = (math.cos(anomaly) - e) / (1 - e * math.cos(anomaly))
    elif math.isclose(e, 1.0, abs_tol=1e-12):
        if p is None or r is None or p <= 0 or r <= 0:
            raise ValueError("parabolic conversion requires positive p and r")
        sin_nu, cos_nu = p * anomaly / r, (p - r) / r
    else:
        sin_nu = -math.sqrt(e**2 - 1) * math.sinh(anomaly) / (1 - e * math.cosh(anomaly))
        cos_nu = (math.cosh(anomaly) - e) / (1 - e * math.cosh(anomaly))
    return math.atan2(sin_nu, cos_nu) % (2 * math.pi)


def kepler_coe(r0, v0, delta_t, mu):
    """Algorithm 7: propagate a two-body state using classical orbital elements."""
    a, p, e, inc, raan, argp, nu0 = rv_to_coe(r0, v0, mu)
    if e < 1 - 1e-10:
        E0 = nu_to_anomaly(nu0, e)
        M = E0 - e*math.sin(E0) + math.sqrt(mu/a**3)*delta_t
        nu = anomaly_to_nu(kep_eqn_e(M, e), e)
    elif e > 1 + 1e-10:
        H0 = nu_to_anomaly(nu0, e)
        M = e*math.sinh(H0) - H0 + math.sqrt(mu/(-a)**3)*delta_t
        nu = anomaly_to_nu(kep_eqn_h(M, e), e)
    else:
        B0 = nu_to_anomaly(nu0, e)
        M = B0 + B0**3/3 + 2*math.sqrt(mu/p**3)*delta_t
        B = 2*math.sinh(math.asinh(1.5*M)/3)
        nu = anomaly_to_nu(B, e, p=p, r=p/(1+math.cos(nu0)))
    return coe_to_rv(p, e, inc, raan, argp, nu, mu)


def kepler_universal(r0, v0, delta_t, mu, tolerance=1e-6, max_iterations=1000):
    """Algorithm 8: propagate a two-body state with universal variables."""
    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    mag = lambda a: math.sqrt(dot(a, a))
    r0, v0 = tuple(map(float, r0)), tuple(map(float, v0))
    rmag, v2 = mag(r0), dot(v0, v0)
    if rmag == 0 or mu <= 0:
        raise ValueError("initial radius and mu must be positive")
    vr = dot(r0, v0) / rmag
    alpha = 2/rmag - v2/mu
    chi = math.sqrt(mu)*abs(alpha)*delta_t if abs(alpha) > 1e-8 else math.sqrt(mu)*delta_t/rmag

    for _ in range(max_iterations):
        z = alpha*chi**2
        c2, c3 = find_c2_c3(z)
        F = (rmag*vr/math.sqrt(mu)*chi**2*c2 + (1-alpha*rmag)*chi**3*c3
             + rmag*chi - math.sqrt(mu)*delta_t)
        dF = (rmag*vr/math.sqrt(mu)*chi*(1-z*c3) + (1-alpha*rmag)*chi**2*c2 + rmag)
        step = F/dF
        chi -= step
        if abs(step) < tolerance:
            break
    else:
        raise RuntimeError("Universal Kepler propagation did not converge")

    z = alpha*chi**2
    c2, c3 = find_c2_c3(z)
    f = 1 - chi**2/rmag*c2
    g = delta_t - chi**3/math.sqrt(mu)*c3
    r = tuple(f*r0[i] + g*v0[i] for i in range(3))
    rmag_new = mag(r)
    fdot = math.sqrt(mu)/(rmag_new*rmag)*(alpha*chi**3*c3-chi)
    gdot = 1 - chi**2/rmag_new*c2
    v = tuple(fdot*r0[i] + gdot*v0[i] for i in range(3))
    return r, v


def rv_to_coe(r, v, mu):
    """Algorithm 9: convert inertial state vectors to classical orbital elements."""
    r = tuple(map(float, r))
    v = tuple(map(float, v))
    if len(r) != 3 or len(v) != 3 or mu <= 0:
        raise ValueError("r and v must be 3-vectors and mu must be positive")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    mag = lambda a: math.sqrt(dot(a, a))
    rm, vm = mag(r), mag(v)
    h = cross(r, v)
    hm = mag(h)
    if rm == 0 or hm == 0:
        raise ValueError("position and angular momentum must be nonzero")
    n = (-h[1], h[0], 0.0)
    nm = mag(n)
    ev = tuple(((vm**2 - mu/rm)*r[i] - dot(r, v)*v[i]) / mu for i in range(3))
    e = mag(ev)
    energy = vm**2 / 2 - mu / rm
    a = math.inf if abs(energy) < 1e-14 else -mu / (2 * energy)
    p = hm**2 / mu
    inc = math.acos(max(-1.0, min(1.0, h[2] / hm)))
    raan = math.atan2(n[1], n[0]) % (2*math.pi) if nm else 0.0
    if e > 1e-12 and nm:
        argp = math.atan2(dot(cross(n, ev), h)/(nm*e*hm), dot(n, ev)/(nm*e)) % (2*math.pi)
        nu = math.atan2(dot(cross(ev, r), h)/(e*rm*hm), dot(ev, r)/(e*rm)) % (2*math.pi)
    elif e > 1e-12:
        argp = math.atan2(ev[1], ev[0]) % (2*math.pi)
        nu = math.atan2(dot(cross(ev, r), h)/(e*rm*hm), dot(ev, r)/(e*rm)) % (2*math.pi)
    elif nm:
        argp = 0.0
        nu = math.atan2(dot(cross(n, r), h)/(nm*rm*hm), dot(n, r)/(nm*rm)) % (2*math.pi)
    else:
        argp = 0.0
        nu = math.atan2(r[1], r[0]) % (2*math.pi)
    return a, p, e, inc, raan, argp, nu


def coe_to_rv(p, e, inc, raan, argp, nu, mu, *, u=None,
              lambda_true=None, omega_true=None):
    """Algorithm 10: convert orbital elements to inertial position and velocity.

    Angles are in radians. For singular orbits, optionally pass argument of
    latitude ``u``, true longitude ``lambda_true``, or longitude of periapsis
    ``omega_true`` as appropriate.
    """
    if p <= 0 or e < 0 or mu <= 0:
        raise ValueError("p and mu must be positive, and e must be nonnegative")

    circular = e <= 1e-10
    equatorial = abs(math.sin(inc)) <= 1e-10
    original_raan, original_argp = raan, argp

    if circular and equatorial:
        raan, argp = 0.0, 0.0
        nu = lambda_true if lambda_true is not None else original_raan + original_argp + nu
    elif circular:
        argp = 0.0
        nu = u if u is not None else original_argp + nu
    elif equatorial:
        raan = 0.0
        argp = omega_true if omega_true is not None else original_raan + original_argp

    denominator = 1 + e * math.cos(nu)
    if denominator <= 0:
        raise ValueError("true anomaly is outside the physical orbit")

    radius = p / denominator
    rp = (radius * math.cos(nu), radius * math.sin(nu), 0.0)
    speed = math.sqrt(mu / p)
    vp = (-speed * math.sin(nu), speed * (e + math.cos(nu)), 0.0)

    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    rot = (
        (cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci, sO*si),
        (sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si),
        (sw*si, cw*si, ci),
    )
    r = tuple(sum(rot[i][j] * rp[j] for j in range(3)) for i in range(3))
    v = tuple(sum(rot[i][j] * vp[j] for j in range(3)) for i in range(3))
    return r, v


def find_tof(r0, r, p, mu):
    """Algorithm 11: find time of flight between two position vectors.

    ``r0``, ``r``, and ``p`` use distance units; ``mu`` uses distance**3/time**2.
    Uses the short transfer angle (0 to pi) from the angle between the vectors.
    """
    r0, r = tuple(map(float, r0)), tuple(map(float, r))
    if len(r0) != 3 or len(r) != 3 or p <= 0 or mu <= 0:
        raise ValueError("r0 and r must be 3-vectors; p and mu must be positive")

    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    mag = lambda a: math.sqrt(dot(a, a))
    r0m, rm = mag(r0), mag(r)
    if r0m == 0 or rm == 0:
        raise ValueError("position vectors must have nonzero magnitude")

    cos_dnu = max(-1.0, min(1.0, dot(r0, r)/(r0m*rm)))
    sin_dnu = math.sqrt(max(0.0, 1 - cos_dnu**2))
    dnu = math.atan2(sin_dnu, cos_dnu)
    k = r0m*rm*(1 - cos_dnu)
    ell = r0m + rm
    m = r0m*rm*(1 + cos_dnu)
    denominator = (2*m - ell**2)*p**2 + 2*k*ell*p - k**2
    a = m*k*p/denominator if abs(denominator) > 1e-14 else math.inf
    f = 1 - rm/p*(1 - cos_dnu)
    g = r0m*rm*sin_dnu/math.sqrt(mu*p)

    if a > 1e-10:
        fdot = math.sqrt(mu/p)*math.tan(dnu/2)*((1-cos_dnu)/p - 1/r0m - 1/rm)
        cos_de = 1 - r0m/a*(1-f)
        sin_de = -r0m*fdot/math.sqrt(mu*a)
        delta_e = math.atan2(sin_de, cos_de)
        if delta_e < 0:
            delta_e += 2*math.pi
        return g + math.sqrt(a**3/mu)*(delta_e - sin_de)

    if a < -1e-10:
        cosh_dh = 1 + (f-1)*r0m/a
        if cosh_dh < 1:
            raise ValueError("invalid hyperbolic transfer geometry")
        delta_h = math.acosh(cosh_dh)
        return g + math.sqrt((-a)**3/mu)*(math.sinh(delta_h) - delta_h)

    chord = math.sqrt(max(0.0, r0m**2 + rm**2 - 2*r0m*rm*cos_dnu))
    semiperimeter = (r0m + rm + chord)/2
    return (2/3)*math.sqrt(semiperimeter**3/(2*mu))*(
        1 - ((semiperimeter-chord)/semiperimeter)**1.5
    )

