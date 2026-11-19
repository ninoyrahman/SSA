import math


def _validate_vector_angles(vector, *angles):
    if len(vector) != 3:
        raise ValueError("vector must have three components")
    if not all(math.isfinite(x) for x in (*vector, *angles)):
        raise ValueError("vector components and angles must be finite")


def _rot1(angle, vector):
    """Apply the book's passive ROT1 matrix."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return x, c * y + s * z, -s * y + c * z


def _rot3(angle, vector):
    """Apply the book's passive ROT3 matrix."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return c * x + s * y, -s * x + c * y, z


def rsw_to_eci(vector_rsw, raan, inclination, argument_latitude):
    """Transform an RSW vector to ECI using the book's Eq. (3-34).

    Applies ``ROT3(-raan) ROT1(-inclination) ROT3(-argument_latitude)``.
    Angles are in radians.
    """
    _validate_vector_angles(vector_rsw, raan, inclination, argument_latitude)
    result = _rot3(-argument_latitude, tuple(vector_rsw))
    result = _rot1(-inclination, result)
    return _rot3(-raan, result)


def eci_to_rsw(vector_eci, raan, inclination, argument_latitude):
    """Transform an ECI vector to RSW, the inverse of Eq. (3-34)."""
    _validate_vector_angles(vector_eci, raan, inclination, argument_latitude)
    result = _rot3(raan, tuple(vector_eci))
    result = _rot1(inclination, result)
    return _rot3(argument_latitude, result)


def pqw_to_rsw(vector_pqw, true_anomaly):
    """Transform a PQW vector to RSW using the book's Eq. (3-31)."""
    _validate_vector_angles(vector_pqw, true_anomaly)
    c, s = math.cos(true_anomaly), math.sin(true_anomaly)
    x, y, z = vector_pqw
    return c * x + s * y, -s * x + c * y, z


def rsw_to_pqw(vector_rsw, true_anomaly):
    """Transform an RSW vector to PQW, the inverse of Eq. (3-31)."""
    _validate_vector_angles(vector_rsw, true_anomaly)
    c, s = math.cos(true_anomaly), math.sin(true_anomaly)
    x, y, z = vector_rsw
    return c * x - s * y, s * x + c * y, z


def rsw_to_sez(vector_rsw, delta_phi, delta_lambda):
    """Transform RSW1 components to SEZ with Algorithm 49's matrix.

    Input is ``(R, S, W)`` and output is in ``(S, E, Z)`` order.
    ``delta_phi`` and ``delta_lambda`` are in
    radians.
    """
    _validate_vector_angles(vector_rsw, delta_phi, delta_lambda)
    sp, cp = math.sin(delta_phi), math.cos(delta_phi)
    sl, cl = math.sin(delta_lambda), math.cos(delta_lambda)
    r, s, w = vector_rsw
    return (sp * cl * r + sp * sl * s - cp * w,
            -sl * r + cl * s,
            cp * cl * r + cp * sl * s + sp * w)


def sez_to_rsw(vector_sez, delta_phi, delta_lambda):
    """Transform ``(S, E, Z)`` components to RSW1, inverse of Algorithm 49."""
    _validate_vector_angles(vector_sez, delta_phi, delta_lambda)
    sp, cp = math.sin(delta_phi), math.cos(delta_phi)
    sl, cl = math.sin(delta_lambda), math.cos(delta_lambda)
    south, east, zenith = vector_sez
    return (sp * cl * south - sl * east + cp * cl * zenith,
            sp * sl * south + cl * east + cp * sl * zenith,
            -cp * south + sp * zenith)


def ntw_to_eci(vector_ntw, raan, inclination, argument_latitude, phi_c=0.0):
    """Transform an NTW vector to ECI using the book's Eq. (3-33).

    Applies ``R3(-raan) R1(-inclination) R3(-argument_latitude) R3(phi_c)``
    to ``vector_ntw``. All angles are in radians; returns a 3-tuple.
    """
    if len(vector_ntw) != 3:
        raise ValueError("vector_ntw must have three components")
    if not all(math.isfinite(x) for x in (*vector_ntw, raan, inclination,
                                           argument_latitude, phi_c)):
        raise ValueError("vector components and angles must be finite")

    transformed = _rot3(phi_c, tuple(vector_ntw))
    transformed = _rot3(-argument_latitude, transformed)
    transformed = _rot1(-inclination, transformed)
    return _rot3(-raan, transformed)
