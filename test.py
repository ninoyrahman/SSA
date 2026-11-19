import math

from util import kep_eqn_e


def example_2_1():
    """Solve Example 2-1: Kepler's equation for M=235.4 degrees, e=0.4.

    Returns:
        float: Eccentric anomaly in degrees, in the range [0, 360).
    """
    mean_anomaly = math.radians(235.4)
    eccentric_anomaly = kep_eqn_e(mean_anomaly, 0.4)
    return math.degrees(eccentric_anomaly) % 360
