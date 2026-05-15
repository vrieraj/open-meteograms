"""Thermodynamic helpers for vertical profile calculations."""

from __future__ import annotations

import numpy as np


def compute_theta_v(temperature_c, relative_humidity, pressure_hpa):
    t_k = np.asarray(temperature_c) + 273.15
    p = np.asarray(pressure_hpa)
    rh = np.asarray(relative_humidity)
    theta = t_k * (1000.0 / p) ** 0.286
    e_s = 6.112 * np.exp(17.67 * np.asarray(temperature_c) / (np.asarray(temperature_c) + 243.5))
    e = rh / 100.0 * e_s
    r = 0.622 * e / np.maximum(p - e, 0.1)
    return theta * (1 + 0.61 * r)
