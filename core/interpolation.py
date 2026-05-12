"""Vertical interpolation utilities."""

from __future__ import annotations

import numpy as np


def interpolate_pressure_level(pressure, values, target_pressure):
    pressure = np.asarray(pressure)
    values = np.asarray(values)
    order = np.argsort(pressure)
    return np.interp(target_pressure, pressure[order], values[order])


def interpolate_height_level(height, values, target_height):
    height = np.asarray(height)
    values = np.asarray(values)
    order = np.argsort(height)
    return np.interp(target_height, height[order], values[order])
