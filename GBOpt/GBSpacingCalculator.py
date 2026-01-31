# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary spacing calculator utilities."""

import warnings
from fractions import Fraction
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class GBSpacingCalculatorError(Exception):
    """Base class for Exceptions in the GBSpacingCalculator class."""
    pass


class GBSpacingCalculator:
    """
    Class to calculate periodic spacing and repeat distances for grain boundaries
    given the 5 degrees of freedom (misorientation parameters).

    This class provides methods to evaluate whether a grain boundary configuration
    is feasible before attempting to create the full structure.
    """

    @staticmethod
    def approximate_rotation_matrix_as_int(
        m: np.ndarray, precision: float = 5
    ) -> np.ndarray:
        """
        Approximate a rotation matrix in integer format given the original matrix and
        the desired precision.

        :param m: The matrix to approximate
        :param precision: Decimal precision to use during calculations, defaults to 5
        :return: Integer approximation of the rotation matrix m
        """
        # first round the matrix to the desired precision
        R0 = np.linalg.norm(Rotation.from_matrix(m).as_rotvec(degrees=True))

        def gcd_reduce(matrix):
            gcds = np.gcd.reduce(matrix, axis=1)
            return matrix / gcds[:, np.newaxis]

        def get_angle(matrix):
            return np.linalg.norm(
                Rotation.from_matrix(
                    matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
                ).as_rotvec(degrees=True)
            )

        def get_magnitude_sum(matrix):
            abs_m = np.abs(matrix)
            non_zero_elements = abs_m[abs_m > 0]
            log_magnitudes = np.log(non_zero_elements)
            return np.sum(log_magnitudes)

        def calculate_best_approx(metrics1, metrics2, m1, m2):
            diffs = [metrics1["angle"], metrics2["angle"]]
            keys = list(metrics1.keys())

            # Normalize each metric. Note that the if statement essentially only catches
            # when both matrices gives essentially the same rotation as the original
            # (as calculated by the get_angle function above).
            metric1_norms = np.array(
                [
                    metrics1[key] / max(metrics1[key], metrics2[key])
                    if not max(metrics1[key], metrics2[key]) == 0
                    else 0
                    for key in keys
                ]
            )
            metric2_norms = np.array(
                [
                    metrics2[key] / max(metrics1[key], metrics2[key])
                    if not max(metrics1[key], metrics2[key]) == 0
                    else 0
                    for key in keys
                ]
            )

            # These weights *seem* to work, but there should be a better way to
            # determine these (these were determine through trial and error for the
            # R_right matrix for the misorientation matrix of [0.3, 0.4, 0.5, 0.6, 0.7])
            # In a general sense, placing most of the weighting on the magnitude, then
            # most of the rest on the condition, with the rest on the angle should give
            # a good representation, at least based on the generated matrix mentioned.
            weights = {"angle": 0.1, "condition": 0.3, "magnitude": 0.6}
            weights = np.array([weights[key] for key in keys])  # keep order the same
            metric1_overall = np.sum(metric1_norms * weights) / np.sum(weights)
            metric2_overall = np.sum(metric2_norms * weights) / np.sum(weights)

            if metric1_overall < metric2_overall:
                return m1, diffs[0]
            else:
                return m2, diffs[1]

        # Approximation with the least common multiple of denominators in their
        # fraction representation
        m_as_fractions = np.vectorize(
            lambda val: Fraction(val).limit_denominator(10**precision)
        )(m)
        denominators = np.array(
            [[f.denominator for f in row] for row in m_as_fractions]
        )
        scaling_factors = np.array([np.lcm.reduce(row) for row in denominators])
        scaled_matrix = m * scaling_factors[:, np.newaxis]
        approx_m_from_fractions = gcd_reduce(np.round(scaled_matrix).astype(int))
        approx_m_from_fractions_metrics = {
            "angle": abs(R0-get_angle(approx_m_from_fractions)),
            "condition": np.linalg.cond(approx_m_from_fractions),
            "magnitude": get_magnitude_sum(approx_m_from_fractions)
        }

        # Approximation by taking the ratio of the row values divided by the smallest
        # values, scaling these ratios up by 10**precision, truncating the values,
        # then simplifying.
        min_by_row_excluding_0 = np.ma.amin(
            np.ma.masked_less(np.abs(m), 10**-precision), axis=1).data
        m_ratio = m / min_by_row_excluding_0[:, np.newaxis]  # ratios of values to mins
        m_rounded = np.round(m_ratio, precision)  # round to the desired precision
        m_scaled = (10**precision * m_rounded).astype(int)  # scale by 10**precision
        approx_m_from_scaling = gcd_reduce(m_scaled)
        approx_m_from_scaling_metrics = {
            "angle": abs(R0-get_angle(approx_m_from_scaling)),
            "condition": np.linalg.cond(approx_m_from_scaling),
            "magnitude": get_magnitude_sum(approx_m_from_scaling)
        }

        result, diff = calculate_best_approx(
            approx_m_from_fractions_metrics,
            approx_m_from_scaling_metrics,
            approx_m_from_fractions,
            approx_m_from_scaling
        )

        if diff > 0.5:
            warnings.warn(
                "Approximated rotation matrix error is greater than 0.5 degrees.")
        return result.astype(int)

    @staticmethod
    def calculate_rotation_matrices(
        misorientation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the rotation matrices for left and right grains from misorientation.

        :param misorientation: Array containing the misorientation and inclination Euler
            angles (5 elements: alpha, beta, gamma, theta, phi). Misorientation is the
            first three (ZXZ Euler angles), and inclination is the last two.
        :return: Tuple of (R_left, R_right) rotation matrices
        """
        Rmis = Rotation.from_euler("ZXZ", misorientation[:3]).as_matrix()
        Rincl = (
            Rotation.from_euler("z", misorientation[4])
            * Rotation.from_euler("y", misorientation[3])
        ).as_matrix()

        R_left = Rincl
        R_right = np.dot(Rmis, Rincl)

        return R_left, R_right

    @classmethod
    def calculate_periodic_spacing(
        cls,
        a0: float,
        misorientation: np.ndarray,
        x_dim_min: float,
        threshold: float = None
    ) -> Dict:
        """
        Calculate the periodic spacing based on the rotation matrix and misorientation.

        :param a0: Crystal lattice parameter (Angstroms)
        :param misorientation: Array containing the misorientation and inclination Euler
            angles (5 elements: alpha, beta, gamma, theta, phi)
        :param x_dim_min: Minimum size of one grain in the x dimension (Angstroms)
        :param threshold: The maximum allowed value that any spacing can take. Default
            is 15 * a0.
        :return: Dict containing:
            - 'x': Dict with 'left' and 'right' periodic spacing in x direction
            - 'y': Periodic spacing in y direction
            - 'z': Periodic spacing in z direction
            - 'left_x': Total length of left grain in x direction
            - 'right_x': Total length of right grain in x direction
            - 'x_dim': Total x dimension (left_x + right_x)
            - 'is_periodic': Dict indicating if y and z directions are periodic
        """
        import math

        if threshold is None:
            threshold = a0 * 15

        # Calculate rotation matrices
        R_left, R_right = cls.calculate_rotation_matrices(misorientation)

        # Approximate the rotation matrices as integers
        R_left_approx = cls.approximate_rotation_matrix_as_int(R_left).astype(object)
        R_right_approx = cls.approximate_rotation_matrix_as_int(R_right).astype(object)

        # The periodic distance in each direction is the lattice parameter multiplied by
        # norm of the Miller indices in that direction. This is determined using the
        # usual formula for the interplanar spacing: d = a / sqrt(h**2+k**2+l**2). The
        # square of the denominator here is the number of planes needed before
        # periodicity. Thus, if we multiply that distance by the interplanar spacing we
        # will get the interplanar spacing. This simplifies to
        # (a0**2/d**2)*d = a0**2/d --> spacing = a0 * sqrt(h**2+k**2+l**2)
        spacing_left = {
            axis: a0 * np.linalg.norm(vec)
            for axis, vec in zip(["x", "y", "z"], R_left_approx)
        }
        spacing_right = {
            axis: a0 * np.linalg.norm(vec)
            for axis, vec in zip(["x", "y", "z"], R_right_approx)
        }

        spacing = {"x": {"left": spacing_left["x"], "right": spacing_right["x"]}}
        left_x = math.ceil(x_dim_min / spacing["x"]["left"]) * spacing["x"]["left"]
        right_x = math.ceil(x_dim_min / spacing["x"]["right"]) * spacing["x"]["right"]
        x_dim = left_x + right_x

        spacing["left_x"] = left_x
        spacing["right_x"] = right_x
        spacing["x_dim"] = x_dim

        spacing.update(
            {
                axis: max(spacing_left[axis], spacing_right[axis])
                for axis in ["y", "z"]
            }
        )

        # Track periodicity
        is_periodic = {"y": True, "z": True}
        warnings.simplefilter("once", UserWarning)
        for key, val in list(spacing.items()):
            if key in ['y', 'z']:
                if threshold < val:
                    spacing[key] = threshold
                    is_periodic[key] = False
                    warnings.warn("Resulting boundary is non-periodic.")
        warnings.simplefilter("default", UserWarning)

        spacing["is_periodic"] = is_periodic

        return spacing
