import math
from typing import List, Union, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]


def rpy_to_rotation_matrix(angles: List[float]) -> Optional[np.ndarray]:
    """
    Converts rpy to rotation matrix.
    :param angles:
    :return:
    """
    roll, yaw, pitch = angles
    axes_angles = [(z, -roll), (x, pitch), (y, -yaw)]

    axis, angle = axes_angles[0]
    rotation_matrix = axis_to_rotation_matrix(axis, np.deg2rad(angle))

    for axis, angle in axes_angles[1:]:
        rotation_matrix = np.matmul(axis_to_rotation_matrix(axis, np.deg2rad(angle)), rotation_matrix)

    return rotation_matrix


def rotation_matrix_to_rotation_vector(rotation_matrix: np.ndarray) -> Union[np.ndarray, List[float]]:
    rotation = R.from_matrix(rotation_matrix)

    return rotation.as_rotvec()


def rotation_vector_to_rotation_matrix(rotation_vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    rotation = R.from_rotvec(rotation_vector)

    return rotation.as_matrix()


def rotation_matrix_to_ryp(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    AFLW dataset standard.
    :param rotation_matrix:
    :return: [roll, yaw, pitch]
    """
    roll = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[1, 1])
    yaw = np.arctan2(-rotation_matrix[0, 2], rotation_matrix[2, 2])
    pitch = -np.arcsin(rotation_matrix[1, 2])

    return np.rad2deg([roll, yaw, pitch])


def z_rotation_matrix(angle: float) -> np.ndarray:
    """
    Creates a rotation martix that is rotated by given angles, assuming it is roll.
    :param angle:
    :return: rotation_matrix
    """
    rad_angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(rad_angle), -np.sin(rad_angle), 0],
                                [np.sin(rad_angle),  np.cos(rad_angle), 0],
                                [0,                  0,                 1]])
    return rotation_matrix


def axis_to_rotation_matrix(axis: List[int], theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_rotation_matrix(rotation_matrix: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates rotation by an angle, assuming it is roll angles.
    :param rotation_matrix:
    :param angle:
    :return: rotation_matrix
    """
    rotation_matrix = np.matmul(z_rotation_matrix(angle), rotation_matrix)
    return rotation_matrix


def shift_circle(angle_value: float) -> float:
    """
    Keeps the value between -180 and 180 by rotating it around circle if it is larger/smaller.
    :param angle_value:
    :return: angle_value
    """
    if angle_value < -180:
        return 360 + angle_value
    elif angle_value > 180:
        return angle_value - 360
    else:
        return angle_value


def _ryp_to_rot_mat(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

    return rotation_matrix