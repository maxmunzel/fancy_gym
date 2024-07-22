import numpy as np


# joint constraints for Franka robot
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

q_dot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
q_torque_max = np.array([90.0, 90.0, 90.0, 90.0, 12.0, 12.0, 12.0])
#
desired_rod_quat = np.array([0.0, 1.0, 0.0, 0.0])


def skew(x):
    """
    Returns the skew-symmetric matrix of x
    param x: 3x1 vector
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    param curr_quat: current quaternion
    param des_quat: desired quaternion
    return: difference between current quaternion and desired quaternion
    """
    return (
        curr_quat[0] * des_quat[1:]
        - des_quat[0] * curr_quat[1:]
        - skew(des_quat[1:]) @ curr_quat[1:]
    )


def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(min(0.9999, abs(p @ q)))
    return theta


def rot_to_quat(theta, axis):
    """
    Converts rotation angle along an axis to quaternion
    param theta: rotation angle (rad)
    param axis: rotation axis
    return: quaternion
    """
    quant = np.zeros(4)
    quant[0] = np.sin(theta / 2.0)
    quant[1:] = np.cos(theta / 2.0) * axis
    return quant


def affine_matrix_to_xpos_and_xquat(matrix: np.ndarray):
    """
    takes a 4x4 matrix and returns a vector of the translation and quaternion as used by mujoco
    """
    assert matrix.shape == (4, 4)
    x, y, z = matrix[:3, 3].flatten()

    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to a quaternion
    qw = 0.5 * np.sqrt(
        1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    )
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

    # Concatenate the translation and quaternion

    return np.array([x, y, z]), np.array([qw, qx, qy, qz])
