from enum import Enum
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


class constants(Enum):
    # Recommendation Types
    POINT = "Single point generation"
    WEIGHTED_AXES = "Single point generation with weighted axes"
    FUNCTION_BASED = "Function-based generation"

    # Optimization Types
    MAX_PREF = "Maximum preference optimization"
    WEIGHTED_SUM = "Weighted sum optimization"
    GAUSSIAN_PROCESS = "Gaussian process regression"
    

def slerp(v0, v1, num, t0=0, t1=1):
    """Spherical linear interpolation between two vectors.
    :param v0: start vector
    :param v1: end vector
    :param num: number of interpolation steps
    :param t0: start interpolation value
    :param t1: end interpolation value
    :return: interpolated vectors
    """
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2

    t = np.linspace(t0, t1, num)

    v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))

    return v3

def display_generated_points(generated_points: Tensor, user_profile: Tensor):
    """
    Display the generated points in a 3D plot. The points are assumed to be in 3D.
    :param generated_points: Tensor of shape (n_points, 3) containing the generated points.
    :return: -
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for point in generated_points:
        ax.scatter(point[0], point[1], point[2])

    ax.scatter(user_profile[0], user_profile[1], user_profile[2], marker='x', color='red', s=50)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title("Generated points on surface of sphere")
    plt.show()