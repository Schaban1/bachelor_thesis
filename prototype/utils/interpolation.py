import numpy as np
import torch

def slerp(v0, v1, num, t0=0, t1=1):
    """Spherical linear interpolation between two vectors.
    :param v0: start vector
    :param v1: end vector
    :param num: number of interpolation steps
    :param t0: start interpolation value
    :param t1: end interpolation value
    :return: interpolated vectors
    """
    v0 = v0.numpy()  # v0.detach().cpu().numpy()
    v1 = v1.numpy()  # v1.detach().cpu().numpy()

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