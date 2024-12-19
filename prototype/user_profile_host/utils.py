import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


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
        """
        helper function to spherically interpolate two arrays v1 v2
        :param t: interpolation value
        :param v0: start vector
        :param v1: end vector
        :param DOT_THRESHOLD: threshold for dot product
        :return: interpolated vector
        """
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


def display_generated_points(generated_points: Tensor, user_profile: Tensor, x_bounds=(-1, 1), y_bounds=(-1, 1),
                             z_bounds=(-1, 1)):
    """
    Display the generated points in a 3D plot. The points are assumed to be in 3D.
    :param generated_points: Tensor of shape (n_points, 3) containing the generated points.
    :param user_profile: The user profile in 3D.
    :param x_bounds: The bounds for the x-axis. Default is (-1, 1).
    :param y_bounds: The bounds for the y-axis. Default is (-1, 1).
    :param z_bounds: The bounds for the z-axis. Default is (-1, 1).
    :return: -
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)

    for point in generated_points:
        ax.scatter(point[0], point[1], point[2])

    ax.scatter(user_profile[0], user_profile[1], user_profile[2], marker='x', color='red', s=50)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title("Generated points on surface of sphere")
    plt.show()


def display_generated_points_user_profile_2d(low_d_embeddings: Tensor, low_d_user_profile: Tensor, preferences: Tensor,
                                             compression_technique: str = "PCA", save_path: str = None):
    """
    Display the generated points in a 2D plot. The points are assumed to be in 2D.
    :param low_d_embeddings: Tensor of shape (n_points, 2) containing the generated points,
        i.e. points must be two-dimensional.
    :param low_d_user_profile: The user profile in 2D.
    :param preferences: (n_points,) preferences of the user concerning the generated points.
    :param compression_technique: The technique used to compress the points to 2D. Either "PCA" or "t-SNE".
    :param save_path: The path to save the plot to. Has to be a string and include "/" at the end. Default is None,
        hence, image is not saved.
    :return: -
    """
    cmap = plt.get_cmap('coolwarm')
    fig = plt.figure(figsize=(6, 5))
    p = plt.scatter(low_d_embeddings[:, 0], low_d_embeddings[:, 1], c=preferences, cmap=cmap)
    plt.colorbar(mappable=p)
    plt.scatter(low_d_user_profile[0], low_d_user_profile[1], c='black', s=150, marker='+', linewidth=3.5,
                label='User Profile')
    plt.legend()
    plt.title("Generated points up to now compressed to 2D using " + compression_technique)
    if save_path is not None:
        plt.savefig(save_path + f"generated_points_compressed_using_{compression_technique}.svg",
                    dpi=300, bbox_inches='tight', format='svg')
    plt.show()
